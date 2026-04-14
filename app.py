import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import plotly
from plotly_calplot import calplot
import requests
import logging
from scipy.optimize import root_scalar
from parse_wec_decimated_log import parse_putty_log
import os
from pathlib import Path

colors = plotly.colors.DEFAULT_PLOTLY_COLORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# directory to cache raw WEC text files
WEC_TEXT_CACHE = Path(".cache/wec")
PWRSYS_TEXT_CACHE = Path(".cache/pwrsys")
DATA_DIR = Path("output/data")

# THREDDS-based NDBC data fetching
THREDDS_FILESERVER = "https://dods.ndbc.noaa.gov/thredds/fileServer"
NDBC_STDMET_CACHE = Path(".cache/ndbc/stdmet")
NDBC_SWDEN_CACHE = Path(".cache/ndbc/swden")
SPECTRAL_BUOY = "44014"  # Only buoy for which spectral data is fetched

# THREDDS variable name -> legacy short name used by downstream plotting code
_STDMET_VAR_RENAME = {
    "wind_dir": "WDIR",
    "wind_spd": "WSPD",
    "gust": "GST",
    "wave_height": "WVHT",
    "dominant_wpd": "DPD",
    "average_wpd": "APD",
    "mean_wave_dir": "MWD",
    "air_pressure": "PRES",
    "air_temperature": "ATMP",
    "sea_surface_temperature": "WTMP",
    "dewpt_temperature": "DEWP",
    "visibility": "VIS",
    "water_level": "TIDE",
}


def _ensure_wec_text_cache_dir() -> None:
    WEC_TEXT_CACHE.mkdir(parents=True, exist_ok=True)


def _wec_text_cache_file(date_str: str) -> Path:
    return WEC_TEXT_CACHE / f"{date_str}.wec.dec.10.log"


def _ensure_pwrsys_text_cache_dir() -> None:
    PWRSYS_TEXT_CACHE.mkdir(parents=True, exist_ok=True)


def _pwrsys_text_cache_file(date_str: str) -> Path:
    return PWRSYS_TEXT_CACHE / f"{date_str}.pwrsys.log"


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _thredds_stdmet_urls(buoy_id: str, start_year: int, current_year: int) -> list[dict]:
    """Return ordered list of THREDDS stdmet file descriptors for a buoy/date window."""
    urls = []
    for year in range(start_year, current_year + 1):
        urls.append({
            "url": f"{THREDDS_FILESERVER}/data/stdmet/{buoy_id}/{buoy_id}h{year}.nc",
            "type": "yearly",
            "local": NDBC_STDMET_CACHE / buoy_id / f"{buoy_id}h{year}.nc",
        })
    # Rolling file updated daily; always re-download
    urls.append({
        "url": f"{THREDDS_FILESERVER}/data/stdmet/{buoy_id}/{buoy_id}h9999.nc",
        "type": "current",
        "local": NDBC_STDMET_CACHE / buoy_id / f"{buoy_id}h9999.nc",
    })
    return urls


def _thredds_swden_urls(buoy_id: str, start_year: int, current_year: int) -> list[dict]:
    """Return ordered list of THREDDS swden file descriptors for a buoy/date window."""
    urls = []
    for year in range(start_year, current_year + 1):
        urls.append({
            "url": f"{THREDDS_FILESERVER}/data/swden/{buoy_id}/{buoy_id}w{year}.nc",
            "type": "yearly",
            "local": NDBC_SWDEN_CACHE / buoy_id / f"{buoy_id}w{year}.nc",
        })
    # Rolling file updated daily; always re-download
    urls.append({
        "url": f"{THREDDS_FILESERVER}/data/swden/{buoy_id}/{buoy_id}w9999.nc",
        "type": "current",
        "local": NDBC_SWDEN_CACHE / buoy_id / f"{buoy_id}w9999.nc",
    })
    return urls


def _download_thredds_file(url: str, local_path: Path) -> bool:
    """Download a single file from the THREDDS fileServer to local cache.
    Returns True on success, False if the file is not found or the download fails.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code == 404:
            logger.debug(f"Not found on THREDDS: {url}")
            return False
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 17):
                f.write(chunk)
        logger.info(f"Downloaded {url} -> {local_path}")
        return True
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def _open_cached_netcdf(local_path: Path) -> xr.Dataset:
    """Open cached NetCDF with engine fallbacks.

    THREDDS fileServer often serves classic NetCDF3 files, which are not
    readable by h5netcdf. Try scipy first, then other engines.
    """
    # NDBC period variables advertise "seconds" units; keeping them numeric avoids
    # timedelta coercion warnings and matches downstream expectations.
    open_kwargs = {"mask_and_scale": True, "decode_timedelta": False}
    errors = []
    for engine in ("scipy", "h5netcdf", "netcdf4"):
        try:
            return xr.open_dataset(local_path, engine=engine, **open_kwargs)
        except Exception as e:
            errors.append(f"{engine}: {e}")

    try:
        return xr.open_dataset(local_path, **open_kwargs)
    except Exception as e:
        errors.append(f"auto: {e}")

    raise OSError("; ".join(errors))


def _adapt_ndbc_stdmet(ds: xr.Dataset) -> xr.Dataset:
    """Rename THREDDS stdmet variable names to legacy short names and add dir_diff.
    Called by load_cached_data before returning ds_ndbc to downstream consumers.
    """
    rename = {k: v for k, v in _STDMET_VAR_RENAME.items() if k in ds}
    ds = ds.rename(rename)
    if "MWD" in ds and "WDIR" in ds:
        ds["dir_diff"] = np.abs(ds["MWD"] - ds["WDIR"]) % 180
        ds["dir_diff"].attrs["units"] = "\u00b0"
        ds["dir_diff"].attrs["long_name"] = "Direction Difference"
    return ds


def _adapt_ndbc_spectral(ds: xr.Dataset) -> xr.Dataset:
    """Rename spectral_wave_density -> spectral_density for downstream compatibility."""
    if "spectral_wave_density" in ds:
        ds = ds.rename({"spectral_wave_density": "spectral_density"})
    return ds


def fetch_ndbc(
    buoy_id: str = "44014", start_date: datetime = None, max_retries: int = 1
) -> xr.Dataset:

    now = datetime.now().date()
    if start_date is None:
        start_date = now - timedelta(days=7)
    
    start_year = start_date.year
    current_year = now.year
    num_days = (now - start_date).days + 1
    
    logger.info(
        f"Fetching NDBC stdmet for buoy {buoy_id} from {start_date} ({num_days} days)"
    )

    url_infos = _thredds_stdmet_urls(buoy_id, start_year, current_year)
    dsl = []
    for info in url_infos:
        local = info["local"]
        # Yearly files are cached permanently; rolling current file is always refreshed
        if info["type"] == "current" or not local.exists():
            if not _download_thredds_file(info["url"], local):
                continue
        if not local.exists():
            continue
        try:
            ds1 = _open_cached_netcdf(local)
            # Some files carry tiny lat/lon coordinate differences across years.
            # Collapse singleton geo dims per file before concat to avoid sparse
            # outer-joined latitude/longitude grids.
            if "latitude" in ds1.dims and "longitude" in ds1.dims:
                ds1 = ds1.isel(latitude=0, longitude=0)
            dsl.append(ds1)
            logger.info(f"Loaded {local.name}: {len(ds1.time)} records")
        except Exception as e:
            logger.warning(f"Failed to open {local}: {e}")

    if not dsl:
        logger.error(f"No NDBC stdmet data for buoy {buoy_id}")
        return xr.Dataset()

    ds = xr.concat(dsl, dim="time").sortby("time").drop_duplicates("time")
    ds = ds.sel(time=slice(pd.Timestamp(start_date), pd.Timestamp(now)))
    ds = ds.expand_dims("buoy").assign_coords(buoy=[buoy_id])
    return ds


def fetch_ndbc_spectral(
    start_date: datetime = None, max_retries: int = 1
) -> xr.Dataset:
    """Fetch NDBC spectral wave density data for buoy 44014 only."""
    buoy_id = SPECTRAL_BUOY
    now = datetime.now().date()
    if start_date is None:
        start_date = now - timedelta(days=7)

    start_year = start_date.year
    current_year = now.year
    num_days = (now - start_date).days + 1

    logger.info(
        f"Fetching NDBC swden for buoy {buoy_id} from {start_date} ({num_days} days)"
    )

    url_infos = _thredds_swden_urls(buoy_id, start_year, current_year)
    dsl = []
    for info in url_infos:
        local = info["local"]
        if info["type"] == "current" or not local.exists():
            if not _download_thredds_file(info["url"], local):
                continue
        if not local.exists():
            continue
        try:
            ds1 = _open_cached_netcdf(local)
            # Some files carry tiny lat/lon coordinate differences across years.
            # Collapse singleton geo dims per file before concat to avoid sparse
            # outer-joined latitude/longitude grids.
            if "latitude" in ds1.dims and "longitude" in ds1.dims:
                ds1 = ds1.isel(latitude=0, longitude=0)
            dsl.append(ds1)
            logger.info(f"Loaded {local.name}: {len(ds1.time)} spectral records")
        except Exception as e:
            logger.warning(f"Failed to open {local}: {e}")

    if not dsl:
        logger.error(f"No NDBC swden data for buoy {buoy_id}")
        return xr.Dataset()

    ds = xr.concat(dsl, dim="time", join="outer").sortby("time").drop_duplicates("time")
    ds = ds.sel(time=slice(pd.Timestamp(start_date), pd.Timestamp(now)))
    ds = ds.expand_dims("buoy").assign_coords(buoy=[buoy_id])
    return ds



def fetch_wec_data(start_date: datetime = None, max_retries: int = 3) -> xr.Dataset:

    logger.info(f"Fetching WEC data")

    end_time = pd.Timestamp.utcnow()
    current_date = end_time.date()

    if start_date is None:
        start_date = current_date - timedelta(days=7)

    dates_to_try = [
        start_date + timedelta(days=i)
        for i in range((current_date - start_date).days + 1)
    ]

    base_url = "https://rawdata.oceanobservatories.org/files/CP10CNSM/D00003/cg_data/dcl12/wec_decimated"

    dsl = []
    for attempt_date in dates_to_try:
        date_str = attempt_date.strftime("%Y%m%d")
        file_url = f"{base_url}/{date_str}.wec.dec.10.log"
        _ensure_wec_text_cache_dir()
        cache_file = _wec_text_cache_file(date_str)

        # prefer cached text file
        if cache_file.exists():
            try:
                logger.info(f"Loading cached WEC text for {date_str} from {cache_file}")
                text_to_parse = cache_file.read_text()
            except Exception as e:
                logger.warning(f"Failed to read cached WEC file {cache_file}: {e}")
                text_to_parse = None
        else:
            # download and cache
            try:
                logger.info(f"Fetching data for {date_str}")
                resp = requests.get(file_url, timeout=30)
                if resp.status_code == 404:
                    logger.warning(f"File not found for {date_str}")
                    text_to_parse = None
                else:
                    text_to_parse = resp.text
                    try:
                        cache_file.write_text(resp.text)
                        logger.info(f"Cached WEC text to {cache_file}")
                    except Exception as e:
                        logger.warning(f"Failed to write WEC cache {cache_file}: {e}")
            except Exception as e:
                logger.error(f"Error fetching data for {date_str}: {e}")
                text_to_parse = None

        if text_to_parse is not None and len(text_to_parse) == 0:
            logger.warning(f"Empty WEC text file for {date_str}")
            text_to_parse = None

        if text_to_parse is not None:
            logger.info(f"Parsing data for {date_str}")
            ds1, _ = parse_putty_log(text_to_parse)
            dsl.append(ds1)

    ds = xr.concat(dsl, dim="time")
    ds = ds.sortby("time")

    df = pd.read_csv(
        "Deployment1_Schedule.csv",
        header=None,
        names=["time", "Mode", "Gain"],
        index_col=False,
        sep=",",
    )
    df = df.set_index("time")
    dsg = df.to_xarray()
    dsg["time"] = pd.to_datetime(dsg["time"]).values
    dsg = dsg["Gain"].interp_like(ds["DcP"], method="zero")
    dsg = dsg.fillna(0.130)  # Default value for damping gain
    ds = xr.merge([ds, dsg])

    return ds


def fetch_pwrsys_data(start_date: datetime = None) -> xr.Dataset:
    """
    Fetch power system data (solar PV panels and wind turbines) from OOI.
    All device data is contained in a single daily log file.

    Args:
        start_date: Start date for data fetch. Defaults to 7 days ago.

    Returns:
        xr.Dataset: Power system data with variables (status, voltage, current) and dimensions (device, time)
    """

    logger.info(f"Fetching power system data")

    end_time = pd.Timestamp.utcnow()
    current_date = end_time.date()

    if start_date is None:
        start_date = current_date - timedelta(days=7)

    dates_to_try = [
        start_date + timedelta(days=i)
        for i in range((current_date - start_date).days + 1)
    ]

    base_url = (
        "https://rawdata.oceanobservatories.org/files/CP10CNSM/D00003/cg_data/pwrsys"
    )

    all_data = []
    _ensure_pwrsys_text_cache_dir()

    for attempt_date in dates_to_try:
        date_str = attempt_date.strftime("%Y%m%d")
        file_url = f"{base_url}/{date_str}.pwrsys.log"
        cache_file = _pwrsys_text_cache_file(date_str)

        text_to_parse = None

        # prefer cached text file
        if cache_file.exists():
            try:
                logger.info(
                    f"Loading cached power system text for {date_str} from {cache_file}"
                )
                text_to_parse = cache_file.read_text()
            except Exception as e:
                logger.warning(
                    f"Failed to read cached power system file {cache_file}: {e}"
                )
                text_to_parse = None
        else:
            # download and cache
            try:
                logger.info(f"Fetching power system data for {date_str}")
                resp = requests.get(file_url, timeout=30)

                if resp.status_code == 404:
                    logger.debug(f"Power system file not found for {date_str}")
                    text_to_parse = None
                elif resp.status_code != 200:
                    logger.warning(
                        f"Failed to fetch power system data for {date_str}: HTTP {resp.status_code}"
                    )
                    text_to_parse = None
                else:
                    text_to_parse = resp.text
                    try:
                        cache_file.write_text(resp.text)
                        logger.info(f"Cached power system text to {cache_file}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to write power system cache {cache_file}: {e}"
                        )
            except Exception as e:
                logger.warning(f"Error fetching power system data for {date_str}: {e}")
                text_to_parse = None

        if text_to_parse is not None and len(text_to_parse) == 0:
            logger.debug(f"Empty power system data for {date_str}")
            text_to_parse = None

        if text_to_parse is not None:
            # Parse the data
            df = _parse_pwrsys_log(text_to_parse)
            if df is not None and len(df) > 0:
                all_data.append(df)
                logger.info(f"Successfully parsed power system data for {date_str}")

    if not all_data:
        logger.error("No power system data was successfully fetched")
        return xr.Dataset()

    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=False)
    df_combined = df_combined.sort_index()

    # Extract device names and variable types (include batteries)
    devices = ["pv1", "pv2", "pv3", "pv4", "wt1", "wt2", "bt1", "bt2", "bt3", "bt4"]
    var_map = {
        "status": "status",
        "voltage": "voltage",
        "current": "current",
        "temperature": "temp",
    }

    # Determine which devices have any data in the combined dataframe
    device_coords = [
        d
        for d in devices
        if any(col.startswith(f"{d}_") for col in df_combined.columns)
    ]

    # Build 2D arrays for each variable type: (device, time)
    data_vars = {}
    time_len = len(df_combined.index)

    for var_name, col_suffix in var_map.items():
        var_data = []
        for device in device_coords:
            col_name = f"{device}_{col_suffix}"
            if col_name in df_combined.columns:
                var_data.append(df_combined[col_name].values)
            else:
                # fill missing device variable with NaNs
                var_data.append(np.full(time_len, np.nan))

        if any(~np.isnan(np.array(var_data)).all(axis=1)) or True:
            # Stack into 2D array (device, time)
            data_vars[var_name] = (["device", "time"], np.array(var_data))

    if not data_vars or not device_coords:
        logger.error("No device data found in parsed power system log")
        return xr.Dataset()

    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars, coords={"device": device_coords, "time": df_combined.index.values}
    )

    # Assign device types
    gtype = {"pv": "solar", "wt": "wind", "bt": "battery"}
    ds = ds.assign_coords(
        gtype=(
            "device",
            [gtype.get(d[:2], "unknown") for d in device_coords],
            {"long_name": "Generation type"},
        ),
    )

    # Add units and long names
    ds["time"].attrs["long_name"] = "Time"
    ds["status"].attrs["units"] = "-"
    ds["status"].attrs["long_name"] = "Device Status"
    ds["voltage"].attrs["units"] = "V"
    ds["voltage"].attrs["long_name"] = "Voltage"
    ds["current"] = ds["current"] / 1e3
    ds["current"].attrs["units"] = "A"
    ds["current"].attrs["long_name"] = "Current"
    ds["temperature"].attrs["units"] = "°C"
    ds["temperature"].attrs["long_name"] = "Battery temperature"

    ds1 = ds.where(ds["gtype"] == "battery").dropna(dim="device", how="all")
    ds["ocv"] = ds1["voltage"] + 0.368 * ds1["current"]
    ds["soc"] = (ds["ocv"] - 23.16) * 100 / 2.4
    ds["soc"].attrs["units"] = "%"
    ds["soc"].attrs["long_name"] = "State of charge"

    ds = ds.sortby("time")

    return ds


def _parse_pwrsys_log(content: str) -> pd.DataFrame:
    """
    Parse power system log data from OOI containing all devices.
    Log format: YYYY/MM/DD HH:MM:SS.mmm PwrSys ... pv1 status voltage current pv2 status voltage current ... wt1 status voltage current ...

    For each device: status (integer), voltage (float), current (float)

    Args:
        content: Raw text content from the log file

    Returns:
        pd.DataFrame: Parsed data with time index and columns for each device
    """

    lines = content.strip().split("\n")

    # Skip header lines (those starting with #)
    data_lines = [line for line in lines if line.strip() and not line.startswith("#")]

    if not data_lines:
        logger.warning(f"No data lines found in power system log")
        return None

    try:
        data = []
        devices_of_interest = [
            "pv1",
            "pv2",
            "pv3",
            "pv4",
            "wt1",
            "wt2",
            "bt1",
            "bt2",
            "bt3",
            "bt4",
        ]

        for line in data_lines:
            parts = line.split()

            if len(parts) < 3:
                continue

            try:
                # Parse timestamp: first two tokens are date and time
                timestamp_str = f"{parts[0]} {parts[1]}"
                timestamp = pd.to_datetime(timestamp_str)

                # Find each device in the line and extract its values
                row_data = {"time": timestamp}

                for device in devices_of_interest:
                    try:
                        device_idx = parts.index(device)

                        # Batteries report: temp, voltage, current (mA)
                        if device.startswith("bt"):
                            if device_idx + 3 < len(parts):
                                temp = float(parts[device_idx + 1])
                                voltage = float(parts[device_idx + 2])
                                current = float(parts[device_idx + 3])

                                row_data[f"{device}_temp"] = temp
                                row_data[f"{device}_voltage"] = voltage
                                row_data[f"{device}_current"] = current
                        else:
                            # pv and wt report: status, voltage, current
                            if device_idx + 3 < len(parts):
                                status = int(parts[device_idx + 1])
                                voltage = float(parts[device_idx + 2])
                                current = float(parts[device_idx + 3])

                                row_data[f"{device}_status"] = status
                                row_data[f"{device}_voltage"] = voltage
                                row_data[f"{device}_current"] = current
                    except (ValueError, IndexError):
                        # Device not found or malformed in this line
                        pass

                # Only add row if we found at least some device data
                if len(row_data) > 1:
                    data.append(row_data)

            except (ValueError, IndexError):
                logger.debug(f"Could not parse line: {line}")
                continue

        if not data:
            return None

        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(data)
        df.set_index("time", inplace=True)

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        logger.error(f"Error parsing power system log: {e}")
        return None


def resample_and_combine(ds_wec, dsl, freq="1H"):
    ds1 = ds_wec.resample(time=freq).mean()

    dstm = []
    for dsi in dsl:
        dsi = dsi.dropna("time", how="all").resample(time=freq).mean()
        dstm.append(dsi)

    ds0 = xr.merge([ds1] + dstm)
    ds0 = ds0.sel(time=slice(ds1["time"][0], ds1["time"][-1]))

    return ds0


def make_scatter_3d(ds):
    dstp = ds.mean("buoy")[["WVHT", "Gain", "DcP"]].dropna(dim="time", how="any")
    fig = px.scatter_3d(
        dstp,
        x="WVHT",
        y="Gain",
        z="DcP",
        color="DcP",
        size="DcP",
        color_continuous_scale="reds",
        opacity=0.5,
        labels={
            "WVHT": r"Sig. wave height [m]",
            "DcP": "DC power [W]",
            "Gain": "Controller gain [As/rad]",
        },
    )
    fig.update_layout(
        # template='simple_white',
        xaxis_title="$H_{m0}$ [m]",
        scene_camera=dict(eye=dict(x=2.0, y=2.0, z=0.75)),
        height=800,
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_traces(
        customdata=dstp.time.dt.strftime("%Y-%m-%d %H:%M"),
        hovertemplate="""
        %{customdata}
        <extra></extra>
        """,
    )
    fig.update_layout(
        xaxis=dict(dtick=1),
        yaxis=dict(tickmode="array", tickvals=[0, 30, 60, 90, 120, 150, 180]),
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def make_time_hist(ds):

    # no subplot titles; show info in y-axis labels instead
    fig = make_subplots(rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.03)

    vars_to_plot = [
        {"WVHT": "#1f77b4"},
        {"WSPD": "#17becf"},
        {"DPD": "#1f77b4", "APD": "#605aff"},
        {"MWD": "#1f77b4", "WDIR": "#17becf"},
    ]
    for i, vars in enumerate(vars_to_plot):
        for var, mcolor in vars.items():
            dftp = ds[var].to_pandas()
            for col, color in zip(dftp.columns, colors):
                fig.add_trace(
                    go.Scatter(
                        x=dftp.index,
                        y=dftp[col],
                        mode="lines",
                        name=col,
                        line=dict(color=mcolor, width=0.5),
                        # hovertemplate="%{y:.1f}",
                        hoverinfo="skip",
                    ),
                    row=i + 1,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(
                    x=dftp.index,
                    y=dftp.mean(axis=1),
                    mode="lines",
                    name=f"{ds[var].attrs.get('long_name', var)}",
                    line=dict(color=mcolor, width=2),
                    hovertemplate="%{y:.1f} " + ds[var].attrs.get("units", ""),
                ),
                row=i + 1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=ds["ExP"].time,
            y=ds["ExP"].clip(0, np.infty),
            name="Export power",
            mode="lines",
            line=dict(color="#ff0eb3"),
            hovertemplate="%{y:.1f} W",
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ds["DcP"].time,
            y=ds["DcP"],
            name="DC bus power",
            mode="lines",
            line=dict(color="black"),
            hovertemplate="%{y:.1f} W",
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ds["Gain"].time,
            y=ds["Gain"],
            name="Damping gain",
            mode="lines",
            line_shape="hv",
            line=dict(color="black"),
            hovertemplate="%{y:.3f} As/rad",
        ),
        row=6,
        col=1,
    )

    pow = ds["current"] * ds["voltage"]
    pow = pow.groupby("gtype").sum().sel(gtype=["solar", "wind"])

    fig.add_trace(
        go.Scatter(
            x=pow.sel(gtype="wind").time,
            y=pow.sel(gtype="wind"),
            name="Wind",
            mode="lines",
            line=dict(color="#17becf"),
            hovertemplate="%{y:.1f} W",
        ),
        row=7,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=pow.sel(gtype="solar").time,
            y=pow.sel(gtype="solar"),
            name="Solar",
            mode="lines",
            line=dict(color="#ffb70e"),
            hovertemplate="%{y:.1f} W",
        ),
        row=7,
        col=1,
    )

    # soc = ds["soc"].mean(dim="device").clip(0, 100)

    # fig.add_trace(
    #     go.Scatter(
    #         x=soc.time,
    #         y=soc,
    #         name="State of charge",
    #         mode="lines",
    #         line=dict(color="black"),
    #         hovertemplate="%{y:.1f} W",
    #     ),
    #     row=8,
    #     col=1,
    # )

    ds1 = ds.where(ds["gtype"] == "battery").dropna(dim="device", how="all")
    ds2 = ds1["voltage"].mean(dim="device")

    fig.add_trace(
        go.Scatter(
            x=ds2.time,
            y=ds2,
            name="Battery voltage",
            mode="lines",
            line=dict(color="black"),
            hovertemplate="%{y:.1f} V",
        ),
        row=8,
        col=1,
    )

    fig.update_layout(
        height=1400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Sig. wave<br>height [m]", row=1, col=1)
    fig.update_yaxes(title_text="Wind<br>speed [m/s]", row=2, col=1)
    fig.update_yaxes(title_text="Wave<br>period [s]", row=3, col=1)
    fig.update_yaxes(title_text="Wave & wind<br>dir. [deg]", row=4, col=1)
    fig.update_yaxes(title_text="WEC<br>power [W]", range=[0, np.infty], row=5, col=1)
    fig.update_yaxes(title_text="Damping gain<br>[As/rad]", row=6, col=1)
    fig.update_yaxes(title_text="Power<br>[W]", row=7, col=1)
    # fig.update_yaxes(title_text="State of<br>charge [-]", row=8, col=1)
    fig.update_yaxes(title_text="Battery<br>voltage [V]", row=8, col=1)

    fig.update_layout(
        # title="Pioneer WEC",
        template="simple_white",
        hovermode="x unified",
        height=1000,
        margin=dict(l=60, r=40, t=100, b=50),
        showlegend=False,
        font=dict(size=10),
    )

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=1, label="1d", step="day", stepmode="backward"),
                        dict(count=2, label="2d", step="day", stepmode="backward"),
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(step="all"),
                    ]
                )
            )
        )
    )

    return fig


def make_wec_histograms(ds):
    df = ds[["DcP", "ExP"]].to_array(dim="type").to_pandas().transpose()
    df.rename(columns={"DcP": "DC", "ExP": "Export"}, inplace=True)

    fig = px.histogram(
        df,
        marginal="box",
        labels={"value": "Avg. hourly power [W]", "type": "Type"},
        color_discrete_sequence=["black", "#ff0eb3"],
        orientation="h",
        barmode="overlay",
    )

    fig.update_layout(
        xaxis_title="Count[-]",
        yaxis_title="Avg. hourly power [W]",
    )
    fig.update_yaxes(range=[0, np.infty])

    fig.update_layout(
        template="simple_white",
        # hovermode="y unified",
        # height=1000,
        # margin=dict(l=60, r=40, t=100, b=50),
        # showlegend=False,
        # font=dict(size=10),
    )

    return fig


def make_correlation_matrix(ds):
    ds0 = ds.mean("buoy")
    fig = px.scatter_matrix(
        ds0[
            ["WVHT", "WSPD", "DPD", "APD", "dir_diff", "DcP", "ExP", "Vel"]
        ].to_pandas(),
        labels={
            "WVHT": "Wave Height<br>[m]",
            "WSPD": "Wind Speed<br>[m/s]",
            "DPD": "Peak Period<br>[s]",
            "APD": "Average period<br>[s]",
            "dir_diff": "Wave/wind dir.<br>diff.[deg]",
            "DcP": "DC power<br>[W]",
            "ExP": "Export power<br>[W]",
            "Vel": "RMS velocity<br>[deg/s]",
        },
        width=800,
        height=800,
    )
    fig.update_layout(
        font=dict(size=8),
    )
    fig.update_traces(marker=dict(size=5, color="black", opacity=0.25))
    return fig


def make_jpd(ds):
    ds0 = ds.mean("buoy")
    fig = px.density_heatmap(
        ds0[["DPD", "WVHT"]],
        x="DPD",
        y="WVHT",
        #  color_continuous_scale='Viridis',
        labels={"DPD": "Peak wave period [s]", "WVHT": "Sig. wave height [m]"},
        marginal_x="histogram",
        marginal_y="histogram",
    )

    return fig


def make_power_matrix(ds):
    ds0 = ds.mean("buoy")
    fig = px.density_heatmap(
        ds0[["DPD", "WVHT", "DcP"]],
        x="DPD",
        y="WVHT",
        z="DcP",
        color_continuous_scale="Reds",
        histfunc="avg",
        labels={
            "DPD": "Peak wave period [s]",
            "WVHT": "Sig. wave height [m]",
            "DcP": "DC power [W]",
        },
    )

    fig.add_trace(
        go.Scatter(
            x=ds0["DPD"],
            y=ds0["WVHT"],
            mode="markers",
            marker=dict(
                color="black",  # Set point color to black
                size=5,  # Set point size
                opacity=0.25,  # Set point opacity
            ),
            hoverinfo="skip",
        )
    )

    return fig


def _compute_wavelength_intermediate_depth(period_s, depth_m):
    g = 9.81

    def solve_wavenumber(period):
        if not np.isfinite(period) or period <= 0:
            return np.nan

        omega = 2 * np.pi / period

        def dispersion_residual(k):
            return g * k * np.tanh(k * depth_m) - omega**2

        k_low = 1e-8
        k_high = max(omega**2 / g, k_low * 10)

        while dispersion_residual(k_high) <= 0 and k_high < 1e3:
            k_high *= 2

        try:
            result = root_scalar(
                dispersion_residual,
                bracket=[k_low, k_high],
                method="brentq",
                xtol=1e-12,
                rtol=1e-10,
                maxiter=100,
            )
            if result.converged and result.root > 0:
                return result.root
        except ValueError:
            pass

        # Fallback to deep-water approximation if solver fails
        return omega**2 / g

    periods = np.asarray(period_s, dtype=float)
    k = np.array([solve_wavenumber(period) for period in periods], dtype=float)
    wavelength = 2 * np.pi / k
    return wavelength


def make_power_vs_wave_slope(ds, depth_m=50.0):
    ds0 = ds.mean("buoy")
    df = ds0[["WVHT", "APD", "DcP"]].to_dataframe().dropna()
    df = df[df["APD"] > 0]

    # Intermediate-depth dispersion relation:
    # omega^2 = g k tanh(k h), lambda = 2pi / k
    df["wavelength"] = _compute_wavelength_intermediate_depth(
        df["APD"].to_numpy(dtype=float), depth_m
    )
    df["wave_slope"] = df["WVHT"] / df["wavelength"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["wave_slope", "DcP"])
    df = df[df["wave_slope"] > 0]

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Scatter(
            x=df["wave_slope"],
            y=df["DcP"],
            mode="markers",
            marker=dict(
                size=4,
                color=df["WVHT"],
                colorscale="Viridis",
                opacity=0.5,
                colorbar=dict(title="Sig. wave height [m]"),
            ),
            customdata=np.column_stack(
                [
                    df.index.strftime("%Y-%m-%d %H:%M"),
                    df["WVHT"].to_numpy(dtype=float),
                    df["wavelength"].to_numpy(dtype=float),
                ]
            ),
            hovertemplate="%{customdata[0]}<br>Sig. wave height: %{customdata[1]:.1f} m<br>Wavelength: %{customdata[2]:.0f} m<br>Wave slope: %{x:.4f}<br>DC power: %{y:.1f} W<extra></extra>",
            name="Hourly averages",
        ),
    )

    if len(df) >= 10 and df["wave_slope"].nunique() > 1:
        slope_min = df["wave_slope"].min()
        slope_max = df["wave_slope"].max()
        slope_bins = np.linspace(slope_min, slope_max, 21)
        df["slope_bin"] = pd.cut(df["wave_slope"], bins=slope_bins, include_lowest=True)
        trend = df.groupby("slope_bin", observed=False)["DcP"].mean().dropna()

        if len(trend) > 0:
            bin_centers = np.array([interval.mid for interval in trend.index], dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=trend.values,
                    mode="lines",
                    line=dict(color='black', width=3),
                    hovertemplate="Mean: %{y:.1f} W<extra></extra>",
                    name="Mean",
                ),
            )

    fig.update_layout(
        template="simple_white",
        showlegend=False,
    )
    fig.update_layout(
        xaxis_title="Sig. wave steepness, H/λ [-]",
        yaxis_title="DC power [W]",
    )
    fig.update_yaxes(range=[0, np.infty])
    fig.update_xaxes(range=[0, np.infty])

    return fig


def make_cw_matrix(ds, tp_to_te=0.9):
    ds0 = ds.mean("buoy")
    ds0["Te"] = ds0["DPD"] * tp_to_te
    ds0["J"] = ds0["Te"] * ds0["WVHT"] ** 2 * 1025 * 9.81**2 / (64 * np.pi)
    ds0["cw"] = ds0["DcP"] / ds0["J"]
    fig = px.density_heatmap(
        ds0[["DPD", "WVHT", "cw"]],
        x="DPD",
        y="WVHT",
        z="cw",
        color_continuous_scale="Reds",
        histfunc="avg",
        labels={
            "DPD": "Peak wave period [s]",
            "WVHT": "Sig. wave height [m]",
            "cw": "Capture width [m]",
        },
    )

    fig.add_trace(
        go.Scatter(
            x=ds0["DPD"],
            y=ds0["WVHT"],
            mode="markers",
            marker=dict(
                color="black",  # Set point color to black
                size=5,  # Set point size
                opacity=0.25,  # Set point opacity
            ),
            hoverinfo="skip",
        )
    )

    return fig


def make_gain_scatter(ds):
    dstp = ds[["Gain", "DcP"]].groupby_bins("Gain", bins=20).quantile(0.9).dropna("Gain_bins")

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Scatter(
            x=dstp["Gain"],
            y=dstp["DcP"],
            mode="lines",
            name="DC power",
            line=dict(color="black"),
            hovertemplate="90th percentile: %{y:.1f}W",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=ds["Gain"],
            y=ds["DcP"],
            mode="markers",
            # name="Gain",
            marker=dict(size=4, color="black", opacity=0.25),
            hoverinfo="skip",
        ),
    )

    fig.update_layout(
        template="simple_white",
        showlegend=False,
    )

    fig.update_layout(
        xaxis_title="Damping gain [As/rad]",
        yaxis_title="Power [W]",
    )

    fig.update_yaxes(range=[0, np.infty])

    return fig


def make_spectral_overview(ds, ds_spectral):
    """
    Create a three-panel plot showing:
    1. Wave height vs time
    2. WEC DC power vs time
    3. Wave spectral density contour (time vs frequency)
    """
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.2, 0.2, 0.6],
    )
    
    # Panel 1: Wave height vs time
    ds_mean = ds.mean("buoy")
    fig.add_trace(
        go.Scatter(
            x=ds_mean["WVHT"].time,
            y=ds_mean["WVHT"],
            mode="lines",
            name="Sig. wave height",
            line=dict(color="#1f77b4", width=1),
            hovertemplate="%{y:.2f} m",
        ),
        row=1, col=1
    )
    
    # Panel 2: WEC DC power vs time
    fig.add_trace(
        go.Scatter(
            x=ds["DcP"].time,
            y=ds["DcP"],
            mode="lines",
            name="WEC DC power",
            line=dict(color="black", width=1),
            hovertemplate="%{y:.1f} W",
        ),
        row=2, col=1
    )
    
    # Panel 3: Wave spectral density (downsampled for browser performance)
    if len(ds_spectral.data_vars) > 0 and "spectral_density" in ds_spectral:
        # Use first buoy's data
        ds_spec_buoy = ds_spectral.isel(buoy=0)

        da_spec = ds_spec_buoy["spectral_density"].transpose("time", "frequency")
        da_spec = da_spec.resample(time="1H").mean(skipna=True)

        # Cap number of time columns to keep output HTML responsive.
        max_time_cols = 1500
        if da_spec.sizes.get("time", 0) > max_time_cols:
            stride = int(np.ceil(da_spec.sizes["time"] / max_time_cols))
            da_spec = da_spec.isel(time=slice(None, None, stride))

        times = da_spec.time.values
        freqs = da_spec.frequency.values
        spectral_values = da_spec.values
        spectral_values = np.asarray(spectral_values, dtype=float)
        spectral_values[spectral_values <= 0] = np.nan
        with np.errstate(invalid="ignore"):
            spectral_values = np.log10(spectral_values)

        fig.add_trace(
            go.Heatmap(
                x=times,
                y=freqs,
                z=spectral_values.T,
                zmin=-2,
                zmax=2,
                colorscale="Viridis",
                colorbar=dict(
                    title="log₁₀(m²/Hz)",
                    len=0.3,
                    y=0.15,
                ),
                hovertemplate="Time: %{x}<br>Freq: %{y:.3f} Hz<br>Density: %{z:.3f} m²/Hz<extra></extra>",
            ),
            row=3, col=1
        )
    else:
        # Add text annotation if no spectral data
        fig.add_annotation(
            text="No spectral data available",
            xref="x3", yref="y3",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=14, color="gray"),
            row=3, col=1
        )
    
    # Update axes labels
    fig.update_yaxes(title_text="Sig. wave height [m]", row=1, col=1)
    fig.update_yaxes(title_text="DC power [W]", range=[0, None], row=2, col=1)
    fig.update_yaxes(title_text="Frequency [Hz]", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    
    fig.update_layout(
        template="simple_white",
        height=1000,
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=60, r=60, t=20, b=50),
    )
    
    # Add date selector
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ])
        ),
        row=1, col=1
    )
    
    return fig


def make_calendar(ds):
    fig = calplot(
        ds["DcP"].to_dataframe().resample("D").mean().reset_index(),
        x="time",
        y="DcP",
        start_month=1,
        month_lines_width=2,
        month_lines_color="black",
        colorscale="reds",
        name="DC power",
    )

    return fig


def make_table(ds):
    pow = ds["current"] * ds["voltage"]
    pow = pow.groupby("gtype").sum().sel(gtype=["solar", "wind"])
    tmp1 = ds["DcP"].expand_dims(dim={"gtype": ["WEC"]})
    pow.name = "Power"
    pow = xr.concat([pow, tmp1], dim="gtype")
    pow.attrs = {"units": "W", "long_name": "Power"}
    df = pow.dropna(dim="time").to_pandas()

    def cv(x):
        return x.std() / x.mean()

    def percentile(n):
        def percentile_(x):
            return x.quantile(n)

        def get_ordinal_suffix(n):
            """
            Appends the correct English ordinal suffix to a number.

            Examples:
            1 -> 1st
            2 -> 2nd
            3 -> 3rd
            11 -> 11th
            21 -> 21st
            """
            if 11 <= (n % 100) <= 13:
                suffix = "th"
            elif n % 10 == 1:
                suffix = "st"
            elif n % 10 == 2:
                suffix = "nd"
            elif n % 10 == 3:
                suffix = "rd"
            else:
                suffix = "th"
            return f"{n}{suffix}"

        percentile_.__name__ = get_ordinal_suffix(int(n * 100)) + "-percentile"
        return percentile_

    dft = df.agg(
        [
            "median",
            "mean",
            "std",
            cv,
            "max",
            percentile(0.1),
            percentile(0.25),
            percentile(0.75),
            percentile(0.9),
        ]
    )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[""] + list(dft.columns),
                    fill_color="paleturquoise",
                    align=["right", "center", "center", "center"],
                ),
                cells=dict(
                    values=[dft.index, dft.solar, dft.wind, dft.WEC],
                    format=[None, ".1f", ".1f", ".1f"],
                    fill_color=["paleturquoise", "lavender", "lavender", "lavender"],
                    align=["right", "center", "center", "center"],
                ),
            ),
        ]
    )
    return fig


def make_generators_box(ds):
    pow = ds["current"] * ds["voltage"]
    pow = pow.groupby("gtype").sum().sel(gtype=["solar", "wind"])
    tmp1 = ds["DcP"].expand_dims(dim={"gtype": ["WEC"]})
    pow.name = "Power"
    pow = xr.concat([pow, tmp1], dim="gtype")
    pow.attrs = {"units": "W", "long_name": "Power"}
    df = pow.dropna(dim="time").to_pandas()

    fig = px.box(
        df,
        #  log_y=True,
        labels={"value": "Avg. hourly power [W]", "gtype": "Type"},
        points="all",
        notched=False,
        color_discrete_sequence=px.colors.qualitative.Set1,
        color="gtype",
    )

    fig.update_traces(boxmean=True)

    fig.update_layout(
        template="simple_white",
    )
    # fig.update_yaxes(range=[0, np.infty])
    fig.update_xaxes(ticktext=["Solar", "Wind", "WEC"])

    return fig


def generate_jekyll_includes(ds, ds_pwrsys):
    """
    Generate Jekyll include files for stats, plots, and data downloads.
    These will be included in the index.md Jekyll template.
    """
    from pathlib import Path
    import subprocess
    
    logger.info("Generating Jekyll includes")
    
    # Create _includes directory
    includes_dir = Path('_includes')
    includes_dir.mkdir(exist_ok=True)
    
    # 1. GENERATE STATS INCLUDE
    try:
        start_date = str(ds.time.min().dt.strftime('%Y-%m-%d').values)
        end_date = str(ds.time.max().dt.strftime('%Y-%m-%d').values)
        num_days = (ds.time.max() - ds.time.min()).values / np.timedelta64(1, 'D')
        peak_power = float(ds['DcP'].max())
        mean_power = float(ds['DcP'].mean())
        median_power = float(ds['DcP'].median())
        total_energy_kwh = float(ds['DcP'].sum() * 1 / 1000)
        data_points = len(ds.time)
        
        # Power system stats
        if ds_pwrsys is not None and 'current' in ds_pwrsys and 'voltage' in ds_pwrsys:
            solar_power = (ds_pwrsys['current'] * ds_pwrsys['voltage']).where(
                ds_pwrsys['gtype'] == 'solar'
            ).sum(dim='device').mean().values
            wind_power = (ds_pwrsys['current'] * ds_pwrsys['voltage']).where(
                ds_pwrsys['gtype'] == 'wind'
            ).sum(dim='device').mean().values
            solar_str = f"{float(solar_power):.1f} W"
            wind_str = f"{float(wind_power):.1f} W"
        else:
            solar_str = "N/A"
            wind_str = "N/A"
        
    except Exception as e:
        logger.warning(f"Error calculating statistics: {e}")
        start_date = end_date = "N/A"
        num_days = peak_power = mean_power = median_power = total_energy_kwh = 0
        data_points = 0
        solar_str = wind_str = "N/A"
    
    stats_html = f"""<div class="stats-grid">
    <div class="stat-card">
        <h3>Deployment date</h3>
        <p>{start_date}</p>
    </div>
    <div class="stat-card">
        <h3>Deployment duration</h3>
        <p>{num_days:.0f} days</p>
    </div>
    <div class="stat-card">
        <h3>Peak WEC power</h3>
        <p>{peak_power:.1f} W</p>
    </div>
    <div class="stat-card">
        <h3>Mean WEC power</h3>
        <p>{mean_power:.1f} W</p>
    </div>
    <div class="stat-card">
        <h3>Median WEC power</h3>
        <p>{median_power:.1f} W</p>
    </div>
    <div class="stat-card">
        <h3>Total WEC energy</h3>
        <p>{total_energy_kwh:.1f} kWh</p>
    </div>
    <div class="stat-card">
        <h3>Mean Solar Power</h3>
        <p>{solar_str}</p>
    </div>
    <div class="stat-card">
        <h3>Mean Wind Power</h3>
        <p>{wind_str}</p>
    </div>
</div>"""
    
    (includes_dir / 'stats.html').write_text(stats_html)
    
    # 2. GENERATE PLOTS INCLUDE
    plots_metadata = {
        'time_hist': {'title': 'Time Series History', 'description': 'Multi-panel time series of wave conditions, WEC power, and auxiliary systems', 'icon': '📈'},
        'spectral_overview': {'title': 'Wave Spectral Density', 'description': 'Wave height, WEC power, and wave spectral density (NDBC 44014) over time', 'icon': '🌈'}, #TODO
        'calendar': {'title': 'Power Generation Calendar', 'description': 'Daily average DC power generation heatmap calendar', 'icon': '📅'},
        'scatter_3d': {'title': '3D Performance Scatter', 'description': 'Wave height, controller gain, and DC power in 3D space', 'icon': '🌊'},
        'jpd': {'title': 'Joint Probability Distribution', 'description': 'Wave height vs. peak period occurrence density', 'icon': '📊'},
        'correlation_matrix': {'title': 'Correlation Matrix', 'description': 'Scatter matrix showing correlations between key variables', 'icon': '🔗'},
        'power_matrix': {'title': 'Power Matrix', 'description': 'Average DC power as function of wave height and period', 'icon': '⚡'},
        'power_vs_wave_slope': {'title': 'Wave Slope Analysis', 'description': 'WEC DC power versus wave slope (height/wavelength)', 'icon': '📐'},
        'cw_matrix': {'title': 'Capture Width Matrix', 'description': 'Capture width efficiency across sea states', 'icon': '📏'},
        'histograms': {'title': 'Power Histograms', 'description': 'Distribution of DC and export power', 'icon': '📊'},
        'gain_scatter': {'title': 'Damping Gain Analysis', 'description': 'Power output vs. control system damping gain', 'icon': '🎛️'},
        'generators_box': {'title': 'Power Systems Box Plot', 'description': 'Distribution comparison of solar, wind, and WEC power', 'icon': '🔋'},
        'generators_table': {'title': 'Power Systems Statistics', 'description': 'Statistical summary table for all power sources', 'icon': '📋'},
    }
    
    plots_html = '<ul class="plot-list">\n'
    for plot_id, info in plots_metadata.items():
        plots_html += f"""    <li>
        <a href="{plot_id}.html" target="_blank">
            <span class="icon">{info['icon']}</span>
            <strong>{info['title']}:</strong> {info['description']}
        </a>
    </li>
"""
    plots_html += '</ul>\n'
    
    (includes_dir / 'plots.html').write_text(plots_html)
    
    # 3. GENERATE DATA DOWNLOADS INCLUDE
    data_files = list(Path('output/data/').glob('*.h5.gz'))
    if data_files:
        data_html = '<ul class="data-files">\n'
        for data_file in data_files:
            try:
                size_mb = data_file.stat().st_size / (1024*1024)
                data_html += f"""    <li>
        <a href="data/{data_file.name}" download>
            📁 {data_file.name}
        </a>
        <span class="file-size">({size_mb:.1f} MB)</span>
    </li>
"""
            except Exception:
                pass
        data_html += '</ul>\n'
    else:
        data_html = '<p>No data files available for download.</p>'
    
    (includes_dir / 'data_downloads.html').write_text(data_html)
    
    logger.info("Generated Jekyll include files")


def _parse_start_date(value: str) -> datetime.date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _compress_file(filepath: str) -> None:
    """Compress a file with gzip and remove the original."""
    import gzip
    import shutil

    compressed_path = f"{filepath}.gz"
    temp_compressed_path = f"{compressed_path}.tmp"
    try:
        with open(filepath, 'rb') as f_in:
            with gzip.open(temp_compressed_path, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.replace(temp_compressed_path, compressed_path)
        os.remove(filepath)
        logger.info(f"Compressed {filepath} to {compressed_path}")
    except Exception:
        if os.path.exists(temp_compressed_path):
            os.remove(temp_compressed_path)
        raise


def fetch_data(start_date: datetime.date) -> None:
    _ensure_data_dir()

    ds_pwrsys = fetch_pwrsys_data(start_date=start_date)
    pwrsys_path = os.path.join(DATA_DIR, "pwrsys_data.h5")
    ds_pwrsys.to_netcdf(pwrsys_path, engine="h5netcdf", invalid_netcdf=True)
    _compress_file(pwrsys_path)

    buoys = ["44014", "44079", "41083", "44095"]
    ds_ndbc = xr.concat(
        [fetch_ndbc(buoy_id=buoy_id, start_date=start_date) for buoy_id in buoys],
        dim="buoy",
    )
    ndbc_path = os.path.join(DATA_DIR, "ndbc_data.h5")
    ds_ndbc.to_netcdf(ndbc_path, engine="h5netcdf", invalid_netcdf=True)
    _compress_file(ndbc_path)

    ds_wec = fetch_wec_data(start_date=start_date)
    wec_path = os.path.join(DATA_DIR, "wec_data.h5")
    ds_wec.to_netcdf(wec_path, engine="h5netcdf", invalid_netcdf=True)
    _compress_file(wec_path)

    # Fetch spectral data for buoy 44014 only
    ds_spectral = fetch_ndbc_spectral(start_date=start_date)
    if len(ds_spectral.data_vars) > 0:
        spectral_path = os.path.join(DATA_DIR, "ndbc_spectral.h5")
        ds_spectral.to_netcdf(spectral_path, engine="h5netcdf", invalid_netcdf=True)
        _compress_file(spectral_path)

    logger.info("Data fetching complete")


def load_cached_data() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    import gzip
    import tempfile
    from gzip import BadGzipFile
    
    pwrsys_path = DATA_DIR / "pwrsys_data.h5.gz"
    ndbc_path = DATA_DIR / "ndbc_data.h5.gz"
    wec_path = DATA_DIR / "wec_data.h5.gz"
    spectral_path = DATA_DIR / "ndbc_spectral.h5.gz"

    missing = [path for path in (pwrsys_path, ndbc_path, wec_path) if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing cached data files: {missing_list}. Run 'python app.py fetch-data' first."
        )

    # Prefer plain .h5 when present (e.g., interrupted compression left it behind).
    # Otherwise load .h5.gz by decompressing to a temporary file.
    def load_cached_dataset(base_path: Path):
        plain_path = Path(str(base_path).removesuffix('.gz')) if str(base_path).endswith('.gz') else base_path

        if plain_path.exists():
            return xr.load_dataset(
                plain_path,
                engine="h5netcdf",
                decode_timedelta=False,
            )

        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            temp_path = tmp.name
            try:
                with gzip.open(base_path, 'rb') as f_in:
                    tmp.write(f_in.read())
            except BadGzipFile as e:
                os.remove(temp_path)
                raise RuntimeError(
                    f"Corrupted gzip cache file: {base_path}. Re-run 'python app.py fetch-data'."
                ) from e

        try:
            ds = xr.load_dataset(
                temp_path,
                engine="h5netcdf",
                decode_timedelta=False,
            )
        finally:
            os.remove(temp_path)

        return ds
    
    ds_pwrsys = load_cached_dataset(pwrsys_path)
    ds_ndbc = _adapt_ndbc_stdmet(load_cached_dataset(ndbc_path))
    ds_wec = load_cached_dataset(wec_path)

    # Load spectral data if available
    if spectral_path.exists():
        ds_spectral = _adapt_ndbc_spectral(load_cached_dataset(spectral_path))
        logger.info("Loaded spectral data")
    else:
        ds_spectral = xr.Dataset()
        logger.warning("No spectral data found")

    return ds_wec, ds_ndbc, ds_pwrsys, ds_spectral


def generate_plots() -> None:
    ds_wec, ds_ndbc, ds_pwrsys, ds_spectral = load_cached_data()
    ds = resample_and_combine(ds_wec, [ds_ndbc, ds_pwrsys])

    logger.info("Generating plots")

    fig1 = make_scatter_3d(ds)
    fig1.write_html("output/scatter_3d.html")

    fig2 = make_time_hist(ds)
    fig2.write_html("output/time_hist.html")

    fig3 = make_correlation_matrix(ds)
    fig3.write_html("output/correlation_matrix.html")

    fig4 = make_jpd(ds)
    fig4.write_html("output/jpd.html")

    fig5 = make_power_matrix(ds)
    fig5.write_html("output/power_matrix.html")

    fig6 = make_cw_matrix(ds)
    fig6.write_html("output/cw_matrix.html")

    fig7 = make_wec_histograms(ds)
    fig7.write_html("output/histograms.html")

    fig8 = make_gain_scatter(ds)
    fig8.write_html("output/gain_scatter.html")

    fig9 = make_calendar(ds)
    fig9.write_html("output/calendar.html")

    fig10 = make_generators_box(ds)
    fig10.write_html("output/generators_box.html")

    fig11 = make_table(ds)
    fig11.write_html("output/generators_table.html")

    fig12 = make_spectral_overview(ds, ds_spectral)
    fig12.write_html("output/spectral_overview.html")

    fig13 = make_power_vs_wave_slope(ds)
    fig13.write_html("output/power_vs_wave_slope.html")

    logger.info("Plot generation complete")


def _copy_site_output() -> None:
    import shutil

    source_root = Path("output/site")
    dest_root = Path("output")

    if not source_root.exists():
        logger.warning("Site output not found at output/site")
        return

    # Copy files from Jekyll build to output root
    for root, dirs, files in os.walk(source_root):
        # Skip copying nested output directories
        dirs[:] = [d for d in dirs if d != 'output']
        
        rel_path = Path(root).relative_to(source_root)
        target_dir = dest_root / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            source_path = Path(root) / file_name
            target_path = target_dir / file_name
            shutil.copy2(source_path, target_path)
    
    # Clean up intermediate Jekyll build directory
    shutil.rmtree(source_root)
    logger.info("Cleaned up intermediate Jekyll build directory")


def build_jekyll_site() -> None:
    import subprocess
    import shutil
    from pathlib import Path

    logger.info("Building Jekyll site...")

    # Windows often provides bundle as bundle.bat / bundle.cmd; resolve explicitly
    bundle = (
        shutil.which("bundle")
        or shutil.which("bundle.bat")
        or shutil.which("bundle.cmd")
    )
    if not bundle:
        logger.error(
            "Bundler ('bundle') not found on PATH. Install Ruby/Bundler and ensure Ruby\\bin is on PATH."
        )
        return

    # Build Jekyll somewhere other than output/ so it won't delete output/data and plot HTML
    site_dir = Path("output/site")
    site_dir.mkdir(parents=True, exist_ok=True)

    cmd = [bundle, "exec", "jekyll", "build", "-d", str(site_dir)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("Jekyll site built successfully")
        else:
            logger.error(
                "Jekyll build failed\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
    except subprocess.TimeoutExpired:
        logger.error("Jekyll build timed out")
    except Exception as e:
        logger.error(f"Error running Jekyll: {e}")


def build_site() -> None:
    ds_wec, ds_ndbc, ds_pwrsys, ds_spectral = load_cached_data()
    ds = resample_and_combine(ds_wec, [ds_ndbc, ds_pwrsys])
    generate_jekyll_includes(ds, ds_pwrsys)
    build_jekyll_site()
    _copy_site_output()
    logger.info("Site build complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pioneer WEC dashboard pipeline")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser("fetch-data", help="Fetch and cache raw data")
    fetch_parser.add_argument(
        "--start-date",
        default="2025-11-03",
        help="Start date (YYYY-MM-DD)",
    )

    subparsers.add_parser("generate-plots", help="Generate plots from cached data")
    subparsers.add_parser("build-site", help="Generate includes and build Jekyll site")

    all_parser = subparsers.add_parser("all", help="Run fetch, plots, and site build")
    all_parser.add_argument(
        "--start-date",
        default="2025-11-03",
        help="Start date (YYYY-MM-DD)",
    )

    args = parser.parse_args()
    
    # Default to "all" if no command specified
    if args.command is None:
        args.command = "all"

    if args.command == "fetch-data":
        fetch_data(_parse_start_date(getattr(args, 'start_date', '2025-11-03')))
    elif args.command == "generate-plots":
        generate_plots()
    elif args.command == "build-site":
        build_site()
    elif args.command == "all":
        fetch_data(_parse_start_date(getattr(args, 'start_date', '2025-11-03')))
        generate_plots()
        build_site()

    logger.info("Dashboard generation complete")
