import pandas as pd
import numpy as np
import xarray as xr

expected_fields = [
    "Timestamp",
    "Cnt",
    "State",
    "Flags",
    "Pos",
    "Vel",
    "Iq",
    "DcV",
    "DcI",
    "DcP",
    "ExV",
    "ExI",
    "ExP",
    "WoV",
    "WoI",
    "WoP",
    "Tm",
    "Tr",
]

N_gearbox = 6.6
N_belt = 112/34
N = N_belt*N_gearbox

def parse_putty_log_file(file_name):
    with open(file_name) as f:
        data = __parse_lines(f)

    return data

def parse_putty_log(contents):
    data, proportion_corrupt = __parse_lines(contents.splitlines())
    return data, proportion_corrupt


def __parse_lines(lines):
    data = []

    line_count = 0
    failure = 0
    for i, line in enumerate(lines):
        line_count += 1
        try:
            parsed = __parse_line(line)
        except ValueError as e:
            failure += 1
            continue


        if parsed is not None:
            data.append(parsed)

    if line_count == 0:
        raise ValueError("file is empty")

    df = pd.DataFrame(data=data, columns=expected_fields)
    df.index = df["Timestamp"]
    df.drop(columns="Timestamp", inplace=True)

    ds = df.to_xarray()

    ds['Cnt'].attrs['long_name'] = 'Message count'
    ds['Cnt'].attrs['units'] = ' '

    ds['State'].attrs['long_name'] = 'WEC state'
    ds['State'].attrs['units'] = ' '

    ds['Flags'].attrs['long_name'] = 'Flags'

    ds['Pos'] = ds['Pos']/N*360
    ds['Pos'].attrs['long_name'] = 'Mean pend. pos.'
    ds['Pos'].attrs['units'] = 'deg'

    ds['Vel'] = ds['Vel']/N*360/60
    ds['Vel'].attrs['long_name'] = 'RMS pend. vel.'
    ds['Vel'].attrs['units'] = 'deg/s'

    ds['Iq'].attrs['long_name'] = 'RMS Iq actual'
    ds['Iq'].attrs['units'] = 'A'

    ds['DcV'].attrs['long_name'] = 'Mean DC bus potential'
    ds['DcV'].attrs['units'] = 'V'

    ds['DcI'].attrs['long_name'] = 'RMS DC bus current'
    ds['DcI'].attrs['units'] = 'A'

    ds['DcP'].attrs['long_name'] = 'Mean DC bus power'
    ds['DcP'].attrs['units'] = 'W'

    ds['ExV'].attrs['long_name'] = 'Mean export potential'
    ds['ExV'].attrs['units'] = 'V'

    ds['ExI'].attrs['long_name'] = 'RMS export current'
    ds['ExI'].attrs['units'] = 'A'

    ds['WoV'].attrs['long_name'] = 'Mean WHOI supply potential'
    ds['WoV'].attrs['units'] = 'V'

    ds['WoI'].attrs['long_name'] = 'RMS WHOI supply current'
    ds['WoI'].attrs['units'] = 'A'

    ds['WoP'].attrs['long_name'] = 'Mean WHOI supply power'
    ds['WoP'].attrs['units'] = 'W'

    ds['Tm'].attrs['long_name'] = 'Mean mag. spring torque'
    ds['Tm'].attrs['units'] = 'Nm'

    ds['Tr'].attrs['long_name'] = 'RMS mag. spring torque'
    ds['Tr'].attrs['units'] = 'Nm'

    dsf = extract_flag_data(ds.to_dataframe()).to_xarray().to_array()
    dsf = dsf.rename({'variable':'type'})
    dsf.name = 'Flags'
    ds['Timestamp'] = pd.to_datetime(ds['Timestamp'])

    dsf['Timestamp'] = pd.to_datetime(dsf['Timestamp'])
    ds = ds.rename({'Flags':'Flags_raw'})
    ds = xr.merge([ds,dsf])
    ds['Timestamp'] = pd.to_datetime(ds['Timestamp']) # utc=True?
    ds = ds.rename({'Timestamp':'time'})


    # Correct the export power
    # We have run into cases (eg 2025-11-12) where the dischargTrig never goes below 1.
    # In this case ExP_offset becomes NaN.
    if np.where(ds['Flags'].sel(type='dischargeTrig') < 0.5)[0].size == 0:
        ExP_offset = 0
        corrected = False
    else:
        ExP_offset = ds['ExP'].where((ds['Flags'].sel(type='dischargeTrig') < 0.5) & (ds['ExV'] > 10) ).mean()
        corrected = True

    ds = ds.rename({'ExP':'ExP_raw'})
    ds['ExP'] = ds['ExP_raw'] - ExP_offset

    ds['ExP_raw'].attrs['long_name'] = 'Mean Uncorrected Export Power'
    ds['ExP_raw'].attrs['units'] = 'W'

    ds['ExP'].attrs['long_name'] = 'Mean Export Power'
    ds['ExP'].attrs['units'] = 'W'
    ds['ExP'].attrs['corrected'] = corrected

    return ds, float(failure) / float(line_count)


def __parse_line(line):
    # Timestamp is the first 23 characters
    line = line.strip()
    t = pd.to_datetime(line[:23], format="%Y/%m/%d %H:%M:%S.%f", utc=True)

    fields = line[24:].split("\t")

    data = [t]
    for i, f in enumerate(fields):
        sep = f.index(":")

        # Check the field is the expected one, need to skip
        # timestmap in the expected list.
        if not f[:sep] == expected_fields[i+1]:
            raise ValueError(f"expected {expected_fields[i+1]}, got {f[:sep]}")

        # Take account of colon and following space
        str_value = f[sep+2:]
        if i <= 1:
            val = int(str_value.lstrip("0"), 10)
        elif i == 2:
            val = int(str_value, 0)
        else:
            val = float(str_value)

        data.append(val)

    return data

# Flags are defined in https://github.com/SNL-WaterPower/pioneer_wec_embedded/blob/dev/analysis/fParseFlags.m
__flags = [
    # WHOI 12V supply flags
    "W12VOverV",        # bit 0
    "W12VUnderV",       # bit 1
    "W12VOverI",        # bit 2
    "W12VUnderI",       # bit 3
    "W12VSensorF",      # bit 4

    # Buck Boost flags
    "BBOverV",   # bit 5
    "BBUnderV",  # bit 6
    "BBOverI",   # bit 7
    "BBUnderI",  # bit 8
    "BBSensorF", # bit 9

    # DC Bus flags
    "BusOverV",   # bit 10
    "BusUnderV",  # bit 11
    "BusOverI",   # bit 12
    "BusUnderI",  # bit 13
    "BusSensorF", # bit 14

    "null",         # Note: Bit 15 is not used

    "busV",          # bit 16
    "encoderIndex",  # bit 17
    "watchdog",      # bit 18
    "comsLeader",    # bit 19
    "bridgeEnabled", # bit 20
    "dischargeTrig", # bit 21
    "encoderFault",  # bit 22
    "pendulumWrap",  # bit 23
    "overVelocity",  # bit 24
    "currentLim",    # bit 25
    "drv8305Status", # bit 26
    "drv8305ComsF",  # bit 27
    "leakDetected",  # bit 28
    "leakDetErr",    # bit 29
    "torqueArmErr",  # bit 30
]

def parse_flags(flags):
    if isinstance(flags, int):
        pass
    elif isinstance(flags, np.int64):
        pass
    else:
        raise ValueError("flags must be an integer")

    out = {}
    for i, f in enumerate(__flags):
        if i == 15:
            # Bit 15 is not used
            continue

        out[f] = (flags & 1 << i) != 0

    return out

def extract_flag_data(putty_log, column="Flags"):
    flags = putty_log[column].values

    parsed = [None]*len(flags)
    for i, f in enumerate(flags):
        parsed[i] = parse_flags(f)

    if isinstance(putty_log, pd.DataFrame):
        index=putty_log.index
    elif isinstance(putty_log, xr.Dataset):
        index=putty_log.Timestamp
    else:
        raise ValueError("got something else" + type(putty_log))

    flags_df = pd.DataFrame(data=parsed, index=index)

    return flags_df