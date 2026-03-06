---
layout: default
title: Pioneer WEC Dashboard
carousels:
  - images: 
    - image: images/Pioneer_deployment_4_16x9.webp
      caption: "Engineers ready Pioneer WEC v1 prototype for testing on NLR's Large Amplitude Motion Platform (LAMP) (credit: Taylor Mankle and Josh Baurer, NLR)"
    - image: images/Pioneer_v1_final_assembly_damian_16x9.webp
      caption: "Pioneer WEC v1 final assembly (credit: Taylor Mankle and Josh Baurer, NLR)"
    - image: images/Pioneer_deployment_1_16x9.webp
      caption: "Pioneer WEC v1 prototype undergoes final readiness testing (credit: Taylor Mankle and Josh Baurer, NLR)"
    - image: images/Pioneer_deployment_2_16x9.webp
      caption: "Pioneer WEC v1 pendulum and electronics panel (credit: Taylor Mankle and Josh Baurer, NLR)"
    - image: images/Pioneer_ar98a_two_csms_on_deck_16x9.webp
      caption: "Lifting operation during recovery of a Coastal Surface Mooring"
    - image: images/Pioneer_buoys_on_deck_16x9.webp
      caption: "Two Coastal Surface Mooring buoys on deck of R/V Armstrong (credit: Ocean Observatories Initiative)"
    - image: images/Pioneer_csm_in_water_with_ar98_16x9.webp
      caption: "Central surface mooring with Pioneer WEC v1 prototype; R/V Armstrong in background (credit: Taylor Mankle and Josh Baurer, NLR)"
    - image: images/Pioneer_deployment_3_16x9.webp
      caption: "Central surface mooring with Pioneer WEC v1 prototype operating at sea (credit: Taylor Mankle and Josh Baurer, NLR)"
---

<section id="overview" class="content-section" markdown="1">

## Overview

This dashboard shows data from the Pioneer wave energy converter (WEC) v1 prototype's deployment, which provides power to a mooring within the [Coastal Pioneer Array](https://oceanobservatories.org/pioneer-array-relocation/). The [Coastal Pioneer Array](https://oceanobservatories.org/pioneer-array-relocation/) is an NSF-funded project within the [Ocean Observatories Initiative (OOI)](https://oceanobservatories.org/) that provides oceanographic data relevant to cross-shelf dynamics. The Pioneer WEC uses a novel pitch resonator design {% cite Lee:2024aa Coe:2023aa %} that has been optimized through modeling {% cite Grasberger:2024ab %}, testing {% cite Lee:2024aa Coe:2024ab %}, control co-design {% cite Devin:2024aa Coe:2024aa Keow:2025aa Keow:2025ab %}.
The Pioneer WEC v1 prototype was deployed on November 3rd, 2025 and is rescheduled to be recovered in May, 2026.

## Project goals

- Deliver power to support OOI's scientific mission
- Advance the state-of-the-art for wave energy converter design
- Openly disseminate data and findings

## v1 prototype goals

- WEC functionality proof-of-concept
- System management and interfaces functionality
- Benchmark numerical models for WEC performance
- Gather data to inform future design iterations

## Data Sources

### Wave & Meteorological Data
Wave measurements and environmental conditions from NDBC buoys:
- **[44014](https://www.ndbc.noaa.gov/station_page.php?station=44014):** Virginia Beach, 64 NM Southeast of Cape Henry, VA
- **[44079](https://www.ndbc.noaa.gov/station_page.php?station=44079):** Mid-Atlantic Bight Northern Surface Mooring
- **[41083](https://www.ndbc.noaa.gov/station_page.php?station=41083):** Mid-Atlantic Bight Southern Surface Mooring
- **[44095](https://www.ndbc.noaa.gov/station_page.php?station=44095):** Oregon Inlet, NC

### Pioneer data
Real-time data from the Central Surface Mooring:
- **[WEC data (power, motions, etc.)](https://rawdata.oceanobservatories.org/files/CP10CNSM/D00003/cg_data/dcl12/wec_decimated/):** decimated summary data uploaded via iridium satellite nightly
- **[General data from OOI](https://dataexplorer.oceanobservatories.org):** scientific data collected by OOI on the Central Surface Mooring and other platforms

</section>

<section id="gallery" class="content-section">
<h2>Photo Gallery</h2>
{% include carousel.html height="60" unit="%" number="1" %}
</section>

<section id="stats" class="stats-section">
<h2>Summary Statistics</h2>
{% include stats.html %}
</section>

<section id="plots" class="plots-section">
<h2>Interactive Visualizations</h2>
<p style="margin-bottom: 2rem; color: #666;">
    Click any plot to open in new tab. Hover, zoom, and pan for details.
</p>
{% include plots.html %}
</section>

<section id="data" class="data-section">
<h2>Data Downloads</h2>
<p style="margin-bottom: 1rem; color: #666;">
    Raw data files in HDF5/NetCDF format. Can be opened with xarray, Python, MATLAB, or other scientific tools.
</p>
{% include data_downloads.html %}
</section>

<section id="references" class="references-section">
<h2>References</h2>
{% bibliography --cited %}
</section>
