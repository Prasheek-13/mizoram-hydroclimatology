# Mizoram River Basin Hydroclimatology (1901–2025)

An end-to-end Python pipeline for automated watershed delineation, historical gridded rainfall extraction, trend detection, and extreme value analysis for a river basin in Mizoram, India—completely bypassing manual GIS workflows.

## Overview

This repository automates three core hydroclimatological workflows:
1. **Watershed Delineation (`delineate_basin.py`):** Downloads SRTM 1-arc-second DEM tiles (`N23E092`, `N23E093`, `N22E092`, `N22E093`), merges them, performs Wang & Liu depression filling, computes D8 flow direction/accumulation via WhiteboxTools, snaps the target outlet (`23.0344° N, 92.9653° E`) to the main channel, and exports the catchment shapefile (`mizoram_basin.shp`).
2. **Rainfall Extraction (`extract_rainfall.py`):** Clips 125 years (1901–2025) of IMD $0.25^\circ \times 0.25^\circ$ daily gridded rainfall NetCDF files (`RF25_ind*_rfp25.nc`) to the delineated basin polygon (`all_touched=True`) and computes spatially averaged daily and annual rainfall series.
3. **Statistical & Extreme Value Analysis (`analyze_rainfall.py`):** Performs data quality auditing, aggregates rainfall into standard IMD seasons (JF, MAM, JJAS, OND), runs non-parametric Mann-Kendall trend tests with Sen's slope estimation, and fits a Generalized Extreme Value (GEV) distribution to annual maximum 1-day rainfall.

---

## Key Findings (1901–2025)

### Data Quality
* **Record Length:** 45,656 daily observations with **0 missing (`NaN`) days**.
* **Driest Years:** 1994 (731.2 mm), 1968 (1,031.2 mm), 1969 (1,157.5 mm), 1961 (1,217.7 mm), and 1960 (1,231.2 mm)—all containing complete 365/366-day records.

### Mann-Kendall Trend Test & Sen's Slope
| Metric | Trend | Sen's Slope (mm/yr) | *p*-value | Significance |
| :--- | :--- | :---: | :---: | :--- |
| **Annual Total** | Decreasing | -4.002 | 0.0008 | Significant ($p < 0.01$) |
| **Monsoon (JJAS)** | Decreasing | -2.140 | 0.0088 | Significant ($p < 0.01$) |
| **Pre-Monsoon (MAM)** | Decreasing | -1.746 | 0.0022 | Significant ($p < 0.01$) |
| **Max 1-Day Rainfall** | No trend | +0.008 | 0.9066 | Not significant |

> **Hydrological Takeaway:** While total annual, monsoon, and pre-monsoon rainfall exhibit a statistically significant century-scale decline, annual maximum 1-day rainfall remains stationary—indicating sustained flash-flood hazard despite overall basin drying.

### Extreme 1-Day Rainfall Return Levels (GEV Fit)
| Return Period (Years) | 2 | 5 | 10 | 25 | 50 | 100 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Day Max Rainfall (mm/day)** | 81.5 | 108.0 | 127.5 | 154.5 | 176.5 | 200.2 |

---

## Tech Stack & Dependencies

* **Hydrology & Geospatial:** `whitebox` (WhiteboxTools), `rasterio`, `geopandas`, `shapely`, `rioxarray`, `xarray`, `netCDF4`
* **Statistics & Visualization:** `pandas`, `numpy`, `scipy`, `pymannkendall`, `matplotlib`, `requests`

Install all dependencies:
```bash
pip install whitebox requests rasterio geopandas shapely xarray rioxarray netCDF4 pandas scipy pymannkendall matplotlib