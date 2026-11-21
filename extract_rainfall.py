import glob
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
SHP_PATH = "mizoram_basin.shp"
NC_PATTERN = "RF25_ind*_rfp25.nc"
DAILY_CSV = "mizoram_daily_rainfall_1901_2025.csv"
ANNUAL_CSV = "mizoram_annual_rainfall_1901_2025.csv"

# ---------------------------------------------------------
# 2. LOAD BASIN SHAPEFILE & FIND NETCDF FILES
# ---------------------------------------------------------
basin = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
nc_files = sorted(glob.glob(NC_PATTERN))

if not nc_files:
    raise FileNotFoundError(f"No files matching {NC_PATTERN} found in {os.getcwd()}")

print(f"Found {len(nc_files)} NetCDF files. Starting extraction...")

# ---------------------------------------------------------
# 3. CLIP EACH YEAR & COMPUTE DAILY SPATIAL MEAN
# ---------------------------------------------------------
daily_records = []

for i, fp in enumerate(nc_files, 1):
    ds = xr.open_dataset(fp)

    # Auto-detect coordinate names (handles LATITUDE/lat, LONGITUDE/lon, TIME/time)
    lat_name = next(c for c in ds.coords if "lat" in c.lower())
    lon_name = next(c for c in ds.coords if "lon" in c.lower())
    time_name = next(c for c in ds.coords if "time" in c.lower())

    # Auto-detect the rainfall data variable
    rain_var = next(
        v for v in ds.data_vars
        if "rain" in v.lower() or "rf" in v.lower() or len(ds[v].dims) == 3
    )

    da = ds[rain_var]

    # Mask IMD missing-data fill values (typically -999)
    da = da.where(da >= 0)

    # Attach CRS and spatial dimensions for rioxarray clipping
    da = da.rio.set_spatial_dims(x_dim=lon_name, y_dim=lat_name)
    da = da.rio.write_crs("EPSG:4326")

    # Clip to basin polygon (all_touched=True ensures intersecting 0.25° cells are kept)
    clipped = da.rio.clip(basin.geometry, basin.crs, drop=True, all_touched=True)

    # Spatial average across the basin for each day
    daily_mean = clipped.mean(dim=[lat_name, lon_name], skipna=True)

    df_year = pd.DataFrame({
        "date": pd.to_datetime(daily_mean[time_name].values),
        "rainfall_mm": daily_mean.values
    })
    daily_records.append(df_year)
    ds.close()

    if i % 25 == 0 or i == len(nc_files):
        print(f"Processed {i}/{len(nc_files)} files ({os.path.basename(fp)})")

# ---------------------------------------------------------
# 4. COMBINE, AGGREGATE & EXPORT TO CSV
# ---------------------------------------------------------
daily_df = pd.concat(daily_records, ignore_index=True).sort_values("date")
daily_df.to_csv(DAILY_CSV, index=False)

daily_df["year"] = daily_df["date"].dt.year
annual_df = (
    daily_df.groupby("year")["rainfall_mm"]
    .agg(annual_total_mm="sum", daily_mean_mm="mean", max_daily_mm="max")
    .reset_index()
)
annual_df.to_csv(ANNUAL_CSV, index=False)

print(f"\nDone!")
print(f"  Daily data saved to:  {DAILY_CSV} ({len(daily_df)} days)")
print(f"  Annual data saved to: {ANNUAL_CSV} ({len(annual_df)} years)")

# ---------------------------------------------------------
# 5. QUICK VISUAL CHECK (ANNUAL RAINFALL TIME SERIES)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(annual_df["year"], annual_df["annual_total_mm"], color="steelblue", lw=1.5, label="Annual Total Rainfall")

# Add a simple linear trendline
z = np.polyfit(annual_df["year"], annual_df["annual_total_mm"], 1)
p = np.poly1d(z)
ax.plot(annual_df["year"], p(annual_df["year"]), "r--", lw=1.5, label=f"Trend ({z[0]:+.2f} mm/yr)")

ax.set_title("Mizoram Basin Annual Rainfall (1901–2025)")
ax.set_xlabel("Year")
ax.set_ylabel("Total Rainfall (mm)")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()