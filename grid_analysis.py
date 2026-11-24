import glob
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD BASIN SHAPEFILE & FIND NETCDF FILES
# ---------------------------------------------------------
SHP_PATH = "mizoram_basin.shp"
NC_PATTERN = "RF25_ind*_rfp25.nc"

basin = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")
basin_geom = basin.dissolve().geometry.iloc[0]

nc_files = sorted(glob.glob(NC_PATTERN))
if not nc_files:
    raise FileNotFoundError("No IMD NetCDF files found in D:\\Mizoram")

# ---------------------------------------------------------
# 2. IDENTIFY GRIDS FALLING IN THE CATCHMENT (FROM FIRST FILE)
# ---------------------------------------------------------
ds0 = xr.open_dataset(nc_files[0])
lat_name = next(c for c in ds0.coords if "lat" in c.lower())
lon_name = next(c for c in ds0.coords if "lon" in c.lower())
var_name = next(v for v in ds0.data_vars if v.lower() in ["rf", "rainfall", "rain"])

lats = ds0[lat_name].values
lons = ds0[lon_name].values
ds0.close()

# Filter to a small bounding box around the basin first for speed
minx, miny, maxx, maxy = basin_geom.bounds
cand_lats = lats[(lats >= miny - 0.5) & (lats <= maxy + 0.5)]
cand_lons = lons[(lons >= minx - 0.5) & (lons <= maxx + 0.5)]

strict_grids = []    # Grid center point is strictly inside the basin
overlap_grids = []   # 0.25 x 0.25 deg grid box intersects the basin

for la in cand_lats:
    for lo in cand_lons:
        pt = Point(lo, la)
        cell = box(lo - 0.125, la - 0.125, lo + 0.125, la + 0.125)
        if basin_geom.contains(pt):
            strict_grids.append((float(la), float(lo), cell))
        if basin_geom.intersects(cell):
            overlap_grids.append((float(la), float(lo), cell))

print(f"Grid centers STRICTLY inside basin: {len(strict_grids)}")
for la, lo, _ in strict_grids:
    print(f"  -> ({la:.2f}N, {lo:.2f}E)")

print(f"\n0.25x0.25 Grid cells OVERLAPPING basin: {len(overlap_grids)}")
for la, lo, _ in overlap_grids:
    print(f"  -> ({la:.2f}N, {lo:.2f}E)")

# Use strictly inside grids if available; otherwise fall back to overlapping grids
active_grids = overlap_grids
mode_label = "All Overlapping"
print(f"\nUsing [{mode_label}] grids ({len(active_grids)} grids) for averaging...")

# ---------------------------------------------------------
# 3. EXTRACT YEARLY RAINFALL PER GRID & AVERAGE (1901–2025)
# ---------------------------------------------------------
records = []

for i, fp in enumerate(nc_files, 1):
    # Extract year from filename (e.g., RF25_ind1901_rfp25.nc -> 1901)
    match = re.search(r"(19\d{2}|20\d{2})", os.path.basename(fp))
    year = int(match.group(1))

    ds = xr.open_dataset(fp)
    da = ds[var_name].where(ds[var_name] >= 0)  # Mask -999 missing values

    row = {"year": year}
    valid_vals = []

    for la, lo, _ in active_grids:
        col_name = f"Grid_{la:.2f}N_{lo:.2f}E"
        grid_series = da.sel({lat_name: la, lon_name: lo}, method="nearest")
        valid_days = int(grid_series.count().values)

        if valid_days > 0:
            annual_sum = float(grid_series.sum(skipna=True).values)
            row[col_name] = round(annual_sum, 2)
            valid_vals.append(annual_sum)
        else:
            row[col_name] = np.nan

    row["valid_grid_count"] = len(valid_vals)
    row["catchment_avg_mm"] = round(float(np.mean(valid_vals)), 2) if valid_vals else np.nan
    records.append(row)
    ds.close()

    if i % 25 == 0 or i == len(nc_files):
        print(f"Processed {i}/{len(nc_files)} years...")

df = pd.DataFrame(records).sort_values("year")
df.to_csv("mizoram_grid_wise_rainfall_1901_2025.csv", index=False)
print("\nSaved table: mizoram_grid_wise_rainfall_1901_2025.csv")
print(f"Valid grid count range across 1901-2025: min={df['valid_grid_count'].min()}, max={df['valid_grid_count'].max()}")

# ---------------------------------------------------------
# 4. PLOT 1: SPATIAL MAP OF BASIN & IMD GRIDS
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
basin.plot(ax=ax, facecolor="#cce6ff", edgecolor="black", linewidth=1.8, zorder=2)

# Draw all overlapping 0.25° grid boxes
for la, lo, cell in overlap_grids:
    gpd.GeoSeries([cell]).plot(ax=ax, facecolor="none", edgecolor="gray", linestyle="--", linewidth=1, zorder=1)

# Plot grid centers (red = strictly inside, orange = overlapping boundary)
strict_coords = {(la, lo) for la, lo, _ in strict_grids}
for la, lo, _ in overlap_grids:
    is_inside = (la, lo) in strict_coords
    color = "crimson" if is_inside else "darkorange"
    marker = "o" if is_inside else "s"
    ax.scatter(lo, la, color=color, marker=marker, s=65, zorder=4)
    ax.text(lo + 0.015, la + 0.012, f"({la:.2f}N, {lo:.2f}E)", fontsize=8.5, fontweight="bold" if is_inside else "normal", zorder=5)

# Custom legend
ax.scatter([], [], color="crimson", marker="o", s=65, label=f"Center Strictly Inside ({len(strict_grids)})")
ax.scatter([], [], color="darkorange", marker="s", s=65, label=f"Overlapping Boundary ({len(overlap_grids) - len(strict_grids)})")
ax.plot([], [], color="gray", linestyle="--", label="0.25° × 0.25° IMD Grid Cell")

ax.set_title("IMD 0.25° Grids Falling Inside the Mizoram Catchment", fontsize=12, fontweight="bold")
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig("mizoram_imd_grids_map.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 5. PLOT 2: GRID-WISE RAINFALL, CATCHMENT AVERAGE & GRID COUNT
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

grid_cols = [c for c in df.columns if c.startswith("Grid_")]
for col in grid_cols:
    ax1.plot(df["year"], df[col], linewidth=1.0, alpha=0.45, label=col.replace("Grid_", ""))

ax1.plot(df["year"], df["catchment_avg_mm"], color="black", linewidth=2.2, label="Catchment Average")
ax1.set_title(f"Grid-Wise vs. Averaged Annual Rainfall (1901–2025) [{mode_label} Grids]", fontsize=12, fontweight="bold")
ax1.set_ylabel("Annual Rainfall (mm)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", ncol=2, fontsize=8.5)

ax2.step(df["year"], df["valid_grid_count"], where="mid", color="steelblue", linewidth=2)
ax2.set_ylabel("Valid Grids")
ax2.set_xlabel("Year")
ax2.set_ylim(0, max(df["valid_grid_count"].max() + 2, 4))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mizoram_grid_average_timeseries.png", dpi=300)
plt.show()