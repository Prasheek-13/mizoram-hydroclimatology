import os
import gzip
import shutil
import requests
import rasterio
from rasterio.merge import merge
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape, Point
import whitebox
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
TARGET_LON = 92.9653
TARGET_LAT = 23.0344
TILES = ["N23E092", "N23E093", "N22E092", "N22E093"]
WORK_DIR = os.getcwd()
DATA_DIR = os.path.join(WORK_DIR, "dem_tiles")
MERGED_DEM = os.path.join(WORK_DIR, "merged_dem.tif")
OUTPUT_SHP = os.path.join(WORK_DIR, "mizoram_basin.shp")

os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------
# 2. DOWNLOAD & UNZIP SRTM TILES
# ---------------------------------------------------------
hgt_files = []
for tile in TILES:
    lat_folder = tile[:3]
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_folder}/{tile}.hgt.gz"
    gz_path = os.path.join(DATA_DIR, f"{tile}.hgt.gz")
    hgt_path = os.path.join(DATA_DIR, f"{tile}.hgt")

    if not os.path.exists(hgt_path):
        print(f"Downloading {tile}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(gz_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        with gzip.open(gz_path, "rb") as f_in, open(hgt_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)
    hgt_files.append(hgt_path)

# ---------------------------------------------------------
# 3. MERGE TILES INTO ONE DEM
# ---------------------------------------------------------
print("Merging DEM tiles...")
src_files = [rasterio.open(fp) for fp in hgt_files]
mosaic, out_trans = merge(src_files)
out_meta = src_files[0].meta.copy()
out_meta.update({
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_trans,
    "dtype": "float32"
})
with rasterio.open(MERGED_DEM, "w", **out_meta) as dest:
    dest.write(mosaic.astype("float32"))
for src in src_files:
    src.close()

# ---------------------------------------------------------
# 4. HYDROLOGY & AUTOMATIC SNAPPING (WhiteboxTools)
# ---------------------------------------------------------
wbt = whitebox.WhiteboxTools()
wbt.set_working_dir(WORK_DIR)
wbt.verbose = False

filled_dem = os.path.join(WORK_DIR, "filled_dem.tif")
d8_pointer = os.path.join(WORK_DIR, "d8_pointer.tif")
flow_acc = os.path.join(WORK_DIR, "flow_acc.tif")
pour_pt_shp = os.path.join(WORK_DIR, "pour_point.shp")
snapped_pt_shp = os.path.join(WORK_DIR, "snapped_point.shp")
watershed_tif = os.path.join(WORK_DIR, "watershed.tif")

# Save target coordinate as a point shapefile
gpd.GeoDataFrame(geometry=[Point(TARGET_LON, TARGET_LAT)], crs="EPSG:4326").to_file(pour_pt_shp)

print("Filling depressions (Wang & Liu)...")
wbt.fill_depressions_wang_and_liu(MERGED_DEM, filled_dem)

print("Calculating D8 flow direction & accumulation...")
wbt.d8_pointer(filled_dem, d8_pointer)
wbt.d8_flow_accumulation(filled_dem, flow_acc)

print("Snapping pour point to nearest main stream...")
# snap_dist is in degrees (~0.005 deg ≈ 550 meters search radius)
wbt.snap_pour_points(pour_pt_shp, flow_acc, snapped_pt_shp, snap_dist=0.005)

print("Delineating watershed...")
wbt.watershed(d8_pointer, snapped_pt_shp, watershed_tif)

# ---------------------------------------------------------
# 5. POLYGONIZE & EXPORT SHAPEFILE
# ---------------------------------------------------------
print("Converting watershed raster to shapefile...")
with rasterio.open(watershed_tif) as src:
    image = src.read(1)
    mask = image > 0
    results = (
        {"properties": {"basin": int(v)}, "geometry": s}
        for s, v in shapes(image, mask=mask, transform=src.transform)
    )
    geoms = list(results)

gdf = gpd.GeoDataFrame.from_features(geoms, crs="EPSG:4326").dissolve()
gdf.to_file(OUTPUT_SHP)
print(f"Success! Saved catchment shapefile to {OUTPUT_SHP}")

# ---------------------------------------------------------
# 6. VISUAL VERIFICATION
# ---------------------------------------------------------
snapped_gdf = gpd.read_file(snapped_pt_shp)
fig, ax = plt.subplots(figsize=(8, 8))
gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue")
snapped_gdf.plot(ax=ax, color="red", markersize=50, label="Snapped Pour Point")
ax.set_title("Delineated Mizoram River Basin")
ax.legend()
plt.show()