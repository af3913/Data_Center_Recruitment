from pathlib import Path
import geopandas as gpd


DATA_FOLDER = Path("data")

COUNTY_SHP_PATH = (
    DATA_FOLDER
    / "county_shapefile"
    / "tl_2019_us_county.shp"
)

COUNTY_PARQUET_PATH = (
    DATA_FOLDER
    / "us_counties_simplified.parquet"
)


# Read shapefile
counties = gpd.read_file(COUNTY_SHP_PATH)

# Build five-digit county FIPS
if "GEOID" in counties.columns:
    counties["County"] = (
        counties["GEOID"]
        .astype(str)
        .str.zfill(5)
    )

elif "GEOID20" in counties.columns:
    counties["County"] = (
        counties["GEOID20"]
        .astype(str)
        .str.zfill(5)
    )

else:
    counties["County"] = (
        counties["STATEFP"].astype(str).str.zfill(2)
        + counties["COUNTYFP"].astype(str).str.zfill(3)
    )

# Convert once to the geographic CRS used by Plotly
counties = counties.to_crs(epsg=4326)

# Simplify county boundaries for faster web rendering.
# Preserve topology avoids creating invalid or disconnected polygons.
counties["geometry"] = counties.geometry.simplify(
    tolerance=0.01,
    preserve_topology=True
)

# Keep only fields needed by the app
keep_cols = [
    col for col in [
        "County",
        "NAME",
        "STATEFP",
        "COUNTYFP",
        "geometry"
    ]
    if col in counties.columns
]

counties = counties[keep_cols].copy()

# Save efficient GeoParquet
counties.to_parquet(
    COUNTY_PARQUET_PATH,
    index=False
)

print(f"Saved: {COUNTY_PARQUET_PATH}")
print(f"Counties: {len(counties):,}")