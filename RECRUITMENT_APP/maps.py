from pathlib import Path
import json
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_soc_recruitment_network_map(
    soc_code,
    recruitment_df,
    counties_gdf,
    output_folder,
    destination_county_fips="56021",
    destination_name="Laramie County, WY",
    county_col="County",
    origin_fips_col="Origin County FIPS",
    max_counties=25,
    title_name=None,
    color_quantile_cap=0.95,
    save_html=True
):
    """
    Create an interactive recruitment network map for a selected SOC.

    Map elements
    ------------
    County fill:
        Number of 2025 jobs for the selected SOC.

    Origin marker size:
        Number of in-migrants from the origin county to the destination.

    Line width:
        Number of in-migrants from the origin county to the destination.

    Recruitment score:
        Weighted combination of:
            - National Jobs Percentile
            - Migration Percentile
    """

    counties = counties_gdf.copy()
    df = recruitment_df.copy()

    # ------------------------------------------------------------
    # Validate recruitment columns
    # ------------------------------------------------------------
    required_df_cols = [
        origin_fips_col,
        "Origin State",
        "Origin County",
        "In-Migrants",
        "SOC 2025 Jobs",
        "National Jobs Percentile",
        "Migration Percentile",
        "Recruitment Score",
        "Recruitment Rank",
        "Eligible Recruitment County"
    ]

    missing_df_cols = [
        col for col in required_df_cols
        if col not in df.columns
    ]

    if missing_df_cols:
        raise KeyError(
            f"Missing required recruitment columns: {missing_df_cols}"
        )

    # ------------------------------------------------------------
    # Create county FIPS in shapefile
    # ------------------------------------------------------------
    if county_col not in counties.columns:

        if "GEOID" in counties.columns:
            counties[county_col] = (
                counties["GEOID"]
                .astype(str)
                .str.zfill(5)
            )

        elif "GEOID20" in counties.columns:
            counties[county_col] = (
                counties["GEOID20"]
                .astype(str)
                .str.zfill(5)
            )

        elif {"STATEFP", "COUNTYFP"}.issubset(counties.columns):
            counties[county_col] = (
                counties["STATEFP"]
                .astype(str)
                .str.zfill(2)
                + counties["COUNTYFP"]
                .astype(str)
                .str.zfill(3)
            )

        else:
            raise KeyError(
                "Could not construct county FIPS. Expected GEOID, "
                "GEOID20, or STATEFP and COUNTYFP."
            )

    # ------------------------------------------------------------
    # Clean county FIPS fields
    # ------------------------------------------------------------
    counties[county_col] = (
        counties[county_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    df[origin_fips_col] = (
        df[origin_fips_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    destination_county_fips = (
        str(destination_county_fips)
        .strip()
        .replace(".0", "")
        .zfill(5)
    )

    # Plotly maps require longitude and latitude
    counties = counties.to_crs(epsg=4326)

    # ------------------------------------------------------------
    # Keep top-ranked eligible recruitment counties
    # ------------------------------------------------------------
    plot_df = (
        df[
            df["Eligible Recruitment County"]
            & df["Recruitment Score"].notna()
        ]
        .sort_values(
            by=[
                "Recruitment Score",
                "SOC 2025 Jobs",
                "In-Migrants"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .head(max_counties)
        .copy()
    )

    if plot_df.empty:
        raise ValueError(
            f"No eligible recruitment counties found for SOC {soc_code}"
        )

    # ------------------------------------------------------------
    # Merge recruitment data to county polygons
    # ------------------------------------------------------------
    map_gdf = counties.merge(
        plot_df,
        left_on=county_col,
        right_on=origin_fips_col,
        how="inner"
    )

    if map_gdf.empty:
        raise ValueError(
            "No county geometries matched the recruitment data."
        )

    # ------------------------------------------------------------
    # Create representative county points for lines and markers
    # ------------------------------------------------------------
    county_points = counties.copy()

    county_points["point_geometry"] = (
        county_points.geometry.representative_point()
    )

    county_points["lon"] = (
        county_points["point_geometry"].x
    )

    county_points["lat"] = (
        county_points["point_geometry"].y
    )

    coordinates = county_points[
        [
            county_col,
            "lon",
            "lat"
        ]
    ].copy()

    # Add origin coordinates
    plot_df = plot_df.merge(
        coordinates.rename(columns={
            county_col: origin_fips_col,
            "lon": "origin_lon",
            "lat": "origin_lat"
        }),
        on=origin_fips_col,
        how="left"
    )

    # Get destination coordinates
    destination = coordinates[
        coordinates[county_col].eq(
            destination_county_fips
        )
    ]

    if destination.empty:
        raise ValueError(
            f"Destination county FIPS {destination_county_fips} "
            "was not found in counties_gdf."
        )

    dest_lon = destination["lon"].iloc[0]
    dest_lat = destination["lat"].iloc[0]

    plot_df = plot_df.dropna(
        subset=[
            "origin_lon",
            "origin_lat"
        ]
    ).copy()

    # ------------------------------------------------------------
    # Formatting helper for custom hover text
    # ------------------------------------------------------------
    def fmt_num(value, decimals=0, dollar=False, percent=False):
        if pd.isna(value):
            return "N/A"

        if percent:
            return f"{value:.{decimals}%}"

        formatted = f"{value:,.{decimals}f}"

        if dollar:
            return f"${formatted}"

        return formatted

    # ------------------------------------------------------------
    # Prepare choropleth color scale
    # ------------------------------------------------------------
    geojson = json.loads(
        map_gdf.to_json()
    )

    color_min = map_gdf[
        "SOC 2025 Jobs"
    ].min()

    color_max = map_gdf[
        "SOC 2025 Jobs"
    ].quantile(
        color_quantile_cap
    )

    if pd.isna(color_min) or pd.isna(color_max):
        raise ValueError(
            "SOC 2025 Jobs contains no valid values for mapping."
        )

    if color_min == color_max:
        color_max = color_min + 1

    label = (
        title_name
        if title_name
        else f"SOC {soc_code}"
    )

    # ------------------------------------------------------------
    # Polygon hover fields
    # ------------------------------------------------------------
    hover_data = {
        "Origin State": True,
        origin_fips_col: True,
        "SOC 2025 Jobs": ":,.0f",
        "National Jobs Percentile": ":.2f",
        "Migration Percentile": ":.2f",
        "Recruitment Score": ":.2f",
        "Recruitment Rank": ":,.0f"
    }

    optional_hover_formats = {
        "Median Hourly Earnings": ":$,.2f",
    }

    # Only add optional fields that actually exist
    for col, display_format in optional_hover_formats.items():
        if col in map_gdf.columns:
            hover_data[col] = display_format

    # Remove rank columns if not present
    hover_data = {
        col: display_format
        for col, display_format in hover_data.items()
        if col in map_gdf.columns
    }

    # ------------------------------------------------------------
    # Base choropleth
    # ------------------------------------------------------------
    fig = px.choropleth_mapbox(
        map_gdf,
        geojson=geojson,
        locations=county_col,
        featureidkey=f"properties.{county_col}",
        color="SOC 2025 Jobs",
        hover_name=(
            "Origin County"
            if "Origin County" in map_gdf.columns
            else county_col
        ),
        hover_data=hover_data,
        color_continuous_scale="Viridis",
        range_color=(
            color_min,
            color_max
        ),
        mapbox_style="carto-positron",
        center={
            "lat": 39.5,
            "lon": -98.35
        },
        zoom=3,
        opacity=0.75,
        title=(
                f"Priority Recruitment Markets<br>"
                f"{label} → {destination_name}"
                f"<br><sup>"
                f"County color = 2025 SOC jobs | "
                f"Line width = In-migrants | "
                f"Marker size = Recruitment Score"
                f"</sup>"
)
    )

    # ------------------------------------------------------------
    # Scale migration lines and markers
    # ------------------------------------------------------------
    max_migrants = max(
        plot_df["In-Migrants"].max(),
        1
    )

    # ------------------------------------------------------------
    # Add migration-flow lines
    # ------------------------------------------------------------
    for _, row in plot_df.iterrows():

        line_width = (
            1
            + 5
            * row["In-Migrants"]
            / max_migrants
        )

        line_hover = (
            f"<b>{row['Origin County']}, {row['Origin State']}</b>"
            f"<br>Destination: {destination_name}"
            f"<br>2025 SOC Jobs: "
            f"{fmt_num(row.get('SOC 2025 Jobs'), 0)}"
            f"<br>National Jobs Percentile: "
            f"{fmt_num(row.get('National Jobs Percentile'), 1, percent=True)}"
            f"<br>Average Hourly Earnings: "
            f"{fmt_num(row.get('Avg. Hourly Earnings'), 2, dollar=True)}"
            f"<br>Migration Percentile: "
            f"{fmt_num(row.get('Migration Percentile'), 1, percent=True)}"
            f"<br>Recruitment Score: "
            f"{fmt_num(row.get('Recruitment Score'), 3)}"
            f"<br>Recruitment Rank: "
            f"{fmt_num(row.get('Recruitment Rank'), 0)}"
        )

        fig.add_trace(
            go.Scattermapbox(
                lon=[row["origin_lon"], dest_lon],
                lat=[row["origin_lat"], dest_lat],
                mode="lines",
                line=dict(
                    width=line_width,
                    color="lightgray"
                ),
                opacity=0.65,
                hoverinfo="skip",
                showlegend=False
            )
        )

    # ------------------------------------------------------------
    # Add origin-county markers
    # Marker size based on Recruitment Score
    # ------------------------------------------------------------

    max_score = plot_df["Recruitment Score"].max()
    min_score = plot_df["Recruitment Score"].min()

    # Handle the rare case where all scores are identical
    if max_score == min_score:
        marker_sizes = [16] * len(plot_df)
    else:
        marker_sizes = (
            8
            + 18
            * (
                (plot_df["Recruitment Score"] - min_score)
                / (max_score - min_score)
            )
     )

    marker_hover_text = [
    (
        f"<b>{row['Origin County']}, {row['Origin State']}</b>"
        f"<br>Origin County FIPS: {row[origin_fips_col]}"
        f"<br>2025 SOC Jobs: "
        f"{fmt_num(row.get('SOC 2025 Jobs'), 0)}"
        f"<br>National Jobs Percentile: "
        f"{fmt_num(
            row.get('National Jobs Percentile'),
            2,
            percent=True
        )}"
        f"<br>Migration Percentile: "
        f"{fmt_num(
            row.get('Migration Percentile'),
            2,
            percent=True
        )}"
        f"<br>Median Hourly Earnings: "
        f"{fmt_num(
            row.get('Median Hourly Earnings'),
            2,
            dollar=True
        )}"
        f"<br>Recruitment Score: "
        f"{fmt_num(row.get('Recruitment Score'), 2)}"
        f"<br>Recruitment Rank: "
        f"{fmt_num(row.get('Recruitment Rank'), 0)}"
    )
        for _, row in plot_df.iterrows()
    ]

    fig.add_trace(
        go.Scattermapbox(
            lon=plot_df["origin_lon"],
            lat=plot_df["origin_lat"],
            mode="markers",
            marker={
                "size": marker_sizes,
                "opacity": 0.9
            },
            text=marker_hover_text,
            hoverinfo="text",
            name="Priority recruitment origin"
        )
    )

    # ------------------------------------------------------------
    # Add destination marker
    # ------------------------------------------------------------
    fig.add_trace(
        go.Scattermapbox(
            lon=[dest_lon],
            lat=[dest_lat],
            mode="markers",
            marker={
                "size": 18
            },
            text=[
                (
                    f"<b>{destination_name}</b>"
                    f"<br>Destination County FIPS: "
                    f"{destination_county_fips}"
                )
            ],
            hoverinfo="text",
            name="Destination county",
            showlegend=False
        )
    )

    # ------------------------------------------------------------
    # Final layout
    # ------------------------------------------------------------
    fig.update_layout(
    margin=dict(
        l=0,
        r=20,
        t=95,
        b=0
    ),

    legend=dict(
        title="Map Elements",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=11)
    ),

    coloraxis_colorbar=dict(
        title="2025 SOC Jobs",
        thickness=18,
        len=0.75,
        y=0.5,
        yanchor="middle",
        x=1.02
    )
)

    # ------------------------------------------------------------
    # Save interactive HTML
    # ------------------------------------------------------------
    out_path = None

    if save_html:

        output_folder = Path(output_folder)
        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        clean_soc = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            soc_code
        )

        out_path = output_folder / (
            f"SOC_{clean_soc}_recruitment_network_map.html"
        )

        fig.write_html(
            out_path
        )

        print(
            f"Saved: {out_path}"
        )

    return fig, out_path, plot_df
