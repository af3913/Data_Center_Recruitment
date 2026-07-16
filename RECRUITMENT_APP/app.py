from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st

from analysis import (
    get_top_migration_origins,
    analyze_soc_recruitment_origins
)

from maps import plot_soc_recruitment_network_map


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Data Center Recruitment Markets",
    layout="wide"
)

st.title("Data Center Labor Recruitment Explorer")


# ============================================================
# Paths
# ============================================================

DATA_FOLDER = Path("data")

OCC_PATH = DATA_FOLDER / "occupation_county.csv"
SITE_PATH = DATA_FOLDER / "sites.csv"
MIGRATION_PATH = DATA_FOLDER / "migration_flows.xlsx"
COUNTY_PARQUET_PATH = DATA_FOLDER / "us_counties_simplified.parquet"


# ============================================================
# Cached data loading
# ============================================================

@st.cache_data(show_spinner="Loading occupational data...")
def load_occupation_data():

    df = pd.read_csv(
        OCC_PATH,
        dtype={
            "County": "string",
            "SOC": "string",
            "state_fips": "string"
        }
    )

    # Clean SOC codes
    df["SOC"] = (
        df["SOC"]
        .astype("string")
        .str.strip()
    )

    # Clean county FIPS
    df["County"] = (
        df["County"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)", expand=False)
        .str.zfill(5)
    )

    # Clean jobs explicitly
    df["2025 Jobs"] = (
        df["2025 Jobs"]
        .astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .replace({
            "<10": pd.NA,
            "Insf. Data": pd.NA,
            "Insufficient Data": pd.NA,
            "nan": pd.NA,
            "None": pd.NA
        })
    )

    df["2025 Jobs"] = pd.to_numeric(
        df["2025 Jobs"],
        errors="coerce"
    )

    return df


@st.cache_data(show_spinner="Loading site list...")
def load_sites():
    return pd.read_csv(
        SITE_PATH,
        dtype={"County FIPS": str}
    )


@st.cache_resource(show_spinner="Loading county geography...")
def load_counties():
    return gpd.read_parquet(
        COUNTY_PARQUET_PATH
    )

@st.cache_data
def load_migration_origins(
    destination_state,
    destination_county,
    top_n
):
    return get_top_migration_origins(
        excel_path=MIGRATION_PATH,
        destination_state=destination_state,
        destination_county=destination_county,
        top_n=top_n
    )


occ_df = load_occupation_data()
sites = load_sites()
counties_gdf = load_counties()
site_lookup = sites.set_index("Site Name")


def format_site_option(site_name):

    site = site_lookup.loc[site_name]

    return (
        f"{site['County']}, {site['State Abbreviation']}"
    )


# Sidebar controls 
soc_lookup = {
    "15-1231": "Computer Network Support Specialists",
    "15-1241": "Computer Network Architects",
    "17-2061": "Computer Hardware Engineers",
    "17-2071": "Electrical Engineers",
    "17-2112": "Industrial Engineers",
    "17-2141": "Mechanical Engineers",
    "33-9032": "Security Guards",
    "47-2031": "Carpenters",
    "47-2051": "Cement Masons and Concrete Finishers",
    "47-2061": "Construction Laborers",
    "47-2111": "Electricians",
    "47-2152": "Plumbers, Pipefitters, and Steamfitters",
    "47-2211": "Sheet Metal Workers",
    "47-2221": "Structural Iron and Steel Workers",
    "47-3013": "Helpers--Electricians",
    "47-3015": "Helpers--Pipelayers, Plumbers, Pipefitters, and Steamfitters",
    "49-2094": "Electrical and Electronics Repairers, Commercial and Industrial Equipment",
    "49-9021": "Heating, Air Conditioning, and Refrigeration Mechanics and Installers",
    "49-9052": "Telecommunications Line Installers and Repairers",
    "51-4121": "Welders, Cutters, Solderers, and Brazers",
}

with st.sidebar.form("recruitment_parameters"):

    st.header("Analysis Parameters")

    selected_site = st.selectbox(
        "Destination DC location",
        options=site_lookup.index.tolist(),
        format_func=format_site_option
    )

    selected_soc = st.selectbox(
        "Occupation",
        options=list(soc_lookup),
        format_func=lambda x: f"{x} — {soc_lookup[x]}"
    )

    top_n_origins = st.slider(
        "Migration-origin counties",
        min_value=10,
        max_value=100,
        value=50,
        step=5
    )

    min_jobs = st.number_input(
        "Minimum 2025 SOC jobs",
        min_value=0,
        value=100,
        step=25
    )

    jobs_weight = st.slider(
        "National jobs weight",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05
    )

    migration_weight = 1 - jobs_weight

    map_counties = st.slider(
        "Recruitment counties displayed",
        min_value=5,
        max_value=50,
        value=25,
        step=5
    )

    run_analysis = st.form_submit_button(
        "Run analysis",
        type="primary"
    )


if run_analysis:

    site = site_lookup.loc[selected_site]

    destination_state = site["State Name"]
    destination_county = site["County"]
    destination_fips = str(site["County FIPS"]).zfill(5)
    destination_name = (
        f"{destination_county}, {site['State Abbreviation']}"
    )

    with st.spinner("Calculating recruitment markets..."):

        migration_df = load_migration_origins(
            destination_state=destination_state,
            destination_county=destination_county,
            top_n=top_n_origins
        )


        recruitment_df, summary_df = (
            analyze_soc_recruitment_origins(
                soc_code=selected_soc,
                occ_df=occ_df,
                migration_df=migration_df,
                jobs_col="2025 Jobs",
                min_jobs_threshold=min_jobs,
                jobs_weight=jobs_weight,
                migration_weight=migration_weight
            )
        )

        

        fig, _, map_df = plot_soc_recruitment_network_map(
            soc_code=selected_soc,
            recruitment_df=recruitment_df,
            counties_gdf=counties_gdf,
            output_folder=None,
            destination_county_fips=destination_fips,
            destination_name=destination_name,
            max_counties=map_counties,
            title_name=(
                f"{soc_lookup[selected_soc]}, "
                f"SOC {selected_soc}"
            ),
            save_html=False
        )


# Display map, metrics and data
    col1, col2, col3 = st.columns(3)

    eligible_count = int(
        recruitment_df[
            "Eligible Recruitment County"
        ].sum()
    )

    col1.metric(
        "Eligible recruitment counties",
        f"{eligible_count:,}"
    )

    col2.metric(
        "Top recruitment market",
        map_df.iloc[0]["Origin County"]
    )

    col3.metric(
        "Top recruitment index",
        f"{map_df.iloc[0]['Recruitment Score']:.3f}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    display_cols = [
        "Recruitment Rank",
        "Origin State",
        "Origin County",
        "In-Migrants",
        "SOC 2025 Jobs",
        "National Jobs Percentile",
        "Migration Percentile",
        "Median Hourly Earnings",
        "Recruitment Score"
    ]

    results = map_df[
    [c for c in display_cols if c in map_df.columns]
    ].copy()

    # ------------------------------------------------------------
    # Format numeric columns for display
    # ------------------------------------------------------------

    # Round to nearest whole number
    whole_number_cols = [
    "SOC 2025 Jobs",
    "In-Migrants",
    "Recruitment Rank"
    ]

    # Round to 2 decimal places
    two_decimal_cols = [
    "National Jobs Percentile",
    "Migration Percentile",
    "Median Hourly Earnings",
    "Recruitment Score"
    ]

    for col in whole_number_cols:
        if col in results.columns:
            results[col] = results[col].round(0).astype("Int64")

    for col in two_decimal_cols:
        if col in results.columns:
            results[col] = results[col].round(2)

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download recruitment results",
        data=csv_data,
        file_name=(
            f"{destination_fips}_"
            f"{selected_soc}_recruitment.csv"
        ),
        mime="text/csv"
    )

