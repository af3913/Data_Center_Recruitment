import numpy as np
import pandas as pd


# Migration flow function
def get_top_migration_origins(
    excel_path,
    destination_state,
    destination_county,
    top_n=100
):
    df = pd.read_excel(
        excel_path,
        sheet_name=destination_state,
        header=[1, 2]
    )

    # Flatten two-row headers
    df.columns = [
        f"{a} {b}".strip()
        if not str(b).startswith("Unnamed")
        else str(a).strip()
        for a, b in df.columns
    ]

    df = df.rename(columns={
        "State Code of Geography A": "Dest State Code",
        "FIPS County Code of Geography A": "Dest County Code",
        "State/U.S. Island Area/Foreign Region Code of Geography B": "Origin State Code",
        "FIPS County Code of Geography B": "Origin County Code",
        "State Name of Geography A": "Dest State",
        "County Name of Geography A": "Dest County",
        "State/U.S. Island Area/Foreign Region of Geography B": "Origin State",
        "County Name of Geography B": "Origin County",
        "Flow from Geography B to Geography A Estimate": "In-Migrants",
        "Flow from Geography B to Geography A MOE": "In-Migrants MOE",
        "Counterflow from Geography A to Geography B1 Estimate": "Out-Migrants",
        "Counterflow from Geography A to Geography B1 MOE": "Out-Migrants MOE",
        "Net Migration from Geography B to Geography A1 Estimate": "Net Migration",
        "Net Migration from Geography B to Geography A1 MOE": "Net Migration MOE",
        "Gross Migration between Geography A and Geography B1 Estimate": "Gross Migration",
        "Gross Migration between Geography A and Geography B1 MOE": "Gross Migration MOE",
    })

    # Filter to destination county
    df = df[df["Dest County"].eq(destination_county)].copy()

    # Clean numeric migration columns
    numeric_cols = [
        "In-Migrants", "In-Migrants MOE",
        "Out-Migrants", "Out-Migrants MOE",
        "Net Migration", "Net Migration MOE",
        "Gross Migration", "Gross Migration MOE"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Clean origin codes and build origin county FIPS
    df["Origin State Code"] = (
        pd.to_numeric(df["Origin State Code"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )

    df["Origin County Code"] = (
        pd.to_numeric(df["Origin County Code"], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
        .str.zfill(3)
    )

    df["Origin County FIPS"] = (
        df["Origin State Code"] + df["Origin County Code"]
    )

    return (
        df[[
            "Dest State", "Dest County",
            "Origin State", "Origin County",
            "Origin State Code", "Origin County Code", "Origin County FIPS",
            "In-Migrants", "In-Migrants MOE",
            "Out-Migrants", "Out-Migrants MOE",
            "Net Migration", "Net Migration MOE",
            "Gross Migration", "Gross Migration MOE"
        ]]
        .sort_values("In-Migrants", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# PATH for migration data Excel file
migration_path = (
    r"V:\# Consultancy\Projects\Google\Labor market analysis\03 analysis\Migration flows county to county\county-to-county-2016-2020-ins-outs-nets-gross.xlsx"
)

# Example usage for Laramie County, Wyoming
laramie_origins = get_top_migration_origins(
    excel_path=migration_path,
    destination_state="Wyoming",
    destination_county="Laramie County",
    top_n=50
)

laramie_origins.head()


def analyze_soc_recruitment_origins(
    soc_code,
    occ_df,
    migration_df,
    jobs_col="2025 Jobs",
    county_col="County",
    migration_county_col="Origin County FIPS",
    min_jobs_threshold=100,
    jobs_weight=0.50,
    migration_weight=0.50
):
    """
    Rank migration-origin counties as potential recruitment markets for one SOC.

    Recruitment Score =
        jobs_weight * National Jobs Percentile
        + migration_weight * Migration Percentile

    National Jobs Percentile:
        Rank of the county's SOC employment among all U.S. counties
        meeting the minimum jobs threshold for that SOC.

    Migration Percentile:
        Rank of the county's in-migration flow among all migration-origin
        counties included in migration_df.
    """

    if not np.isclose(jobs_weight + migration_weight, 1.0):
        raise ValueError(
            "jobs_weight and migration_weight must sum to 1."
        )

    occ = occ_df.copy()
    migration = migration_df.copy()

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------
    required_occ_cols = [
        "SOC",
        county_col,
        jobs_col
    ]

    missing_occ_cols = [
        col for col in required_occ_cols
        if col not in occ.columns
    ]

    if missing_occ_cols:
        raise KeyError(
            f"Missing required columns in occ_df: {missing_occ_cols}"
        )

    required_migration_cols = [
        migration_county_col,
        "In-Migrants"
    ]

    missing_migration_cols = [
        col for col in required_migration_cols
        if col not in migration.columns
    ]

    if missing_migration_cols:
        raise KeyError(
            f"Missing required columns in migration_df: "
            f"{missing_migration_cols}"
        )

    # ------------------------------------------------------------
    # Clean county FIPS
    # ------------------------------------------------------------
    occ[county_col] = (
        occ[county_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    migration[migration_county_col] = (
        migration[migration_county_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # ------------------------------------------------------------
    # Clean numeric variables
    # ------------------------------------------------------------
    numeric_occ_cols = [
        jobs_col,
        "2025 Resident Workers",
        "2025 Net Commuters",
        "Avg. Hourly Earnings",
        "Median Hourly Earnings",
        "Median Annual Earnings"
    ]

    for col in numeric_occ_cols:
        if col in occ.columns:
            occ[col] = pd.to_numeric(
                occ[col],
                errors="coerce"
            )

    migration["In-Migrants"] = pd.to_numeric(
        migration["In-Migrants"],
        errors="coerce"
    ).fillna(0)

    # ------------------------------------------------------------
    # Subset national occupational data to selected SOC
    # ------------------------------------------------------------
    soc_jobs = occ[
        occ["SOC"].eq(soc_code)
    ].copy()

    if soc_jobs.empty:
        raise ValueError(
            f"No occupational records found for SOC {soc_code}"
        )

    # Keep counties with a measurable and sufficiently large labor pool
    soc_jobs = soc_jobs[
        soc_jobs[jobs_col].notna()
        & (soc_jobs[jobs_col] >= min_jobs_threshold)
    ].copy()

    if soc_jobs.empty:
        raise ValueError(
            f"No counties remain for SOC {soc_code} after applying "
            f"min_jobs_threshold={min_jobs_threshold}."
        )

    # ------------------------------------------------------------
    # National jobs percentile across all eligible U.S. counties
    # ------------------------------------------------------------
    soc_jobs["National Jobs Percentile"] = (
        soc_jobs[jobs_col]
        .rank(
            pct=True,
            method="average"
        )
    )

    # Optional national job rank, where 1 is the largest labor pool
    soc_jobs["National Jobs Rank"] = (
        soc_jobs[jobs_col]
        .rank(
            ascending=False,
            method="min"
        )
    )

    # ------------------------------------------------------------
    # Preserve informative labor-market fields
    # ------------------------------------------------------------
    keep_cols = [
        "SOC",
        "State",
        county_col,
        "County Name",
        jobs_col,
        "National Jobs Percentile",
        "National Jobs Rank"
    ]

    optional_cols = [
        "2025 Resident Workers",
        "2025 Net Commuters",
        "Avg. Hourly Earnings",
        "Median Hourly Earnings",
        "Median Annual Earnings"
    ]

    keep_cols += [
        col for col in optional_cols
        if col in soc_jobs.columns
    ]

    soc_jobs = soc_jobs[keep_cols].copy()

    soc_jobs = soc_jobs.rename(columns={
        county_col: migration_county_col,
        jobs_col: "SOC 2025 Jobs"
    })

    # ------------------------------------------------------------
    # Migration percentile among all included origin counties
    # ------------------------------------------------------------
    migration["Migration Percentile"] = (
        migration["In-Migrants"]
        .rank(
            pct=True,
            method="average"
        )
    )

    migration["Migration Rank"] = (
        migration["In-Migrants"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    # ------------------------------------------------------------
    # Merge migration origins with national SOC labor pool data
    # ------------------------------------------------------------
    recruitment_df = migration.merge(
        soc_jobs,
        on=migration_county_col,
        how="left"
    )

    # Eligible only when origin county meets the national jobs threshold
    recruitment_df["Eligible Recruitment County"] = (
        recruitment_df["SOC 2025 Jobs"].notna()
    )

    eligible = recruitment_df["Eligible Recruitment County"]

    # ------------------------------------------------------------
    # Composite recruitment score
    # ------------------------------------------------------------
    recruitment_df.loc[eligible, "Recruitment Score"] = (
        jobs_weight
        * recruitment_df.loc[
            eligible,
            "National Jobs Percentile"
        ]
        + migration_weight
        * recruitment_df.loc[
            eligible,
            "Migration Percentile"
        ]
    )

    recruitment_df["Recruitment Rank"] = (
        recruitment_df["Recruitment Score"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    recruitment_df = recruitment_df.sort_values(
        by=[
            "Recruitment Score",
            "SOC 2025 Jobs",
            "In-Migrants"
        ],
        ascending=[
            False,
            False,
            False
        ],
        na_position="last"
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------
    total_origins = len(recruitment_df)

    eligible_origins = int(
        recruitment_df[
            "Eligible Recruitment County"
        ].sum()
    )

    total_inmigrants = recruitment_df[
        "In-Migrants"
    ].sum()

    eligible_inmigrants = recruitment_df.loc[
        recruitment_df["Eligible Recruitment County"],
        "In-Migrants"
    ].sum()

    summary = pd.DataFrame({
        "SOC": [soc_code],
        "Migration Origin Counties Considered": [
            total_origins
        ],
        "Eligible Recruitment Counties": [
            eligible_origins
        ],
        "Share of Origins Meeting Jobs Threshold": [
            eligible_origins / total_origins
            if total_origins > 0
            else 0
        ],
        "Total In-Migrants from Origins": [
            total_inmigrants
        ],
        "In-Migrants from Eligible Recruitment Counties": [
            eligible_inmigrants
        ],
        "Share of In-Migrants from Eligible Recruitment Counties": [
            eligible_inmigrants / total_inmigrants
            if total_inmigrants > 0
            else 0
        ],
        "Minimum Jobs Threshold": [
            min_jobs_threshold
        ],
        "Jobs Benchmark": [
            "All eligible U.S. counties for selected SOC"
        ],
        "Migration Benchmark": [
            "All counties included in migration_df"
        ],
        "Jobs Weight": [
            jobs_weight
        ],
        "Migration Weight": [
            migration_weight
        ]
    })

    return recruitment_df, summary
