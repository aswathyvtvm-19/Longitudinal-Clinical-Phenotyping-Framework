# ====================================================
# Section 2 & 3: Final BAV vs Non-BAV Comparison + Subgroup Analysis (2000–2019)
# ====================================================
# - Section 2: Comorbidity group-level comparison (TOP_CAT_FIRST_*)
# - Section 3: Sub-disease subgroup comparison (FIRST_*)
#
# Key rules (as per your final logic):
# - Diagnosis counted only if DATE column year <= analysis year
# - Weighted prevalence (%) = diagnosed_count / total_patients * 100
# - Bootstrap 95% CI (n=1000) for difference
# - Welch t-test for p-value
# - Suppression if BAV count < 10 OR Non-BAV count < 10
# - Saves yearly full tables + yearly >=10-only tables
# - Produces final long-format outputs for:
#   (a) top 10 comorbidities (counts>=10 only)
#   (b) top 15 subgroups (counts>=10 only)
# ====================================================

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.ticker import MaxNLocator


# ----------------------------
# Utility: Weighted prevalence
# ----------------------------
def calculate_weighted_prevalence(
    df: pd.DataFrame,
    comorbidity_cols: List[str],
    year: int
) -> pd.Series:
    """
    Calculates prevalence (%) for comorbidities diagnosed on or before the given year.
    Requires corresponding date columns: f"{col}_DATE"
    """
    prevalence: Dict[str, float] = {}
    total_patients = len(df)

    if total_patients == 0:
        return pd.Series({c: 0.0 for c in comorbidity_cols})

    for comorb in comorbidity_cols:
        date_col = f"{comorb}_DATE"
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            diagnosed = dates.dt.year.le(year)  # diagnosis year <= current year
            prevalence[comorb] = round(diagnosed.sum() / total_patients * 100, 2)
        else:
            prevalence[comorb] = 0.0

    return pd.Series(prevalence)


# ----------------------------
# Utility: Bootstrap 95% CI
# ----------------------------
def bootstrap_difference(
    bav_vals: np.ndarray,
    nonbav_vals: np.ndarray,
    n_bootstrap: int = 1000,
    seed: Optional[int] = 42
) -> Tuple[float, float]:
    """
    Bootstraps the prevalence difference (BAV - Non-BAV) in percentage points.
    Inputs are 0/1 arrays.
    Returns (lower_ci, upper_ci) for 95% CI.
    """
    rng = np.random.default_rng(seed)
    boot_diffs = []

    bav_vals = np.asarray(bav_vals, dtype=float)
    nonbav_vals = np.asarray(nonbav_vals, dtype=float)

    for _ in range(n_bootstrap):
        b = rng.choice(bav_vals, size=len(bav_vals), replace=True)
        n = rng.choice(nonbav_vals, size=len(nonbav_vals), replace=True)
        boot_diffs.append(b.mean() * 100 - n.mean() * 100)

    lower_ci = round(np.percentile(boot_diffs, 2.5), 2)
    upper_ci = round(np.percentile(boot_diffs, 97.5), 2)
    return lower_ci, upper_ci


# ------------------------------------------------
# Section 2: Year-wise BAV vs Non-BAV comparisons
# ------------------------------------------------
def compare_bav_nonbav(
    data_directory: str,
    years: range,
    out_dir: str = "Section2_Results",
    top_n: int = 10,
    n_bootstrap: int = 1000
) -> Tuple[pd.DataFrame, List[str], Dict[int, pd.DataFrame]]:
    """
    Compares comorbidity prevalence between BAV (data['original']) and Non-BAV (data['matched'])
    for each year.

    Writes per-year:
      - comparison_{year}.csv                       (full, includes suppressed rows)
      - comparison_{year}_counts_ge10.csv           (only rows where BOTH counts >= 10)
      - comparison_top10_{year}.csv                 (top 10 by difference, may include suppressed)

    Returns:
      final_df: long-format table across all years
      top_comorbidities: list of top-N comorbidity names by mean (non-suppressed) difference
      yearly_tables: dict[year] = full year table
    """
    os.makedirs(out_dir, exist_ok=True)

    all_results: List[Dict] = []
    yearly_tables: Dict[int, pd.DataFrame] = {}

    for year in years:
        file_path = os.path.join(data_directory, f"{year}.pkl")
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file for year {year}: {file_path}")
            continue

        data = pd.read_pickle(file_path)
        bav_df = data["original"]
        nonbav_df = data["matched"]

        # Comorbidity flag columns (exclude _DATE)
        comorbidity_cols = [
            c for c in bav_df.columns
            if c.startswith("TOP_CAT_FIRST_") and not c.endswith("_DATE")
        ]

        # Weighted prevalence (%)
        bav_wp = calculate_weighted_prevalence(bav_df, comorbidity_cols, year)
        nonbav_wp = calculate_weighted_prevalence(nonbav_df, comorbidity_cols, year)

        year_results: List[Dict] = []

        for comorb in comorbidity_cols:
            date_col = f"{comorb}_DATE"

            # If date columns missing, treat as 0 diagnosed (but still write row)
            if date_col not in bav_df.columns or date_col not in nonbav_df.columns:
                bav_count = 0
                nonbav_count = 0
                suppressed = True
                bav_values = np.array([])
                nonbav_values = np.array([])
            else:
                bav_dates = pd.to_datetime(bav_df[date_col], errors="coerce")
                nonbav_dates = pd.to_datetime(nonbav_df[date_col], errors="coerce")

                bav_mask = bav_dates.dt.year.le(year)
                nonbav_mask = nonbav_dates.dt.year.le(year)

                bav_count = int(bav_mask.sum())
                nonbav_count = int(nonbav_mask.sum())

                bav_values = bav_mask.astype(float).values
                nonbav_values = nonbav_mask.astype(float).values
                suppressed = (bav_count < 10) or (nonbav_count < 10)

            display_name = comorb.replace("TOP_CAT_FIRST_", "").replace("_", " ")

            if suppressed:
                result = {
                    "Comorbidity": display_name,
                    "Year": year,
                    "BAV Patients (n)": bav_count,
                    "Non-BAV Patients (n)": nonbav_count,
                    "BAV Weighted Prevalence (%)": "Suppressed (<10)",
                    "Non-BAV Weighted Prevalence (%)": "Suppressed (<10)",
                    "Weighted Prevalence Difference (%)": "Suppressed (<10)",
                    "95% CI Lower": "Suppressed (<10)",
                    "95% CI Upper": "Suppressed (<10)",
                    "P-Value": "Suppressed (<10)",
                    "P-Value (Rounded)": "Suppressed (<10)",
                }
            else:
                lower_ci, upper_ci = bootstrap_difference(
                    bav_values, nonbav_values,
                    n_bootstrap=n_bootstrap,
                    seed=42
                )
                _, p_val = ttest_ind(bav_values, nonbav_values, equal_var=False)

                result = {
                    "Comorbidity": display_name,
                    "Year": year,
                    "BAV Patients (n)": bav_count,
                    "Non-BAV Patients (n)": nonbav_count,
                    "BAV Weighted Prevalence (%)": bav_wp[comorb],
                    "Non-BAV Weighted Prevalence (%)": nonbav_wp[comorb],
                    "Weighted Prevalence Difference (%)": round(bav_wp[comorb] - nonbav_wp[comorb], 2),
                    "95% CI Lower": lower_ci,
                    "95% CI Upper": upper_ci,
                    "P-Value": float(p_val),
                    "P-Value (Rounded)": round(float(p_val), 4),
                }

            year_results.append(result)
            all_results.append(result)

        # SAFE SORT: sort by numeric difference (suppressed -> NaN -> bottom)
        df_year = pd.DataFrame(year_results)
        df_year["_sort_diff"] = pd.to_numeric(df_year["Weighted Prevalence Difference (%)"], errors="coerce")
        df_year = (
            df_year.sort_values(by="_sort_diff", ascending=False, na_position="last")
                  .drop(columns=["_sort_diff"])
                  .reset_index(drop=True)
        )

        df_year.to_csv(os.path.join(out_dir, f"comparison_{year}.csv"), index=False)
        df_year.head(10).to_csv(os.path.join(out_dir, f"comparison_top10_{year}.csv"), index=False)

        df_year_ge10 = df_year[
            (df_year["BAV Patients (n)"] >= 10) & (df_year["Non-BAV Patients (n)"] >= 10)
        ].copy()
        df_year_ge10.to_csv(os.path.join(out_dir, f"comparison_{year}_counts_ge10.csv"), index=False)

        yearly_tables[year] = df_year
        print(f"✅ Saved Section 2 outputs for year {year}")

    final_df = pd.DataFrame(all_results)

    # Top comorbidities by MEAN numeric difference across years (ignoring suppressed)
    numeric_final = final_df[pd.to_numeric(final_df["Weighted Prevalence Difference (%)"], errors="coerce").notnull()].copy()
    numeric_final["Weighted Prevalence Difference (%)"] = pd.to_numeric(
        numeric_final["Weighted Prevalence Difference (%)"], errors="coerce"
    )

    avg_diff_df = (
        numeric_final.groupby("Comorbidity", as_index=False)["Weighted Prevalence Difference (%)"]
                    .mean()
                    .sort_values("Weighted Prevalence Difference (%)", ascending=False)
                    .reset_index(drop=True)
    )

    top_comorbidities = avg_diff_df.head(top_n)["Comorbidity"].tolist()

    print("\n🏆 Final Top Comorbidities (by mean difference, non-suppressed only):")
    for i, c in enumerate(top_comorbidities, 1):
        print(f"{i}. {c}")

    return final_df, top_comorbidities, yearly_tables


# --------------------------------------
# Section 2 helper: final top10 long CSV
# --------------------------------------
def write_final_top10_csv(
    final_df: pd.DataFrame,
    top_comorbidities: List[str],
    out_path: str
) -> pd.DataFrame:
    """
    Writes long-format CSV for top comorbidities containing ONLY rows where both counts >= 10.
    """
    df = final_df.copy()
    df = df[df["Comorbidity"].isin(top_comorbidities)]
    df = df[(df["BAV Patients (n)"] >= 10) & (df["Non-BAV Patients (n)"] >= 10)]

    keep_cols = [
        "Comorbidity", "Year",
        "BAV Patients (n)", "Non-BAV Patients (n)",
        "BAV Weighted Prevalence (%)", "Non-BAV Weighted Prevalence (%)"
    ]
    df = df[keep_cols].sort_values(["Comorbidity", "Year"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Saved final top-10 long CSV: {out_path}")
    return df

