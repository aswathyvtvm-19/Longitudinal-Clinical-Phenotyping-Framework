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


# ----------------------------
# Section 3: subgroup mapping
# ----------------------------
def build_subgroup_mapping(
    top_comorbidities: List[str],
    excel_path: str = "Final_JAN.xlsx",
    usecols: Tuple[int, int] = (1, 4)
) -> Dict[str, List[str]]:
    """
    Builds mapping: {GroupName -> [SUBGROUP_CODE_1, ...]}

    Assumes Excel has:
      - Column index usecols[0] = subgroup code (e.g., in col B)
      - Column index usecols[1] = group code (e.g., in col E)

    Normalization:
      - Group: strip + upper
      - Subgroup: strip + replace spaces with '_' + upper
    """
    mapping_df = pd.read_excel(excel_path, usecols=list(usecols), header=0)
    mapping_df.columns = ["Subgroup", "Group"]

    mapping_df["Group"] = mapping_df["Group"].astype(str).str.strip().str.upper()
    mapping_df["Subgroup"] = (
        mapping_df["Subgroup"].astype(str)
                               .str.strip()
                               .str.replace(" ", "_", regex=False)
                               .str.upper()
    )

    subgroup_map: Dict[str, List[str]] = {}

    for group in top_comorbidities:
        group_upper = group.strip().upper()
        subgroups = (
            mapping_df.loc[mapping_df["Group"] == group_upper, "Subgroup"]
                      .dropna()
                      .unique()
                      .tolist()
        )
        subgroup_map[group] = sorted(subgroups)

    return subgroup_map


# -----------------------------------------
# Section 3: subgroup prevalence comparison
# -----------------------------------------
def subgroup_prevalence_loop(
    subgroup_map: Dict[str, List[str]],
    data_directory: str,
    years: range,
    save_dir: str = "Section3_Results",
    n_bootstrap: int = 1000
) -> pd.DataFrame:
    """
    Writes per-year:
      - subgroup_sorted_{year}.csv                 (full, includes suppressed)
      - subgroup_sorted_{year}_counts_ge10.csv     (only rows where BOTH counts >= 10)
    And overall:
      - subgroup_groupwise_overall_sorted.csv      (mean non-suppressed differences by Group/Subgroup)

    Returns:
      overall_df (long format across years)
    """
    os.makedirs(save_dir, exist_ok=True)

    all_records: List[Dict] = []

    for year in years:
        file_path = os.path.join(data_directory, f"{year}.pkl")
        if not os.path.exists(file_path):
            print(f"⚠️ Missing file for year {year}: {file_path}")
            continue

        data = pd.read_pickle(file_path)
        bav_df = data["original"]
        nonbav_df = data["matched"]

        year_records: List[Dict] = []

        for group, subgroups in subgroup_map.items():
            group_records: List[Dict] = []

            for subgroup in subgroups:
                flag_col = f"FIRST_{subgroup}"
                date_col = f"{flag_col}_DATE"

                if flag_col not in bav_df.columns or flag_col not in nonbav_df.columns:
                    continue

                if date_col not in bav_df.columns or date_col not in nonbav_df.columns:
                    bav_count, nonbav_count = 0, 0
                    suppressed = True
                    bav_vals = np.array([])
                    nonbav_vals = np.array([])
                else:
                    bav_dates = pd.to_datetime(bav_df[date_col], errors="coerce")
                    nonbav_dates = pd.to_datetime(nonbav_df[date_col], errors="coerce")

                    bav_mask = bav_dates.dt.year.le(year)
                    nonbav_mask = nonbav_dates.dt.year.le(year)

                    bav_count = int(bav_mask.sum())
                    nonbav_count = int(nonbav_mask.sum())

                    bav_vals = bav_mask.astype(float).values
                    nonbav_vals = nonbav_mask.astype(float).values
                    suppressed = (bav_count < 10) or (nonbav_count < 10)

                subgroup_name = subgroup.replace("_", " ").title()

                if suppressed:
                    record = {
                        "Group": group,
                        "Subgroup": subgroup_name,
                        "Year": year,
                        "BAV Patients (n)": bav_count,
                        "Non-BAV Patients (n)": nonbav_count,
                        "BAV Weighted Prevalence (%)": "Suppressed (<10)",
                        "Non-BAV Weighted Prevalence (%)": "Suppressed (<10)",
                        "Difference": "Suppressed (<10)",
                        "95% CI Lower": "Suppressed (<10)",
                        "95% CI Upper": "Suppressed (<10)",
                        "P-Value": "Suppressed (<10)",
                    }
                else:
                    lower_ci, upper_ci = bootstrap_difference(
                        bav_vals, nonbav_vals,
                        n_bootstrap=n_bootstrap,
                        seed=42
                    )
                    _, p_val = ttest_ind(bav_vals, nonbav_vals, equal_var=False)

                    record = {
                        "Group": group,
                        "Subgroup": subgroup_name,
                        "Year": year,
                        "BAV Patients (n)": bav_count,
                        "Non-BAV Patients (n)": nonbav_count,
                        "BAV Weighted Prevalence (%)": round(bav_vals.mean() * 100, 2),
                        "Non-BAV Weighted Prevalence (%)": round(nonbav_vals.mean() * 100, 2),
                        "Difference": round((bav_vals.mean() - nonbav_vals.mean()) * 100, 2),
                        "95% CI Lower": lower_ci,
                        "95% CI Upper": upper_ci,
                        "P-Value": float(p_val),
                    }

                group_records.append(record)
                all_records.append(record)

            # Sort within group safely by numeric diff
            group_df = pd.DataFrame(group_records)
            if not group_df.empty:
                group_df["_sort_diff"] = pd.to_numeric(group_df["Difference"], errors="coerce")
                group_df = (
                    group_df.sort_values("_sort_diff", ascending=False, na_position="last")
                            .drop(columns=["_sort_diff"])
                )
                year_records.extend(group_df.to_dict("records"))

        # Year table: sort by Group + numeric diff
        year_df = pd.DataFrame(year_records)
        if not year_df.empty:
            year_df["_sort_diff"] = pd.to_numeric(year_df["Difference"], errors="coerce")
            year_df = (
                year_df.sort_values(by=["Group", "_sort_diff"], ascending=[True, False], na_position="last")
                      .drop(columns=["_sort_diff"])
                      .reset_index(drop=True)
            )

        year_df.to_csv(os.path.join(save_dir, f"subgroup_sorted_{year}.csv"), index=False)

        year_df_ge10 = year_df[
            (year_df["BAV Patients (n)"] >= 10) & (year_df["Non-BAV Patients (n)"] >= 10)
        ].copy()
        year_df_ge10.to_csv(os.path.join(save_dir, f"subgroup_sorted_{year}_counts_ge10.csv"), index=False)

        print(f"✅ Saved Section 3 subgroup outputs for year {year}")

    overall_df = pd.DataFrame(all_records)

    # Overall rankings (non-suppressed only)
    numeric_overall = overall_df[pd.to_numeric(overall_df["Difference"], errors="coerce").notnull()].copy()
    numeric_overall["Difference"] = pd.to_numeric(numeric_overall["Difference"], errors="coerce")

    overall_grouped = (
        numeric_overall.groupby(["Group", "Subgroup"], as_index=False)["Difference"]
                      .mean()
                      .sort_values(by=["Group", "Difference"], ascending=[True, False], na_position="last")
                      .reset_index(drop=True)
    )

    overall_grouped.to_csv(os.path.join(save_dir, "subgroup_groupwise_overall_sorted.csv"), index=False)

    print("\n=== Final Overall Subgroup Rankings Within Each Group (2000–2019) ===")
    for group in overall_grouped["Group"].unique():
        print(f"\nGroup: {group}")
        sub_df = overall_grouped[overall_grouped["Group"] == group]
        for _, row in sub_df.iterrows():
            print(f" - {row['Subgroup']}: {row['Difference']:.2f}%")

    return overall_df


# --------------------------------------
# Section 3 helper: final top15 long CSV
# --------------------------------------
def write_final_top15_subgroups_csv(
    subgroup_results_df: pd.DataFrame,
    out_path: str
) -> pd.DataFrame:
    """
    Selects top 15 subgroups by mean BAV Patients (n) across years (among those with counts>=10),
    then writes long-format CSV containing ONLY rows where both counts>=10.
    """
    df = subgroup_results_df.copy()
    df_ge10 = df[(df["BAV Patients (n)"] >= 10) & (df["Non-BAV Patients (n)"] >= 10)].copy()

    cols = [
        "Group", "Subgroup", "Year",
        "BAV Patients (n)", "Non-BAV Patients (n)",
        "BAV Weighted Prevalence (%)", "Non-BAV Weighted Prevalence (%)"
    ]

    if df_ge10.empty:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
        print(f"⚠️ No subgroup rows met counts>=10. Wrote empty file: {out_path}")
        return df_ge10

    # Rank by mean BAV patient count across years
    top15_keys = (
        df_ge10.groupby(["Group", "Subgroup"], as_index=False)["BAV Patients (n)"]
              .mean()
              .sort_values("BAV Patients (n)", ascending=False)
              .head(15)[["Group", "Subgroup"]]
    )

    df_top15 = df_ge10.merge(top15_keys, on=["Group", "Subgroup"], how="inner")
    df_top15 = df_top15[cols].sort_values(["Group", "Subgroup", "Year"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_top15.to_csv(out_path, index=False)
    print(f"✅ Saved final top-15 subgroups long CSV: {out_path}")
    return df_top15


# ----------------------------
# Optional plots (Section 2)
# ----------------------------
def plot_combined_error_bars_top10(
    final_df: pd.DataFrame,
    top_comorbidities: List[str],
    save_dir: str = "Section2_Results",
    filename: str = "combined_error_bar_top10.jpeg"
) -> None:
    """
    Error bar plot of yearly difference with 95% CI for top comorbidities.
    Automatically skips suppressed rows.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(14, 8))

    for comorb in top_comorbidities:
        df_c = final_df[final_df["Comorbidity"] == comorb].copy()
        df_c["diff"] = pd.to_numeric(df_c["Weighted Prevalence Difference (%)"], errors="coerce")
        df_c["lo"] = pd.to_numeric(df_c["95% CI Lower"], errors="coerce")
        df_c["hi"] = pd.to_numeric(df_c["95% CI Upper"], errors="coerce")
        df_c = df_c.dropna(subset=["diff", "lo", "hi"]).sort_values("Year")

        if df_c.empty:
            continue

        yerr = [
            df_c["diff"] - df_c["lo"],
            df_c["hi"] - df_c["diff"],
        ]

        plt.errorbar(df_c["Year"], df_c["diff"], yerr=yerr, capsize=4, fmt="-o", label=comorb)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Weighted Prevalence Difference (BAV - Non-BAV) (%)")
    plt.title("Top Comorbidities: Difference with 95% CI (non-suppressed rows only)")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    plt.tight_layout()

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=400)
    plt.close()
    print(f"✅ Saved plot: {out_path}")


def plot_pvalue_trends_top10(
    final_df: pd.DataFrame,
    top_comorbidities: List[str],
    save_dir: str = "Section2_Results",
    filename: str = "pvalue_trends_top10.jpeg"
) -> None:
    """
    P-value trend plot for top comorbidities (log scale). Skips suppressed.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))

    for comorb in top_comorbidities:
        df_c = final_df[final_df["Comorbidity"] == comorb].copy()
        df_c["p"] = pd.to_numeric(df_c["P-Value"], errors="coerce")
        df_c = df_c.dropna(subset=["p"]).sort_values("Year")
        if df_c.empty:
            continue

        plt.plot(df_c["Year"], df_c["p"], "-o", label=comorb)

    plt.axhline(0.05, linestyle="--", linewidth=1, label="p=0.05")
    plt.xlabel("Year")
    plt.ylabel("P-Value")
    plt.title("Yearly P-Value Trends (non-suppressed rows only)")
    plt.yscale("log")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    plt.tight_layout()

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, dpi=400)
    plt.close()
    print(f"✅ Saved plot: {out_path}")


# ----------------------------
# Optional plots (Section 3)
# ----------------------------
def load_all_yearly_subgroups(
    path: str = "Section3_Results",
    years: range = range(2000, 2020)
) -> pd.DataFrame:
    """
    Loads Section 3 yearly subgroup CSVs (subgroup_sorted_{year}.csv) into one DataFrame.
    """
    all_dfs = []
    for y in years:
        f = os.path.join(path, f"subgroup_sorted_{y}.csv")
        if os.path.exists(f):
            df = pd.read_csv(f)
            all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def generate_top15_subgroup_summary(
    df: pd.DataFrame,
    output_dir: str = "Section3_Results"
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Creates top 15 subgroups by mean Difference (%) (non-suppressed only),
    saves CSV, and returns (summary_df, [(Group, Subgroup), ...]).
    """
    os.makedirs(output_dir, exist_ok=True)

    df2 = df.copy()
    df2["Difference_num"] = pd.to_numeric(df2["Difference"], errors="coerce")
    df2 = df2.dropna(subset=["Difference_num"])

    grouped = df2.groupby(["Group", "Subgroup"])["Difference_num"]
    summary = grouped.agg(["mean", "std"]).reset_index()
    summary = summary.rename(columns={"mean": "Mean Difference (%)", "std": "Standard Deviation"})
    summary = summary.sort_values("Mean Difference (%)", ascending=False).head(15).reset_index(drop=True)

    summary.to_csv(os.path.join(output_dir, "top15_subgroup_summary.csv"), index=False)

    top15_list = list(summary[["Group", "Subgroup"]].itertuples(index=False, name=None))
    return summary, top15_list


def plot_top15_subgroups_ci(
    df: pd.DataFrame,
    top15_subgroups: List[Tuple[str, str]],
    output_path: str = "Section3_Results/top15_subgroup_CI_plot.jpeg"
) -> None:
    """
    Plots CI error bars for the top 15 subgroups (non-suppressed only).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(14, 8))

    for group, subgroup in top15_subgroups:
        temp = df[(df["Group"] == group) & (df["Subgroup"] == subgroup)].copy()
        temp["d"] = pd.to_numeric(temp["Difference"], errors="coerce")
        temp["lo"] = pd.to_numeric(temp["95% CI Lower"], errors="coerce")
        temp["hi"] = pd.to_numeric(temp["95% CI Upper"], errors="coerce")
        temp = temp.dropna(subset=["d", "lo", "hi"]).sort_values("Year")

        if temp.empty:
            continue

        yerr = [temp["d"] - temp["lo"], temp["hi"] - temp["d"]]
        plt.errorbar(temp["Year"], temp["d"], yerr=yerr, fmt="-o", capsize=3, label=f"{subgroup} ({group})")

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Weighted Prevalence Difference (BAV - Non-BAV) (%)")
    plt.title("Top 15 Subgroups: Difference with 95% CI (non-suppressed only)")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {output_path}")


def plot_top15_subgroups_pvalues(
    df: pd.DataFrame,
    top15_subgroups: List[Tuple[str, str]],
    output_path: str = "Section3_Results/top15_subgroup_pvalues.jpeg"
) -> None:
    """
    Plots p-value trends for top 15 subgroups (log scale).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(14, 8))

    for group, subgroup in top15_subgroups:
        temp = df[(df["Group"] == group) & (df["Subgroup"] == subgroup)].copy()
        temp["p"] = pd.to_numeric(temp["P-Value"], errors="coerce")
        temp = temp.dropna(subset=["p"]).sort_values("Year")
        if temp.empty:
            continue
        plt.plot(temp["Year"], temp["p"], marker="o", label=f"{subgroup} ({group})")

    plt.axhline(0.05, linestyle="--", label="p=0.05")
    plt.yscale("log")
    plt.xlabel("Year")
    plt.ylabel("P-Value")
    plt.title("P-Value Trends for Top 15 Subgroups (non-suppressed only)")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot: {output_path}")


# ====================================================
# Execution (edit these paths for your environment)
# ====================================================
if __name__ == "__main__":
    # Update to your PKL folder on SAIL VMHorizon
    DATA_DIRECTORY = "../PKL_CAT_Jan_2025/CAT2_BAV_1_5"
    YEARS = range(2000, 2020)  # 2000–2019 inclusive

    # ---------- Section 2 ----------
    final_df, top_comorbidities, yearly_tables = compare_bav_nonbav(
        data_directory=DATA_DIRECTORY,
        years=YEARS,
        out_dir="Section2_Results",
        top_n=10,
        n_bootstrap=1000
    )

    # Save full long table (includes suppressed rows)
    final_df.to_csv("Section2_Results/bav_nonbav_comparison.csv", index=False)

    # Save final top-10 long table (counts>=10 only)
    _ = write_final_top10_csv(
        final_df=final_df,
        top_comorbidities=top_comorbidities,
        out_path="Section2_Results/final_top10_comorbidities_2000_2019.csv"
    )

    # Optional plots for Section 2
    plot_combined_error_bars_top10(final_df, top_comorbidities, save_dir="Section2_Results")
    plot_pvalue_trends_top10(final_df, top_comorbidities, save_dir="Section2_Results")

    # ---------- Section 3 ----------
    # Put Final_JAN.xlsx in the same folder as this script, OR provide full path below.
    EXCEL_MAPPING_PATH = "Final_JAN.xlsx"

    subgroup_mapping = build_subgroup_mapping(
        top_comorbidities=top_comorbidities,
        excel_path=EXCEL_MAPPING_PATH,
        usecols=(1, 4)  # matches your sheet: B (subgroup) and E (group)
    )

    subgroup_results_df = subgroup_prevalence_loop(
        subgroup_map=subgroup_mapping,
        data_directory=DATA_DIRECTORY,
        years=YEARS,
        save_dir="Section3_Results",
        n_bootstrap=1000
    )

    # Save final top-15 subgroups long table (counts>=10 only)
    _ = write_final_top15_subgroups_csv(
        subgroup_results_df=subgroup_results_df,
        out_path="Section3_Results/final_top15_subgroups_2000_2019.csv"
    )

    # Optional: Top-15 subgroup summary + plots (based on saved yearly CSVs)
    all_sub = load_all_yearly_subgroups(path="Section3_Results", years=YEARS)
    if not all_sub.empty:
        summary_df, top15_list = generate_top15_subgroup_summary(all_sub, output_dir="Section3_Results")
        plot_top15_subgroups_ci(all_sub, top15_list, output_path="Section3_Results/top15_subgroup_CI_plot.jpeg")
        plot_top15_subgroups_pvalues(all_sub, top15_list, output_path="Section3_Results/top15_subgroup_pvalues.jpeg")

    print("✅ Section 2 & 3 complete. Outputs saved to Section2_Results/ and Section3_Results/")
