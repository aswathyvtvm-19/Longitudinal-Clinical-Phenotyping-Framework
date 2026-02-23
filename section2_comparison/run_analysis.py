from section2_bav_vs_nonbav import compare_bav_nonbav, write_final_top10_csv
from section3_subgroups import subgroup_prevalence_loop, write_final_top15_subgroups_csv

DATA_DIRECTORY = "../PKL_CAT_Jan_2025/CAT2_BAV_1_5"
YEARS = range(2000, 2020)

final_df, top_comorbidities, _ = compare_bav_nonbav(
    data_directory=DATA_DIRECTORY,
    years=YEARS
)

write_final_top10_csv(
    final_df=final_df,
    top_comorbidities=top_comorbidities,
    out_path="../Section2_Results/final_top10.csv"
)
