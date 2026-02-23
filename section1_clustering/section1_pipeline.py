# ================================
# Imports
# ================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import MCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from IPython.display import display, HTML


# ================================
# Exporting Tabular Outputs and Stylized Tables
# ================================
def save_dataframe_as_jpeg(df, path, title=""):
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.4 * (len(df) + 1))))
    ax.axis('off')
    formatted_df = df.copy()
    formatted_df.columns = [str(col).replace('_', ' ') for col in formatted_df.columns]
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].astype(str).str.replace('_', ' ', regex=False)

    table = ax.table(cellText=formatted_df.values, 
                     colLabels=formatted_df.columns, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
            cell.set_linewidth(0.5)

    plt.title(title, fontsize=14, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def save_all_outputs(df, year, filename, title, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    csv_path = os.path.join(folder_path, f"{filename}.csv")
    jpeg_path = os.path.join(folder_path, f"{filename}.jpeg")
    df.to_csv(csv_path, index=False)
    display(df)  # 📄 Print in notebook output as requested
    save_dataframe_as_jpeg(df, jpeg_path, title)

def interpret_silhouette(score):
    if score >= 0.7:
        return "Excellent"
    elif score >= 0.5:
        return "Good"
    elif score >= 0.3:
        return "Moderate"
    else:
        return "Poor"



# ================================
# KSS Rule for MCA Component Selection
# ================================
def kss_rule(eigenvalues, n_individuals, n_variables):
    # 🎯 Determines optimal MCA components using the KSS rule
    trace_total = sum(eigenvalues)
    threshold = trace_total / min(n_individuals - 1, n_variables)
    return sum(eig > threshold for eig in eigenvalues)

# ================================
# Perform MCA using optimal components
# ================================
def perform_mca_kss(df):
    n_individuals, n_variables = df.shape
    mca_temp = MCA(n_components=min(n_variables, 20), random_state=42)
    mca_result_temp = mca_temp.fit(df.astype(str))
    eigenvalues = mca_result_temp.eigenvalues_
    optimal_components = kss_rule(eigenvalues, n_individuals, n_variables)
    mca = MCA(n_components=optimal_components, random_state=42)
    mca_result = mca.fit(df.astype(str))
    return mca, mca_result

# ================================


# ================================
# Cluster Evaluation Metrics
# ================================
def evaluate_clusters(row_coords, labels, year, save_dir, cohort):
    silhouette = silhouette_score(row_coords, labels)
    ch = calinski_harabasz_score(row_coords, labels)
    db = davies_bouldin_score(row_coords, labels)

    centroids = np.vstack([row_coords[labels == i].mean(axis=0) for i in np.unique(labels)])
    intra = sum([np.sum((row_coords[labels == i] - c) ** 2) for i, c in enumerate(centroids)])
    inter = np.min([
        np.sum((c1 - c2) ** 2) for i, c1 in enumerate(centroids)
        for j, c2 in enumerate(centroids) if i != j
    ])
    xb = intra / (row_coords.shape[0] * inter)

    def get_remark(metric, value):
        # 🧠 Qualitative interpretations for metrics
        if metric == "Silhouette Score":
            return ("Good", "Clear structure but potential overlap") if value >= 0.5 else \
                   ("Moderate", "Potential structure, some overlap") if value >= 0.25 else \
                   ("Poor", "No substantial structure")
        elif metric == "Calinski-Harabasz":
            return ("Excellent", "Distinct clustering structure") if value >= 500 else \
                   ("Good", "Well-defined clusters") if value >= 200 else \
                   ("Poor", "Clusters may not be distinct")
        elif metric == "Davies-Bouldin":
            return ("Good", "Reasonably good clustering") if value <= 1 else \
                   ("Moderate", "Some cluster overlap") if value <= 2 else \
                   ("Poor", "Likely poor clustering")
        elif metric == "Xie-Beni Index":
            return ("Good", "Tight and well-separated clusters") if value <= 0.5 else \
                   ("Moderate", "Some structure") if value <= 0.8 else \
                   ("Poor", "Loose or overlapping clusters")
        return "", ""

    metrics = [
        ("Silhouette Score", silhouette),
        ("Calinski-Harabasz", ch),
        ("Davies-Bouldin", db),
        ("Xie-Beni Index", xb)
    ]
    rows = [{"Metric": m, "Value": round(v, 4), "Quality": get_remark(m, v)[0], "Remarks": get_remark(m, v)[1]} 
            for m, v in metrics]

    df_metrics = pd.DataFrame(rows)
    eval_dir = os.path.join("Results", "Section_1_Clustering", cohort, str(year))
    os.makedirs(eval_dir, exist_ok=True)
    save_all_outputs(df_metrics, year, "cluster_evaluation", f"Cluster Evaluation - ({cohort} - {year})", eval_dir)


# ================================
# Demographic Analysis by Cluster
# ================================
# for correcting GP-Coverage and cohort end date



# ================================
# Cluster Demographics with GP and Cohort End
# ================================
def cluster_demographic_analysis(df, year, folder_path, cohort):
    bins = [0, 20, 40, 60, 80]
    labels = ['0-20', '21-40', '41-60', '61-80']
    df['AGE_GROUP'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

    df['DOD'] = pd.to_datetime(df['DOD'], errors='coerce')
    df['GP_COVERAGE_END_DATE'] = pd.to_datetime(df['GP_COVERAGE_END_DATE'], errors='coerce')
    df['COHORT_END_DATE'] = pd.to_datetime(df['COHORT_END_DATE'], errors='coerce')

    def format_count(val):
        return '-' if val < 5 else val

    basic_rows, age_rows = [], []

    for cluster in sorted(df['Cluster'].unique()):
        cdf = df[df['Cluster'] == cluster]

        row = {
            'Cluster': cluster,
            'Total Patients': len(cdf),
            'Male': (cdf['GNDR_NAME'] == 'Male').sum(),
            'Female': (cdf['GNDR_NAME'] == 'Female').sum(),
            'Deaths': format_count(cdf['DOD'].notna().sum()),
            'GP Coverage End': format_count((cdf['GP_COVERAGE_END_DATE'] <= pd.Timestamp(year, 12, 31)).sum()),
            'Cohort End': format_count((cdf['COHORT_END_DATE'] <= pd.Timestamp(year, 12, 31)).sum())
        }
        basic_rows.append(row)

        age_counts = cdf['AGE_GROUP'].value_counts().sort_index()
        age_gender = cdf.groupby(['AGE_GROUP', 'GNDR_NAME']).size().unstack(fill_value=0)

        for group in labels:
            agdf = cdf[cdf['AGE_GROUP'] == group]
            age_row = {
                'Cluster': cluster,
                'Age Group': group,
                'Total': age_counts.get(group, 0),
                'Male': age_gender.loc[group, 'Male'] if 'Male' in age_gender.columns and group in age_gender.index else 0,
                'Female': age_gender.loc[group, 'Female'] if 'Female' in age_gender.columns and group in age_gender.index else 0,
                'Deaths': format_count(agdf['DOD'].notna().sum()),
                'GP Coverage End': format_count((agdf['GP_COVERAGE_END_DATE'] <= pd.Timestamp(year, 12, 31)).sum()),
                'Cohort End': format_count((agdf['COHORT_END_DATE'] <= pd.Timestamp(year, 12, 31)).sum())
            }
            age_rows.append(age_row)

    demo_df = pd.DataFrame(basic_rows)
    age_df = pd.DataFrame(age_rows)

    save_all_outputs(demo_df, year, "cluster_demographics_basic", f"Cluster Demographics – ({cohort} - {year})", folder_path)
    save_all_outputs(age_df, year, "cluster_demographics_agegroup", f"Age Group Breakdown – ({cohort} - {year})", folder_path)



def analyse_clusters_by_comorbidity(df, comorbidity_cols, year, folder_path, cohort):
    """
    Analyzes comorbidity prevalence per cluster using correct date filtering.
    Saves:
    - Full prevalence tables
    - Top 10 JPEGs
    - Yearly heatmap across clusters
    """  

    heat_data = {}

    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        
        cluster_prevalence = {}
        cluster_counts = {}

        for comorb in comorbidity_cols:
            date_col = f"{comorb}_DATE"
            if date_col in cluster_data.columns:
                cluster_data[date_col] = pd.to_datetime(cluster_data[date_col], errors='coerce')
                diagnosed_mask = cluster_data[date_col].dt.year <= year
                count = diagnosed_mask.sum()
                if count == 0:
                    continue
                percentage = round(count / len(cluster_data) * 100, 2)

                cluster_counts[comorb] = count
                cluster_prevalence[comorb] = percentage

        # ➕ Filter and sort by prevalence
        prevalence_series = pd.Series(cluster_prevalence)
        prevalence_series = prevalence_series[prevalence_series > 0].sort_values(ascending=False)

        comorb_names = prevalence_series.index.str.replace('TOP_CAT_FIRST_', '', regex=False)\
                                               .str.replace('TOP_CAT_', '', regex=False)\
                                               .str.replace('_', ' ', regex=False)

        # ✅ Align patient counts with sorted prevalence
        counts_aligned = [cluster_counts[comorb] for comorb in prevalence_series.index]

        result_df = pd.DataFrame({
            "Comorbidity": comorb_names,
            "Patient Count": counts_aligned,
            "Prevalence (%)": prevalence_series.values
        })

        result_df.to_csv(os.path.join(folder_path, f"full_comorbidities_cluster_{cluster}.csv"), index=False)

        top10_df = result_df.head(10).copy()
        save_dataframe_as_jpeg(
            top10_df,
            path=os.path.join(folder_path, f"top10_comorbidities_cluster_{cluster}.jpeg"),
            title=f"{cohort} - Top 10 Comorbidities – Cluster {cluster} – {year}"
        )

        # 🔥 For heatmap
        heat_data[f'Cluster {cluster} (n={len(cluster_data)})'] = pd.Series(prevalence_series.values, index=comorb_names)

    plot_yearly_heatmap(heat_data, year, folder_path, cohort, top_n=20)


import pandas as pd
import os

def select_best_algorithm(results_dir="Results/Section_1_Clustering/BAV"):
    """
    Reads evaluation files from each clustering method and year,
    normalizes the metric scores, ranks them, and identifies the best overall method.
    """
    eval_frames = []

    # 1. Loop through all clustering methods and years
    for method in ["kmeans", "gmm", "hierarchical", "fuzzy", "hdbscan"]:
        method_dir = os.path.join(results_dir, method)
        for year in range(2000, 2020):
            eval_path = os.path.join(method_dir, str(year), "cluster_evaluation.csv")
            if os.path.exists(eval_path):
                df = pd.read_csv(eval_path)

                # 2. Reshape from wide to long format for easier metric comparison
                melted = df.melt(
                    id_vars=["n_clusters"],
                    value_vars=["silhouette", "calinski", "davies_bouldin"],
                    var_name="Metric",
                    value_name="Value"
                )
                melted["Method"] = method
                melted["Year"] = year
                eval_frames.append(melted)

    if not eval_frames:
        print("⚠️ No evaluation files found.")
        return None, None

    # 3. Combine all evaluation records
    all_evals = pd.concat(eval_frames, ignore_index=True)

    # 4. Normalize scores within each year-metric group
    all_evals["Normalized_Score"] = all_evals.groupby(["Year", "Metric"])["Value"].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )

    # 5. Invert metrics where lower values are better (like DB)
    invert_metrics = ["davies_bouldin"]
    all_evals.loc[all_evals["Metric"].isin(invert_metrics), "Normalized_Score"] = (
        1 - all_evals.loc[all_evals["Metric"].isin(invert_metrics), "Normalized_Score"]
    )

    # 6. Rank each method within year-metric
    all_evals["Rank"] = all_evals.groupby(["Year", "Metric"])["Normalized_Score"].rank(ascending=False)

    # 7. Average rank per method across all metrics and years
    avg_rank = all_evals.groupby("Method")["Rank"].mean().sort_values()

    best_method = avg_rank.idxmin()
    print(f"🏆 Best algorithm: {best_method} (Avg Rank: {avg_rank.min():.2f})")

    return best_method, avg_rank


# ================================
# Cosine Similarity Across Clusters using Connected Components

# ================================
# Imports
# ================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prince import MCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from IPython.display import display, HTML

# ================================
# Exporting Tabular Outputs and Stylized Tables
# ================================
def save_dataframe_as_jpeg(df, path, title=""):
    fig, ax = plt.subplots(figsize=(12, max(2.5, 0.4 * (len(df) + 1))))
    ax.axis('off')
    formatted_df = df.copy()
    formatted_df.columns = [str(col).replace('_', ' ') for col in formatted_df.columns]
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].astype(str).str.replace('_', ' ', regex=False)

    table = ax.table(cellText=formatted_df.values, 
                     colLabels=formatted_df.columns, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
            cell.set_linewidth(0.5)

    plt.title(title, fontsize=14, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()



def save_all_outputs(df, year, filename, title, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    csv_path = os.path.join(folder_path, f"{filename}.csv")
    jpeg_path = os.path.join(folder_path, f"{filename}.jpeg")
    df.to_csv(csv_path, index=False)
    display(df)  # 📄 Print in notebook output as requested
    save_dataframe_as_jpeg(df, jpeg_path, title)


def summarize_top_comorbidities_by_group(grouped_df, output_dir):
    summary = []
    all_groups = grouped_df.copy()

    for group_id, group_data in all_groups.groupby("Group_ID"):
        comorb_count = defaultdict(list)

        for _, row in group_data.iterrows():
            for entry in row["Top_Comorbidities"].split("; "):
                if "(" in entry and entry.endswith("%)"):
                    name, pct = entry.rsplit(" (", 1)
                    pct = float(pct.strip("%)"))
                    comorb_count[name].append(pct)

        averaged = [(k, round(np.mean(v), 2)) for k, v in comorb_count.items()]
        averaged = sorted(averaged, key=lambda x: -x[1])[:10]

        for comorb, avg_pct in averaged:
            summary.append({
                "Group_ID": group_id,
                "Comorbidity": comorb,
                "Average Prevalence (%)": avg_pct
            })

    result_df = pd.DataFrame(summary)
    result_df = result_df.sort_values(["Group_ID", "Average Prevalence (%)"], ascending=[True, False])
    save_all_outputs(
        result_df,
        year='cosine_groups',
        filename='groupwise_top_comorbidities',
        title='Top 10 Comorbidities by Cosine Similarity Group',
        folder_path=output_dir
    )

# ================================
# Cosine Similarity Grouping Logic
# ================================
#✅ find_multilevel_similarity_groups() — With Weighted Prevalence 
def find_multilevel_similarity_groups(folder_path, thresholds=[0.9, 0.85, 0.8]):
    all_clusters = []
    years = range(2000, 2020)

    for year in years:
        year_dir = os.path.join(folder_path, str(year))
        if not os.path.exists(year_dir):
            continue

        # ✅ Load accurate patient counts from cluster_demographics_basic.csv
        demo_path = os.path.join(year_dir, "cluster_demographics_basic.csv")
        if not os.path.exists(demo_path):
            print(f"⚠️ Missing demographics for year {year}")
            continue

        # 🔢 Create a lookup dictionary: {Cluster ID → Total Patients}
        demo_df = pd.read_csv(demo_path)
        demo_dict = {
            int(row["Cluster"]): int(row["Total Patients"])
            for _, row in demo_df.iterrows()
        }

        for file in os.listdir(year_dir):
            if file.startswith("full_comorbidities_cluster_") and file.endswith(".csv"):
                cluster_df = pd.read_csv(os.path.join(year_dir, file))
                cluster_id = int(file.split("_")[-1].replace(".csv", ""))

                # 📊 Load comorbidity prevalence values
                comorbs = cluster_df.set_index("Comorbidity")["Prevalence (%)"]

                # ✅ Use patient count from demographics file
                patient_count = demo_dict.get(cluster_id, 0)

                all_clusters.append({
                    'year': year,
                    'cluster': cluster_id,
                    'vector': comorbs,            # 📈 Prevalence values per comorbidity
                    'size': patient_count,        # 👥 Accurate number of patients in cluster
                    'label': f"{year}_C{cluster_id}"
                })

    # 🧭 Create a master list of all comorbidities across all clusters
    all_comorbs = sorted(set().union(*[set(c['vector'].index) for c in all_clusters]))

    # 📐 Align all cluster vectors to the same comorbidity index (fill missing with 0)
    for c in all_clusters:
        c['aligned'] = np.array([c['vector'].get(com, 0) for com in all_comorbs])

    used_labels = set()
    summary_rows = []

    for t_idx, threshold in enumerate(thresholds):
        # 📌 Work only on clusters that haven't yet been grouped
        labels = [c['label'] for c in all_clusters if c['label'] not in used_labels]
        vectors = [c['aligned'] for c in all_clusters if c['label'] not in used_labels]

        if len(vectors) < 2:
            print(f"[Skipped] Threshold {threshold}: Not enough ungrouped vectors ({len(vectors)} remaining)")
            continue

        # 🔗 Compute pairwise cosine similarity matrix
        cosine_mat = cosine_similarity(vectors)

        # 🌐 Build adjacency matrix based on cross-year similarity threshold
        adjacency = np.zeros_like(cosine_mat, dtype=int)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j:
                    continue
                if labels[i].split("_")[0] != labels[j].split("_")[0] and cosine_mat[i, j] >= threshold:
                    adjacency[i, j] = 1

        # 🔎 Use connected components to find groups of similar clusters
        n_components, group_labels = connected_components(adjacency, directed=False)
        groups = defaultdict(list)
        for cluster_label, group_id in zip(labels, group_labels):
            groups[group_id].append(cluster_label)

        # 🔁 For each group, calculate weighted average prevalence
        cluster_map = {c['label']: c for c in all_clusters}
        for gid, group_list in groups.items():
            if len(group_list) < 2:
                continue

            # 🧮 Stack all vectors and get their weights (patient counts)
            vectors = [cluster_map[lab]['aligned'] for lab in group_list]
            weights = np.array([cluster_map[lab]['size'] for lab in group_list])
            stacked = np.vstack(vectors)

            # ✅ Weighted prevalence across clusters in the group
            avg_vec = np.average(stacked, axis=0, weights=weights)

            # 🔝 Top 10 comorbidities based on weighted average
            top10_idx = np.argsort(avg_vec)[::-1][:10]
            top_comorbs = [(all_comorbs[i], round(avg_vec[i], 2)) for i in top10_idx]

            # 📏 Max pairwise cosine similarity within the group
            max_sim = np.max([
                cosine_similarity([cluster_map[a]['aligned']], [cluster_map[b]['aligned']])[0, 0]
                for i, a in enumerate(group_list)
                for j, b in enumerate(group_list) if i < j
            ])

            # 📝 Add each cluster’s row to summary
            for cluster in sorted(group_list):
                used_labels.add(cluster)
                summary_rows.append({
                    "Similarity_Level": f"L{t_idx + 1} (≥ {threshold})",
                    "Group_ID": f"G{t_idx}_{gid}",
                    "Cluster": cluster,
                    "Year": cluster_map[cluster]['year'],
                    "Patients": cluster_map[cluster]['size'],
                    "Top_Comorbidities": "; ".join([f"{c} ({v}%)" for c, v in top_comorbs]),
                    "Max_Cosine": round(max_sim, 3)
                })

    # 🧾 Add unmatched clusters
    for c in all_clusters:
        if c['label'] not in used_labels:
            avg_vec = c['aligned']
            top10_idx = np.argsort(avg_vec)[::-1][:10]
            top_comorbs = [(all_comorbs[i], round(avg_vec[i], 2)) for i in top10_idx]
            summary_rows.append({
                "Similarity_Level": "Unmatched (< 0.8)",
                "Group_ID": "G_unmatched",
                "Cluster": c['label'],
                "Year": c['year'],
                "Patients": c['size'],
                "Top_Comorbidities": "; ".join([f"{x} ({y}%)" for x, y in top_comorbs]),
                "Max_Cosine": "-"
            })

    # 🗂️ Final output
    summary_df = pd.DataFrame(summary_rows)
    output_dir = os.path.join(folder_path, "cosine_groups")
    os.makedirs(output_dir, exist_ok=True)

    # 💾 Save results
    save_all_outputs(summary_df, 'cosine_groups', 'multi_level_similarity_groups_summary', 'Cosine Similarity Groups by Level', output_dir)
    summarize_top_comorbidities_by_group(summary_df, output_dir)

    return summary_df



# ================================
# 🎨 Heatmap Plot - Yearly based on clustering
# ================================

def plot_yearly_heatmap(heat_data, year, output_dir, cohort, top_n=20):
    """
    Plots a heatmap of top N comorbidities across clusters for a given year.

    Note:
    The prevalence values in `heat_data` are already filtered to include only
    diagnoses that occurred on or before the current year. This filtering is
    applied upstream in `analyse_clusters_by_comorbidity(...)`.
    """
    # 📊 Convert dictionary to DataFrame: rows = comorbidities, columns = clusters
    heat_df = pd.DataFrame(heat_data).fillna(0)

    # 📈 Compute average prevalence across clusters
    mean_prevalence = heat_df.mean(axis=1)

    # 🔍 Select top N comorbidities based on average prevalence
    top_comorbidities = mean_prevalence.sort_values(ascending=False).head(top_n).index
    heat_df_top = heat_df.loc[top_comorbidities]

    # 🎨 Plot heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(heat_df_top, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5, linecolor='gray')
    plt.title(f"Yearly Heatmap of Top {top_n} Comorbidities ({cohort} - {year})", fontsize=16)
    plt.ylabel("Comorbidity", fontsize=14)
    plt.xlabel("Cluster \n(n = Number of patients)", fontsize=14)
    plt.tight_layout()

    # 💾 Save heatmap to file
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"yearly_cluster_heatmap_{year}.jpeg"), dpi=600, bbox_inches='tight')
    plt.close()

# ================================
#  🌀 MCA Scatter Plot Visualization
# ================================
def plot_cluster_scatter(row_coords, labels, year, folder_path, cohort):
    """
    Creates and saves a scatterplot of MCA components with cluster labels.
    """
    df_plot = row_coords.copy()
    df_plot['Cluster'] = labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df_plot.iloc[:, 0], y=df_plot.iloc[:, 1],
        hue=df_plot['Cluster'].astype(str),
        palette='tab10', alpha=0.7
    )
    plt.title(f'\nMCA Cluster Scatter – ({cohort} - {year})\n\n')
    plt.xlabel('MCA Dimension 1')
    plt.ylabel('MCA Dimension 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, "cluster_scatter.jpeg"), dpi=300)
    plt.close()

from mpl_toolkits.mplot3d import Axes3D  # 📦 Required for 3D plotting

def plot_cluster_scatter_3d(row_coords, labels, year, folder_path, cohort):
    """
    Creates and saves a 3D scatterplot of MCA components with cluster labels.
    """
    df_plot = row_coords.copy()
    df_plot['Cluster'] = labels

    # 🎯 Extract first three MCA dimensions
    x = df_plot.iloc[:, 0]
    y = df_plot.iloc[:, 1]
    z = df_plot.iloc[:, 2] if df_plot.shape[1] > 2 else np.zeros_like(x)  # Fallback if only 2D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 🖌️ Plot each cluster in 3D space
    for cluster in sorted(df_plot['Cluster'].unique()):
        mask = df_plot['Cluster'] == cluster
        ax.scatter(x[mask], y[mask], z[mask], label=f'Cluster {cluster}', alpha=0.7)

    ax.set_title(f'3D MCA Cluster Scatter – ({cohort} - {year})', fontsize=14)
    ax.set_xlabel('MCA Dimension 1')
    ax.set_ylabel('MCA Dimension 2')
    ax.set_zlabel('MCA Dimension 3')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, "cluster_scatter_3D.jpeg"), dpi=300, bbox_inches='tight')
    plt.close()

# ================================
# Clustering with Support for KMeans, GMM, and Hierarchical
# ================================
def cluster_patients(mca_result, df, year, cohort, min_size=5, method="kmeans"):
    row_coords = mca_result.row_coordinates(df)
    row_coords.index = df.index

  
    #initialize variables to store the best cluster configuration.
    best_score = -1
    best_n = 2
    best_labels = None

    # Loop Over 2 to 10 Clusters:
    #Test each possible number of clusters (2 to 10).
    for n in range(2, 11):
        #Apply Selected Clustering Method:  (One of the three methods is used for each n.)
        if method == "kmeans":
            model = KMeans(n_clusters=n, random_state=0)
            labels = model.fit_predict(row_coords)
        elif method == "gmm":
            model = GaussianMixture(n_components=n, random_state=0)
            labels = model.fit(row_coords).predict(row_coords)
        elif method == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n)
            labels = model.fit_predict(row_coords)
          # One of the three methods is used for each n.
        else:
            raise ValueError("Unsupported clustering method")

        if len(np.unique(labels)) > 1:
            # Evaluate Using Silhouette Score:
            score = silhouette_score(row_coords, labels)
          # This score measures how well-separated and cohesive the clusters are:
                # High (close to 1): very good clustering              
                #Near 0: overlapping clusters                
                #Negative: incorrect clustering


          #Save the Best Result:
            if score > best_score:
                best_score = score
                best_n = n
                best_labels = labels
              # The loop keeps track of the number of clusters that gave the best silhouette score.
              

    final_labels = pd.Series(best_labels, index=row_coords.index)
    cluster_sizes = final_labels.value_counts()

    # Filter Tiny Clusters:
    #Identify clusters below min_size (<5 patients)
    small_clusters = cluster_sizes[cluster_sizes < min_size]
    filtered_idx = final_labels[~final_labels.isin(small_clusters.index)].index

  
  # Clusters with <5 patients are removed (above), and the remaining ones are re-labeled by size so Cluster 0 is always the biggest.
    if len(small_clusters) > 0:
        removed_counts = final_labels[final_labels.isin(small_clusters.index)].value_counts().to_dict()
        print(f"📉 Removed {len(final_labels) - len(filtered_idx)} patients in year {year} from small clusters: {removed_counts}")
    else:
        print(f"✅ No patients removed due to small clusters in year {year}.")

    print(f"📌 Best silhouette score for {method} in {year}: {best_score:.4f} using {best_n} clusters")

    row_coords_filtered = row_coords.loc[filtered_idx].copy()
    filtered_labels = final_labels.loc[filtered_idx]

    # Re-label based on cluster size (descending)
    relabel_order = filtered_labels.value_counts().sort_values(ascending=False).index
    relabel_map = {old: new for new, old in enumerate(relabel_order)}
    filtered_labels = filtered_labels.map(relabel_map)

    return row_coords_filtered, best_n, filtered_labels.values, filtered_idx, best_score

# ================================
# 🔁 Updated Section 1 Execution Loop with clustering_method
# ================================
def process_all_years_clustering(data_directory, years, is_bav=True, clustering_method="kmeans"):
    cohort = "BAV" if is_bav else "Non-BAV"
    base_path = os.path.join("Results", "Section_1_Clustering", cohort, clustering_method)

    demographics_summary = []
    scores_summary = []

    for year in years:
        print(f"🔍 Processing {cohort} – {year} using {clustering_method}")
        file_path = os.path.join(data_directory, f"{year}.pkl")
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            continue

        data = pd.read_pickle(file_path)
        df = data['original'] if is_bav else data['matched']
        df['AGE'] = year - df['YOB']

        gender_col = 'GNDR_NAME' if 'GNDR_NAME' in df.columns else 'GENDER'
        gender_counts = df[gender_col].value_counts()
        male_count = gender_counts.get('Male', 0)
        female_count = gender_counts.get('Female', 0)

        print(f"📊 Year {year} patient counts: {len(df)} | Male: {male_count}, Female: {female_count}")

        demographics_summary.append({
            "Year": year,
            "Total Patients": len(df),
            "BAV Patients": len(data['original']) if 'original' in data else 0,
            "Non-BAV Patients": len(data['matched']) if 'matched' in data else 0,
            "BAV Male": data['original']['GNDR_NAME'].value_counts().get('Male', 0) if 'original' in data else 0,
            "BAV Female": data['original']['GNDR_NAME'].value_counts().get('Female', 0) if 'original' in data else 0,
            "Non-BAV Male": data['matched']['GNDR_NAME'].value_counts().get('Male', 0) if 'matched' in data else 0,
            "Non-BAV Female": data['matched']['GNDR_NAME'].value_counts().get('Female', 0) if 'matched' in data else 0
        })

        year_path = os.path.join(base_path, str(year))
        os.makedirs(year_path, exist_ok=True)

        comorb_cols = [c for c in df.columns if c.startswith('TOP_CAT_FIRST_') and not c.endswith('_DATE')]

        mca, mca_result = perform_mca_kss(df[comorb_cols])
        row_coords, n_clusters, labels, filtered_idx, best_score = cluster_patients(
            mca_result, df[comorb_cols], year, cohort, method=clustering_method)

        scores_summary.append({
            "Year": year,
            "Cohort": cohort,
            "Method": clustering_method,
            "Best #Clusters": n_clusters,
            "Silhouette Score": round(best_score, 4),
            "Rating": interpret_silhouette(best_score)
        })

        df_filtered = df.loc[filtered_idx].copy()
        df_filtered['Cluster'] = labels

        if len(np.unique(labels)) > 1:
            evaluate_clusters(row_coords.values, labels, year, year_path, cohort)
        else:
            print(f"⚠️ Skipping evaluation for {cohort} {year} – only one cluster found")
            pd.DataFrame({"Year": [year], "Cohort": [cohort], "Message": ["Only one cluster found – skipping evaluation"]}) \
              .to_csv(os.path.join(year_path, "skipped_evaluation_note.csv"), index=False)

        cluster_demographic_analysis(df_filtered, year, year_path, cohort)
        analyse_clusters_by_comorbidity(df_filtered, comorb_cols, year, year_path, cohort)
        plot_cluster_scatter(row_coords, labels, year, year_path, cohort)
        plot_cluster_scatter_3d(row_coords, labels, year, year_path, cohort)

    demo_df = pd.DataFrame(demographics_summary)
    print("\n📊 Annual Demographics Summary:")
    display(HTML(demo_df.to_html(index=False)))

    score_df = pd.DataFrame(scores_summary)
    score_path = os.path.join(base_path, "clustering_score_summary.csv")
    score_df.to_csv(score_path, index=False)

    print("\n📈 Clustering Quality Summary:")
    display(HTML(score_df.to_html(index=False)))


# ================================
# 😁 Final Calls to Execute Section 1 (Choose clustering method)
# ================================
years = range(2000, 2020)
data_path = "../PKL_CAT_Jan_2025/Cat2_BAV_1_5"

# Run for BAV with KMeans
process_all_years_clustering(data_path, years, is_bav=True, clustering_method="kmeans")
a
# Run for BAV with GMM
# process_all_years_clustering(data_path, years, is_bav=True, clustering_method="gmm")

# Run for BAV with Hierarchical
# process_all_years_clustering(data_path, years, is_bav=True, clustering_method="hierarchical")

# Identify cluster similarity using cosine similarity (optional call)


import matplotlib.pyplot as plt

def save_tree_as_image(tree_text, output_path, dpi=300, font_size=10):
    """Save the tree structure as a high-resolution image."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')  # Hide axes
    
    # Use monospace font for proper alignment
    plt.text(
        0, 1, tree_text,
        fontfamily='monospace',
        fontsize=font_size,
        verticalalignment='top',
        linespacing=1.5
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"✅ Tree saved as high-quality image: {output_path}")


def generate_comorbidity_tree(output_dir):
    # Read the final comorbidity summary table
    summary_path = os.path.join(output_dir, "cosine_groups", "groupwise_top_comorbidities.csv")
    df = pd.read_csv(summary_path)
    
    # Create consistent sequential group numbering
    unique_groups = df[df['Group_ID'] != 'G_unmatched']['Group_ID'].unique()
    group_mapping = {old: f"G{i+1}" for i, old in enumerate(sorted(unique_groups))}
    group_mapping['G_unmatched'] = 'G_unmatched'  # Keep unmatched as is
    
    # Apply the new group names
    df['New_Group_ID'] = df['Group_ID'].map(group_mapping)
    
    # Generate the tree structure
    tree_lines = ["BAV Patients"]
    
    # Process groups in order (G1, G2,...) then unmatched
    ordered_groups = sorted(df[df['New_Group_ID'] != 'G_unmatched']['New_Group_ID'].unique()) + ['G_unmatched']
    
    for group_id in ordered_groups:
        group_data = df[df['New_Group_ID'] == group_id]
        tree_lines.append(f"   ├── {group_id}")
        
        # Add top 10 comorbidities
        comorb_lines = []
        for i, row in group_data.head(10).iterrows():
            comorb = row["Comorbidity"]
            prevalence = row["Average Prevalence (%)"]
            prefix = "   │     ├──" if i < len(group_data)-1 else "   │     └──"
            comorb_lines.append(f"{prefix} {comorb} ({prevalence}%)")
        
        tree_lines.extend(comorb_lines)
    
    # Convert to single string
    tree_output = "\n".join(tree_lines)
    
    # Save to file with UTF-8 encoding
    tree_path = os.path.join(output_dir, "cosine_groups", "comorbidity_tree.txt")
    with open(tree_path, "w", encoding="utf-8") as f:
        f.write(tree_output)
    
    print(tree_output)

    # Save as TXT
    tree_path_txt = os.path.join(output_dir, "cosine_groups", "comorbidity_tree.txt")
    with open(tree_path_txt, "w", encoding="utf-8") as f:
        f.write(tree_output)
    
    # Save as PNG (high quality)
    tree_path_png = os.path.join(output_dir, "cosine_groups", "comorbidity_tree.png")
    save_tree_as_image(tree_output, tree_path_png, dpi=300)
    
    print(tree_output)
    return tree_output

# Add this to your existing code after find_multilevel_similarity_groups()
summary_df = find_multilevel_similarity_groups("Results/Section_1_Clustering/BAV/hierarchical")
comorbidity_tree = generate_comorbidity_tree("Results/Section_1_Clustering/BAV/hierarchical")"


