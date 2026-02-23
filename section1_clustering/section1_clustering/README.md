# Section 1 – Multimorbidity Clustering Pipeline (2000–2019)

This module performs:

• MCA dimensionality reduction (KSS rule)
• Patient clustering (KMeans, GMM, Hierarchical)
• Silhouette-based model selection
• Cluster filtering (<5 patients removed)
• Cluster re-labelling by size
• Cluster-level evaluation metrics
• Demographic breakdown by cluster
• Comorbidity prevalence analysis
• Yearly heatmaps
• 2D & 3D MCA scatter plots
• Cosine similarity grouping across years
• Comorbidity tree generation

## Years Covered
2000 – 2019

## Output Structure

Results are automatically saved under:

# Results/Section_1_Clustering/{Cohort}/{Method}/{Year}/

## Run Example

```python
years = range(2000, 2020)
process_all_years_clustering(data_path, years, is_bav=True, clustering_method="hierarchical")


## 🔒 Data Availability

This project was developed using anonymised, privacy-protected data accessed through the
Secure Anonymised Information Linkage (SAIL) Databank.

Due to governance, ethical, and data protection regulations, the underlying dataset
cannot be publicly shared.

Researchers wishing to access SAIL data must apply directly through the
SAIL Databank governance process:
https://saildatabank.com/

All code required to reproduce the analytical pipeline is provided in this repository.

Or
Data used in this project were accessed via the SAIL Databank under approved governance frameworks.
No identifiable patient data are stored or shared in this repository.
