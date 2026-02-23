# Section 2 & Section 3

## Section 2
Cohort-level weighted prevalence comparison between BAV and Non-BAV (2000–2019).

Includes:
- Diagnosis-year filtering
- Weighted prevalence (%)
- Bootstrap 95% CI
- Welch t-test
- Suppression rule (<10 counts)

## Section 3
Subgroup expansion using mapping file (Final_JAN.xlsx).

Includes:
- Sub-disease expansion
- Mean difference ranking
- CI plots and p-value trends

## Data
This project was developed using SAIL Databank data.
No patient-level data are included.

## Expected input
Each year:
<DATA_DIRECTORY>/<year>.pkl

Containing:
- data['original']
- data['matched']
