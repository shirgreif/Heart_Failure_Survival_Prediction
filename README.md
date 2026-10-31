# Heart Failure Survival Prediction

Academic project for **Selected Topics in Statistics** (Ben-Gurion University of the Negev).  
We predict post–heart-failure mortality from clinical records and rank risk factors by importance.

**Authors:** Sivan Raviv, Shir Michal Grief, Avital Finanser  
**Supervisor / lecturer:** Prof. Yisrael Parmet  
**Date:** March 2024

📄 Full write-up: [`docs/Final_Project_Report.pdf`](docs/Final_Project_Report.pdf)

---

## Overview

Cardiovascular disease is a leading cause of death worldwide. Using electronic medical records from **299 heart-failure patients** (2015), we train and compare classifiers to:

1. **Predict** the binary outcome `DEATH_EVENT` (death during follow-up).
2. **Identify** the most informative clinical risk factors.

Across models, **follow-up time (`time`)**, **serum creatinine**, and **ejection fraction** consistently emerge as the most important predictors — aligning with prior clinical ML findings on this cohort.

---

## Dataset

| Item | Detail |
|------|--------|
| Patients | 299 (105 women, 194 men), age > 40 |
| Setting | Left-ventricular systolic dysfunction; NYHA class III–IV |
| Target | `DEATH_EVENT` — ~32% death / ~68% survival (imbalanced) |
| Features | Age, anaemia, CPK, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, time |

Local copy: [`data/heart_failure_clinical_records.csv`](data/heart_failure_clinical_records.csv)  
Source & citation: see [`data/README.md`](data/README.md).

---

## Method

### Preprocessing
- No missing values; duplicate check
- Outliers handled with a conservative **3×IQR** rule (to retain minority-class cases)
- Multicollinearity screened with **VIF** (dropped `serum_sodium`, then `age`)
- **Min–Max** scaling where needed
- Train/test split **80/20**

### Evaluation (imbalanced data)
Primary metrics emphasize clinical cost of missing deaths:
- **AUC-ROC**
- **F2 score** (recall-weighted; β = 2)
- **Matthews Correlation Coefficient (MCC)**

Final model choice used the **average** of AUC-ROC, MCC, and F2.

---

## Models & Results

| Model | AUC-ROC | MCC | F2 | Average |
|-------|---------|-----|-----|---------|
| Logistic regression (GLM) | 0.907 | 0.284 | 0.286 | 0.492 |
| **Decision tree (selected)** | **0.859** | **0.681** | **0.759** | **0.766** |
| Random forest | 0.872 | 0.528 | 0.592 | 0.664 |
| Artificial neural network (MLP) | 0.897 | 0.631 | 0.682 | 0.736 |

**Selected model:** Decision tree (entropy, max depth 3, min samples leaf 7, max features 6) after `GridSearchCV` (10-fold CV, AUC-ROC scoring).

**Key predictors:** `time`, `ejection_fraction`, `serum_creatinine`.

---

## Repository structure

```text
├── README.md
├── data/
│   ├── README.md
│   └── heart_failure_clinical_records.csv
└── docs/
    └── Final_Project_Report.pdf
```

---

## Limitations

- Small sample size (n = 299)
- Class imbalance (~32% positive)
- Ambiguity in the clinical meaning of `time` (follow-up / observation length after the event)

---

## License / academic use

This repository is shared for portfolio and educational purposes.  
The dataset remains subject to the UCI / original authors’ terms. Please cite Chicco & Jurman (2020) when using the data.
