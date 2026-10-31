# Heart Failure Survival Prediction

Academic project for **Selected Topics in Statistics**, Ben-Gurion University of the Negev (March 2024).  
Predict post–heart-failure mortality from clinical records and rank risk factors by importance.

**Course lecturer:** Prof. Yisrael Parmet  
**Team:** Sivan Raviv, Avital Finanser (with Shir Greif)

📄 Full write-up: [`docs/Final_Project_Report.pdf`](docs/Final_Project_Report.pdf)

---

## Overview

Using electronic medical records from **299 heart-failure patients** (2015), we:

1. **Predict** the binary outcome `DEATH_EVENT` (death during follow-up).
2. **Identify** the most informative clinical risk factors.

**Follow-up time (`time`)**, **serum creatinine**, and **ejection fraction** emerge as the strongest predictors.

---

## Repository structure

```text
├── README.md
├── data/
│   ├── README.md
│   └── heart_failure_clinical_records_dataset.csv
├── docs/
│   └── Final_Project_Report.pdf
└── src/
    ├── Heart_failure.py   # EDA, preprocessing, DT / RF / ANN (Python)
    └── GLM_Model.R        # Logistic regression / GLM (R)
```

---

## Dataset

| Item | Detail |
|------|--------|
| Patients | 299 (105 women, 194 men), age > 40 |
| Target | `DEATH_EVENT` — ~32% death / ~68% survival (imbalanced) |
| Features | Age, anaemia, CPK, diabetes, ejection fraction, blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, time |

Source & citation: [`data/README.md`](data/README.md).

---

## Method

**Preprocessing:** missing/duplicate checks; conservative **3×IQR** outlier rule; **VIF** multicollinearity screen; Min–Max scaling; **80/20** train–test split.

**Metrics (imbalanced data):** AUC-ROC, **F2** (β = 2), **MCC**. Final choice used their average.

---

## Models & results

| Model | AUC-ROC | MCC | F2 | Average |
|-------|---------|-----|-----|---------|
| Logistic regression (GLM) | 0.907 | 0.284 | 0.286 | 0.492 |
| **Decision tree (selected)** | **0.859** | **0.681** | **0.759** | **0.766** |
| Random forest | 0.872 | 0.528 | 0.592 | 0.664 |
| Artificial neural network (MLP) | 0.897 | 0.631 | 0.682 | 0.736 |

**Selected model:** Decision tree (entropy, max depth 3, min samples leaf 7, max features 6).

---

## How to run

### Python (`src/Heart_failure.py`)
EDA, preprocessing, decision tree, random forest, and neural network.

```bash
cd src
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels mlxtend
python Heart_failure.py
```

### R (`src/GLM_Model.R`)
Binomial logistic regression / GLM analysis. Requires prepared CSVs (`prepared_data.csv`, `df_train.csv`, `df_test.csv`) in the working directory — install packages listed at the top of the script, then:

```r
source("GLM_Model.R")
```

---

## Limitations

- Small sample (n = 299) and class imbalance
- Clinical interpretation of `time` (follow-up length) is ambiguous

---

## License / academic use

Shared for portfolio and educational purposes. Dataset terms: UCI / Chicco & Jurman (2020).
