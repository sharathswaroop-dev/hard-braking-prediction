# Hard-Braking Prediction Pipeline

A machine learning pipeline for real-time driving telemetry to predict hard-braking events (`label_hard_braking_3s`) within the next 3 seconds using strictly causal, trailing historical sensor signals.

---

## 🚗 Overview

- **Objective**: Predict whether a hard-braking event occurs in $[t, t + 3.0\text{s}]$ using only sensor data available up to time $t$.
- **Sampling Frequency**: $10\text{ Hz}$ ($\Delta t = 0.1\text{ s}$).
- **Models**:
  - Baseline: `brake_pressure` threshold
  - Logistic Regression with Standard Scaler
  - Gradient Boosted Trees (`HistGradientBoostingClassifier`)
- **Evaluation**: PR-AUC, ROC-AUC, Recall at Precision $\ge 0.5$, and Leave-One-Trip-Out (LOTO) cross-validation.

---

## 📊 Performance Summary

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test Recall @ P $\ge$ 0.5 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline** (`brake_pressure`) | — | 0.757 | 0.901 | 0.860 |
| **Logistic Regression** | 0.915 | **0.950** | **0.997** | **0.977** |
| **Gradient Boosted Trees (GBT)** | 0.811 | 0.907 | 0.994 | 0.860 |

- **Leave-One-Trip-Out CV (GBT)**: Mean PR-AUC = `0.739 ± 0.146` across 8 held-out trips.

---

## 🛠️ Key Pipeline Components

1. **Data Cleaning & Integrity**:
   - Drops duplicate rows per `(trip_id, t)`.
   - Removes sparse GPS coordinates (`lat`, `lon` at 1Hz).
   - Prevents label leakage: excludes centered rolling averages (e.g. `brake_pressure_ema_5s`) and builds strictly causal trailing EMAs and feature windows.
2. **Feature Engineering**:
   - Rolling statistics (mean, std, min, max) over $1\text{s}$ and $2\text{s}$ trailing windows.
   - Rate-of-change (derivative) metrics for speed and brake pressure.
3. **Trip-Aware Splitting**:
   - Partitioned by `trip_id` (Trips 1–6 for training, Trip 7 for validation, Trip 8 for testing) to prevent cross-trip temporal data leakage.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.9+
- `numpy`, `pandas`, `scikit-learn`

### Running the Pipeline
```bash
python pipeline.py
```

Output artifacts generated:
- `results.json`: Metric evaluation scores and feature importances.
- `cleaned_features.csv`: Cleaned dataset with engineered feature columns.
- `test_predictions.csv`: Model predictions and probabilities on the test set.
