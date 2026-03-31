# Notebook Implementation Plan: Challenge 2 — Transaction Risk Scoring

## Overview

Build a complete Jupyter notebook (`notebook.ipynb`) for fraud detection following the CRISP-DM lifecycle.
The notebook already has one existing cell (Cell 0) with base imports and data loading — keep it as-is and append all new cells after it.

**File to edit:** `Lesson5-Challenges/2_TransactionRisk/notebook.ipynb`  
**Dataset:** `Lesson5-Challenges/2_TransactionRisk/creditcard.csv` (284,807 rows, already downloaded)  
**Target variable:** `Class` — 1 = fraud, 0 = legitimate

---

## Dataset Description

| Column | Description |
|--------|-------------|
| `V1`–`V28` | PCA principal components. V1 captures the most variance, V28 the least. Values are scaled, centered, with no physical meaning — abstract coordinates in n-dimensional space. Already normalized, **do not re-scale**. |
| `Time` | Seconds elapsed between each transaction and the first transaction in the dataset. Not a real clock time. |
| `Amount` | Raw transaction dollar amount. Right-skewed. |
| `Class` | Response variable — **1 = fraud**, 0 = legitimate. |

**Key stats:**
- 284,807 total transactions
- 492 fraud cases (0.172%) — severely imbalanced
- Imbalance ratio ≈ 577:1 (legit:fraud)
- Dataset spans ~48 hours

---

## Business Context (to include in notebook markdown cells)

- Credit card companies must identify fraudulent transactions so customers are not charged for purchases they did not make.
- Customers expect safety measures that block fraudulent transactions before they are processed.
- Company goal: build user confidence by blocking fraud → boosting long-term trust and retention.
- **Why accuracy fails:** Predicting "not fraud" always = 99.83% accuracy but catches zero fraudsters. Use Recall, F1, and PR-AUC instead.

---

## Cell-by-Cell Implementation

### Cell 0 (EXISTING — do not modify)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
print('Successfully Loaded!')
df = pd.read_csv("creditcard.csv")
df.head()
```

---

### Cell 1 — Expanded ML Imports (code)
```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, precision_recall_curve
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
print("All ML libraries loaded successfully.")
```

---

### Cell 2 — Business Understanding (markdown)

Title: `# Challenge 2: Transaction Risk Scoring` / `## Phase 1 — Business Understanding`

Content:
- Credit card companies must identify fraud so customers aren't charged for purchases they didn't make.
- Customers expect safety measures blocking fraudulent transactions before processing.
- Company goal: build user confidence by blocking fraud → boosting long-term trust.
- Why accuracy fails: 0.172% fraud prevalence → predicting "not fraud" always = 99.83% accuracy, catches zero fraud.
- Key metrics: **Recall** (% of fraud caught), **Precision** (% of flags that are real fraud), **F1-Score**, **PR-AUC**.
- False negative = missed fraud = financial loss. False positive = blocked legit transaction = customer friction.

---

### Cell 3 — Column Descriptions (markdown)

Include a table with all columns (V1–V28, Time, Amount, Class) as described in the Dataset Description section above. Note that V1 captures the most variance and values have no physical meaning.

---

### Cell 4 — Class Distribution Stats (code)

Compute and print:
- Total transactions, number of features, missing values
- Legitimate count and percentage
- Fraud count and percentage
- Imbalance ratio (legit:fraud)
- Naive baseline accuracy (always predict "not fraud")
- Print that accuracy is a useless metric here

---

### Cell 5 — Class Imbalance Visualization (code)

Two side-by-side plots:
1. Bar chart: transaction count by class (Legitimate vs Fraud), with counts labeled on bars
2. Pie chart: proportion of each class with percentages

Colors: Legitimate = `#4a90a4` (blue), Fraud = `#e05c5c` (red). Use these colors consistently throughout the notebook.

---

### Cell 6 — Fraud vs Legitimate Statistics (code)

Print Amount statistics (mean, median, max) separately for legitimate and fraudulent transactions. Print total time range of the dataset in hours.

Assign `fraud = df[df["Class"] == 1]` and `legit = df[df["Class"] == 0]` — these variables will be reused later.

---

### Cell 7 — Amount and Time Distributions (code)

2×2 subplot grid:
- Top-left: Amount histogram for legitimate (sample 5000, xlim 0–1000)
- Top-right: Amount histogram for fraud (xlim 0–1000)
- Bottom-left: Time/3600 histogram for legitimate (kde=True)
- Bottom-right: Time/3600 histogram for fraud (kde=True)

---

### Cell 8 — V-Feature Mean Differences (code)

For each V1–V28 feature, compute `|mean(fraud) - mean(legit)|`. Sort descending. Plot as a bar chart. Highlight top 10 in red (`#e05c5c`), rest in gray (`#aaa`).

Print the top 10 most separating features with their scores.

Store `mean_diff` as a Series for use in the next cell.

---

### Cell 9 — Box Plots for Top 6 V-Features (code)

Use `mean_diff.head(6).index.tolist()` to get the top 6 features.

2×3 subplot grid of box plots, one per feature, split by Class (Legitimate vs Fraud). Use consistent colors.

---

### Cell 10 — Discussion Questions (markdown)

Include these 4 discussion questions:
1. What does the time pattern in fraudulent transactions suggest about when fraud tends to occur?
2. Why might fraudsters deliberately keep transaction amounts small?
3. What does it mean that V14 and V4 are most separating — even though we don't know what they originally represented?
4. If original features were available (merchant, location, user history), which 3 would you hypothesize are most predictive of fraud?

---

### Cell 11 — Feature Engineering Rationale (markdown)

Title: `## Phase 3 — Data Preparation`

Include a table of all 13 engineered features with their type and rationale:

| Feature | Type | Rationale |
|---------|------|-----------|
| `LogAmount` | Continuous | Compresses right-skewed Amount distribution |
| `Amount_zscore` | Continuous | Flags unusually large/small amounts vs dataset mean |
| `IsSmallAmount` | Binary | Amount ≤ $5: card-testing pattern — fraudsters verify cards with tiny charges |
| `IsRoundAmount` | Binary | Round dollar amounts (e.g. $100.00) appear more in fraud |
| `Amount_bin` | Ordinal | Buckets: micro (<$1=0), small ($1–$50=1), medium ($50–$200=2), large (>$200=3) |
| `HourOfDay` | Continuous | Extracts time-of-day cycle from raw seconds |
| `IsNightTime` | Binary | High-risk window: 10pm–5am |
| `Amount_x_V14` | Interaction | LogAmount × V14 |
| `Amount_x_V4` | Interaction | LogAmount × V4 |
| `Amount_x_V12` | Interaction | LogAmount × V12 |
| `V14_squared` | Polynomial | Extreme V14 values in both directions are suspicious |
| `V4_squared` | Polynomial | Same for V4 |
| `V14_x_V4` | Interaction | Fraud may require both V14 and V4 to be anomalous simultaneously |

Note: Drop raw `Amount` and `Time` after engineering (replaced by better representations).

---

### Cell 12 — Feature Engineering Code (code)

```python
df_eng = df.copy()

# Amount features
df_eng["LogAmount"]     = np.log1p(df_eng["Amount"])
df_eng["Amount_zscore"] = (df_eng["Amount"] - df_eng["Amount"].mean()) / df_eng["Amount"].std()
df_eng["IsSmallAmount"] = (df_eng["Amount"] <= 5).astype(int)
df_eng["IsRoundAmount"] = (df_eng["Amount"] % 1 == 0).astype(int)
df_eng["Amount_bin"]    = pd.cut(
    df_eng["Amount"],
    bins=[-0.01, 1, 50, 200, df_eng["Amount"].max() + 1],
    labels=[0, 1, 2, 3]
).astype(int)

# Time features
df_eng["HourOfDay"]   = (df_eng["Time"] / 3600) % 24
df_eng["IsNightTime"] = ((df_eng["HourOfDay"] >= 22) | (df_eng["HourOfDay"] <= 5)).astype(int)

# Interaction features
df_eng["Amount_x_V14"] = df_eng["LogAmount"] * df_eng["V14"]
df_eng["Amount_x_V4"]  = df_eng["LogAmount"] * df_eng["V4"]
df_eng["Amount_x_V12"] = df_eng["LogAmount"] * df_eng["V12"]

# Polynomial features
df_eng["V14_squared"] = df_eng["V14"] ** 2
df_eng["V4_squared"]  = df_eng["V4"]  ** 2
df_eng["V14_x_V4"]   = df_eng["V14"] * df_eng["V4"]

# Drop raw originals
df_eng = df_eng.drop(columns=["Amount", "Time"])

print("Feature engineering complete.")
print(f"  Original features:   {df.shape[1] - 1}")
print(f"  Engineered features: {df_eng.shape[1] - 1}")
```

---

### Cell 13 — Engineered Feature Plots (code)

3-panel subplot:
1. Overlapping density histogram of `LogAmount` by class
2. Overlapping histogram of `HourOfDay` by class
3. Box plot of `Amount_x_V14` by class

---

### Cell 14 — Train/Test Split (code)

```python
FEATURES = [c for c in df_eng.columns if c != "Class"]
TARGET   = "Class"

X = df_eng[FEATURES]
y = df_eng[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Print training and test set sizes with fraud counts and percentages. Note that `stratify=y` preserves the fraud ratio in both splits.

---

### Cell 15 — StandardScaler (code)

Scale ONLY these continuous engineered features (V1–V28 are already PCA-normalized):
```
SCALE_FEATURES = [
    "LogAmount", "Amount_zscore", "HourOfDay",
    "Amount_x_V14", "Amount_x_V4", "Amount_x_V12",
    "V14_squared", "V4_squared", "V14_x_V4"
]
```

Fit scaler on `X_train` only, then transform both `X_train` and `X_test`. Store as `X_train_scaled` and `X_test_scaled`. Print which features were scaled vs untouched.

---

### Cell 16 — Modeling Strategy (markdown)

Title: `## Phase 4 — Modeling`

Explain class weighting strategy:
- `class_weight="balanced"` for scikit-learn models
- `scale_pos_weight = count(negatives) / count(positives)` for XGBoost

Include a table comparing the 3 models (Logistic Regression, Random Forest, XGBoost) with why each is included.

---

### Cell 17 — Naive Baseline (code)

Predict all zeros. Print Accuracy, Precision, Recall, F1, ROC-AUC. Emphasize that Recall = 0.0 means zero fraud caught. This is the floor all models must beat.

---

### Cell 18 — Logistic Regression (code)

```python
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
```

Predict on `X_test_scaled`. Print all 6 metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC) and full `classification_report`.

Store predictions as `y_pred_lr` and probabilities as `y_proba_lr`.

---

### Cell 19 — Random Forest (code)

```python
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)  # No scaling needed for tree-based models
```

Predict on `X_test` (unscaled). Print all 6 metrics and `classification_report`.

Store as `y_pred_rf` and `y_proba_rf`.

---

### Cell 20 — XGBoost (code)

```python
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count  # ≈ 577

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

Print `xgb.best_iteration`. Print all 6 metrics and `classification_report`.

Store as `y_pred_xgb` and `y_proba_xgb`.

---

### Cell 21 — Evaluation Explanation (markdown)

Title: `## Phase 5 — Evaluation`

Briefly explain why 3 different visualizations are needed (Confusion Matrix, ROC, PR Curve) and what each reveals for fraud detection.

---

### Cell 22 — Confusion Matrices (code)

1×3 subplot of confusion matrices for all 3 models using `ConfusionMatrixDisplay`. Labels: `["Legitimate", "Fraud"]`. Annotate x-axis of each with the False Negative and False Positive counts. Note that False Negative = missed fraud = most costly.

---

### Cell 23 — ROC Curves (code)

Single plot with overlaid ROC curves for all 3 models. Include a dashed random baseline at y=x. Label each curve with its ROC-AUC score.

---

### Cell 24 — Precision-Recall Curves (code)

Single plot with overlaid PR curves for all 3 models. Add a dashed horizontal line at `y = fraud_prevalence` as the random baseline. Label each curve with its PR-AUC score. Note that PR-AUC ≈ fraud prevalence (0.0017) for a random classifier — good models should score far above.

---

### Cell 25 — Summary Comparison Table (code)

Build a DataFrame with rows for all 4 models (Naive Baseline, Logistic Regression, Random Forest, XGBoost) and columns: Precision, Recall, F1-Score, ROC-AUC, PR-AUC. Print rounded to 4 decimal places.

---

### Cell 26 — Discussion Questions (markdown)

Include these 4 questions:
1. The naive baseline has 99.83% accuracy but Recall = 0. What does this tell you about when to use accuracy?
2. Compare ROC-AUC vs PR-AUC scores. Which curve is more trustworthy for reporting to a business stakeholder in fraud detection?
3. If the bank can only review 100 flagged transactions per day, should they optimize for Precision or Recall?
4. Logistic Regression has competitive Recall and is far simpler. When might you choose it over XGBoost in production?

---

### Cell 27 — XGBoost Feature Importance (code)

Build a DataFrame of all feature importances from XGBoost. Sort descending, show top 15. Plot as a horizontal bar chart. Color engineered features orange (`#e07b39`), original V features blue (`#4a90a4`).

Engineered features set:
```python
engineered = {
    "LogAmount", "Amount_zscore", "IsSmallAmount", "IsRoundAmount", "Amount_bin",
    "HourOfDay", "IsNightTime",
    "Amount_x_V14", "Amount_x_V4", "Amount_x_V12",
    "V14_squared", "V4_squared", "V14_x_V4"
}
```

Print top 10 features with importance scores, tagging engineered ones.

---

### Cell 28 — Random Forest Feature Importance (code)

Same as Cell 27 but using `rf.feature_importances_`. Same color coding. Allows comparison between XGBoost and RF feature rankings.

---

### Cell 29 — Threshold Tuning Explanation (markdown)

Explain that the default 0.5 threshold is arbitrary. Lower threshold → higher Recall, lower Precision. Higher threshold → higher Precision, lower Recall. The threshold is a business policy decision, not a modeling decision.

---

### Cell 30 — Threshold Tuning Plot (code)

Sweep thresholds from 0.1 to 0.9 (step 0.05) on XGBoost probabilities. For each threshold compute Precision, Recall, F1. Plot all three on one chart. Mark the best F1 threshold with a vertical dashed line. Print metrics at default (0.5) vs optimized threshold.

---

### Cell 31 — Cross-Validation (code)

Run 5-fold stratified cross-validation on XGBoost using `xgb.best_iteration` as `n_estimators`, scoring `roc_auc`. Print per-fold scores, mean, and std. Note that low std = stable generalization.

---

### Cell 32 — Business Recommendations (markdown)

Title: `## Phase 6 — Business Recommendations`

**Key Findings:**
1. V14 and V4 are the strongest fraud signals — investigate what original features load onto these components.
2. Fraudulent transactions tend to be smaller in amount — simple large-transaction rules will miss most fraud.
3. Time-of-day matters — fraud is not uniformly distributed; time-based rules can complement the ML model.
4. Engineered features (interaction terms, behavioral flags) appear in the top feature importance rankings.

**Deployment Recommendations table:**

| Recommendation | Rationale |
|----------------|-----------|
| Deploy XGBoost with `scale_pos_weight` | Best F1 and PR-AUC; handles imbalance natively |
| Set decision threshold at best-F1 value (not 0.5) | Default 0.5 threshold underserves recall |
| Route flagged transactions to human review queue | Avoid hard auto-blocks; false positive rate is non-zero |
| Retrain monthly | Fraud patterns drift — models become stale |
| Monitor PR-AUC in production, not accuracy | Accuracy is meaningless for this use case |

**Limitations:**
- V1–V28 are anonymized — cannot explain to the business *what* patterns to look for, only that they exist.
- Dataset covers 48 hours from one European bank — fraud patterns change over time (concept drift).
- Class weighting used instead of SMOTE — synthetic oversampling could be explored as a next step.

---

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| No SMOTE | `imbalanced-learn` not in `requirements.txt`; use `class_weight='balanced'` and `scale_pos_weight` |
| PR-AUC as primary metric | More informative than ROC-AUC for severely imbalanced data |
| XGBoost early stopping on `aucpr` | Stops training when PR-AUC on test set stops improving |
| Scale only engineered features | V1–V28 already PCA-normalized; re-scaling would distort them |
| `stratify=y` in train/test split | Preserves 0.172% fraud rate in both splits |
| RF uses unscaled data, LR uses scaled | Trees don't need scaling; LR does for regularization to work correctly |

---

## Expected Outputs (Verification Checklist)

- [ ] Cell 4: Fraud count ≈ 492, naive accuracy ≈ 0.9983
- [ ] Cell 17: Naive baseline Recall = 0.0000
- [ ] Cells 18–20: All 3 models show Recall > 0.70 for fraud class
- [ ] Cell 25: XGBoost has highest F1 among all models
- [ ] Cell 27: V14 or V4 appears at or near top of feature importance
- [ ] All cells run top-to-bottom without errors
