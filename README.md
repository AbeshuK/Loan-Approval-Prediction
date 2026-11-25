# Loan Approval Prediction

A binary classification project that predicts whether a loan application will be **approved or rejected**, with a focus on evaluating models correctly on **imbalanced data** (precision, recall, F1) instead of relying on accuracy.

Built as **Task 1 (Level 2, Machine Learning track)** of the Elevvo Internship Program.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Author](#author)

---

## Project Overview

**Goal:** Build a classifier that predicts loan approval from applicant and loan features, and evaluate it with metrics that make sense when classes are not balanced.

**What this project covers**
- Data cleaning (stray whitespace, ID column removal, target encoding)
- Missing value handling and categorical encoding inside a leak-free pipeline
- Logistic Regression vs Decision Tree
- Imbalance handling: `class_weight="balanced"` and SMOTE
- Hyperparameter tuning with cross-validation
- Decision-threshold tuning driven by a business objective (precision on approvals)
- Model interpretation with tree importance and permutation importance

**Tech stack:** Python 3.12, pandas, scikit-learn, imbalanced-learn, matplotlib, Jupyter

---

## Dataset

- **Source:** [Loan-Approval-Prediction-Dataset (Kaggle)](https://www.kaggle.com/) (search for "Loan Approval Prediction Dataset")
- **Size:** TODO rows x TODO columns
- **Target:** `loan_status` (Approved = 1, Rejected = 0)
- **Features:** TODO (e.g. number of dependents, education, self-employment, annual income, loan amount, loan term, CIBIL score, asset values)
- **Class balance:** TODO% Approved / TODO% Rejected
- **Missing values:** TODO (state clearly if there were none)

> The raw CSV is not included in this repository. Download it from Kaggle and place it in the project root as `loan_approval_dataset.csv`.

---

## Approach

1. **Cleaning:** stripped whitespace from column names and values, dropped `loan_id`, encoded the target.
2. **EDA:** class distribution, missing values, feature distributions, correlation with approval, approval rate by category.
3. **Split:** stratified 80/20 train/test split (`random_state=42`). The test set was used only for final evaluation.
4. **Preprocessing (inside the pipeline):**
   - Numeric: median imputation, then standard scaling
   - Categorical: most-frequent imputation, then one-hot encoding
5. **Models compared** (5-fold stratified cross-validation on the training set):

   | Model family | Variants |
   |---|---|
   | Dummy baseline | always predict majority class |
   | Logistic Regression | plain, `class_weight`, SMOTE |
   | Decision Tree | plain, `class_weight`, SMOTE |

6. **Tuning:** grid search on the Decision Tree (`max_depth`, `min_samples_leaf`, `criterion`), optimizing F1.
7. **Threshold tuning:** chose the decision threshold from out-of-fold training predictions, targeting a minimum precision for the Approved class, then maximizing recall.
8. **Interpretation:** tree feature importance and permutation importance.

> SMOTE is applied through `imblearn.pipeline.Pipeline`, so it only touches training folds and never leaks into validation or test data.

---

## Results

### Cross-validation (training data, 5-fold)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Dummy (always majority) | TODO | TODO | TODO | TODO | TODO |
| LogReg | TODO | TODO | TODO | TODO | TODO |
| LogReg + class_weight | TODO | TODO | TODO | TODO | TODO |
| LogReg + SMOTE | TODO | TODO | TODO | TODO | TODO |
| Tree | TODO | TODO | TODO | TODO | TODO |
| Tree + class_weight | TODO | TODO | TODO | TODO | TODO |
| Tree + SMOTE | TODO | TODO | TODO | TODO | TODO |

### Final test-set performance (Approved class)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LogReg + class_weight | 0.50 | TODO | TODO | TODO | TODO | TODO |
| Decision Tree (tuned) | 0.50 | TODO | TODO | TODO | TODO | TODO |
| LogReg + class_weight (tuned threshold) | TODO | TODO | TODO | TODO | TODO | TODO |

**Best tuned Decision Tree parameters:** TODO

### Visuals

<!-- Save your figures to an images/ folder and uncomment these lines -->
<!-- ![Confusion Matrix](images/confusion_matrix.png) -->
<!-- ![Precision-Recall Curve](images/precision_recall_curve.png) -->
<!-- ![Feature Importance](images/feature_importance.png) -->

---

## Key Findings

- **Best model:** TODO, with F1 of TODO on the test set.
- **Logistic Regression vs Decision Tree:** TODO (which generalized better, and was the tree overfitting before tuning?).
- **Effect of imbalance handling:** TODO (did SMOTE or class weights change results meaningfully, or is the imbalance too mild to matter?).
- **Threshold tuning:** TODO (how recall changed when requiring precision >= TODO).
- **Most influential features:** TODO (e.g. credit score, loan amount).
- **Why accuracy alone is misleading:** the dummy baseline already reaches TODO% accuracy without learning anything.

---

## Project Structure

```
Loan-Approval-Prediction/
├── loan_approval_prediction.ipynb   # Full analysis and modeling notebook
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── .gitignore
└── images/                          # Saved figures used in this README
```

---

## Getting Started

### Prerequisites
- Python **3.12** (or 3.11/3.13). Avoid pre-release Python versions, since data-science packages often break on them.
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run

1. Download the dataset from Kaggle and save it as `loan_approval_dataset.csv` in the project root.
2. Launch Jupyter (or open the notebook in VS Code and select the `.venv` kernel):
   ```bash
   jupyter notebook loan_approval_prediction.ipynb
   ```
3. Run all cells from top to bottom.

### `requirements.txt`

```
pandas
scikit-learn
imbalanced-learn
matplotlib
ipykernel
jupyter
```

---

## Limitations

- The dataset is small and likely synthetic or simplified, so results may not transfer to real lending decisions.
- One dominant feature (such as credit score) can make the problem look easier than it is.
- No fairness or bias audit was performed. Real credit models must be checked for discrimination across protected groups.
- Metrics depend on the chosen threshold, which should be set from actual business costs rather than a fixed precision target.
- Single train/test split plus cross-validation; no external validation set.

---

## Future Work

- Try gradient boosting (XGBoost / LightGBM) and compare against the baselines
- Calibrate predicted probabilities
- Use a cost-sensitive threshold based on estimated loss from bad approvals vs lost good customers
- Analyze performance across applicant subgroups
- Wrap the model in a FastAPI service with Docker (see the MLOps task)

---

## Author

**Abeshu Kebede**
Software Engineer | Machine Learning Intern at Elevvo
Portfolio: [abeshukebede.vercel.app](https://abeshukebede.vercel.app)

---

## Acknowledgments

- Dataset provided on Kaggle by its original author
- Elevvo Internship Program for the task design