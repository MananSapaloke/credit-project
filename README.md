# Credit Risk Prediction Project

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-0.3.3-green?logo=lightgbm)
![Optuna](https://img.shields.io/badge/Optuna-2.10-purple?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAS1BMVEUAAAD///////////////////////////////////////////////////////////////////////////////////////////////////////////////////8ml4JuAAAAD3RSTlMAEBAgIDAwQEBAYGBgYGBhH6GQcAAAAdSURBVBjTY2BgZGBoYmBiZGBkYGIAAuZgYECQGxAAALiCDMyYFfGAAAAAElFTkSuQmCC)
![SHAP](https://img.shields.io/badge/SHAP-0.41-orange)
![GitHub](https://img.shields.io/badge/GitHub-MananSapaloke-black?logo=github)

**Author:** Manan Sapaloke  
**Email:** manansapaok@gmail.com  
**GitHub:** [https://github.com/MananSapaloke](https://github.com/MananSapaloke)

---

## 📌 Project Overview

Financial institutions rely on credit risk assessment to minimize loan defaults. This project builds a **robust ML pipeline** to predict the probability of loan default using multiple datasets, advanced feature engineering, and explainable machine learning techniques.

**Objectives:**
- Handle large financial datasets efficiently.
- Aggregate historical loan, bureau, and credit card data.
- Engineer advanced features to improve predictive power.
- Train LightGBM models with cross-validation.
- Provide explainability using SHAP values for global and local interpretation.

---

## 🗂 Datasets Used

| Dataset | Rows | Description |
|---------|------|-------------|
| `application_train.csv` | 307,511 | Primary loan applications |
| `bureau.csv` | 171,642 | Customer credit bureau records |
| `bureau_balance.csv` | 1,177,747 | Historical bureau balances |
| `credit_card_balance.csv` | 10,000,000+ | Credit card usage |
| `installments_payments.csv` | 1,000,000+ | Installment payments history |
| `POS_CASH_balance.csv` | 1,000,000+ | Point-of-sale / cash loan balances |
| `previous_application.csv` | 540,000 | Past loan applications |

---

## 🔄 Project Workflow

### **Step 1: Data Loading & Cleaning**
- Loaded datasets in **batches** to avoid memory issues.
- Removed duplicates and handled missing values.
- Saved cleaned datasets in **Parquet format** for faster I/O.

### **Step 2: Batch-wise Aggregation**
- Aggregated historical data per customer (`SK_ID_CURR`) for each dataset.
- Key aggregations:
  - `bureau` → 15 features
  - `previous_application` → 18 features
  - `POS_CASH_balance` → 9 features
  - `installments_payments` → 11 features
  - `credit_card_balance` → 14 features

### **Step 3: Merging Aggregated Datasets**
- Merged `application_train` with all aggregated datasets to create a **master dataset**.
- Final dataset shape: **307,511 rows × 184 features**

### **Step 4: Feature Engineering**
- Added advanced features such as ratios, counts, sums, and temporal metrics.
- Features after engineering: 192  
- Features after encoding categorical variables: 332

### **Step 5: Train-Validation Split**
- Stratified split to maintain target distribution:
  - **Training:** 246,008 rows
  - **Validation:** 61,503 rows

### **Step 6: Baseline LightGBM Model**
- Trained with early stopping and validation set.
- **Baseline ROC-AUC:** 0.778

### **Step 7: Feature Importance & SHAP Analysis**
- Global feature importance computed using LightGBM.
- Top 3 features:
  1. `EXT_SOURCE_3`
  2. `EXT_SOURCE_2`
  3. `EXT_SOURCE_1`
- SHAP plots for global and local feature interpretation.

![Feature Importance](images/feature_importance.png)
![SHAP Summary Plot](images/shap_summary.png)

### **Step 8: Hyperparameter Tuning (Optuna)**
- Optimized LightGBM parameters using Optuna:
  - `learning_rate`, `num_leaves`, `max_depth`, `feature_fraction`, `bagging_fraction`, `lambda_l1/l2`
- Improved model performance (~ROC-AUC 0.76 on validation)

### **Step 9: Model Saving**
- Trained LightGBM model saved for inference:  
`/models/lgb_model_step32.txt`

### **Step 10: Advanced Model Interpretation**
- Global & local SHAP explanations highlight feature contributions.
- Most predictive features: **external credit scores, age, payment behavior**.

---

## 📊 Project Insights
- `EXT_SOURCE_1,2,3` are the most predictive features.
- Payment history and age significantly influence credit risk.
- Aggregating historical data improves model accuracy and interpretability.

---

## 🚀 Usage

1. Clone the repository:
```bash
git clone https://github.com/MananSapaloke/credit-risk-project.git
cd credit-risk-project
