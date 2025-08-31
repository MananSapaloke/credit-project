Credit Risk Prediction Project

Author: Manan Sapaloke
Email: manansapaok@gmail.com
GitHub: https://github.com/MananSapaloke

Project Overview

Financial institutions rely on credit risk assessment to minimize loan defaults. This project builds a robust pipeline to predict the probability of loan default using multiple datasets, advanced feature engineering, and machine learning with explainability via SHAP.

Objectives:

Handle large financial datasets efficiently.

Aggregate historical loan, bureau, and credit card data.

Engineer advanced features to improve predictive power.

Train LightGBM models with cross-validation.

Provide explainability using SHAP values for global and local interpretation.

Datasets Used
Dataset	Rows	Description
application_train.csv	307,511	Primary loan applications
bureau.csv	171,642	Customer credit bureau records
bureau_balance.csv	1,177,747	Historical bureau balances
credit_card_balance.csv	10,000,000+	Credit card usage
installments_payments.csv	1,000,000+	Installment payments history
POS_CASH_balance.csv	1,000,000+	Point-of-sale / cash loan balances
previous_application.csv	540,000	Past loan applications
Project Workflow
Step 1: Data Loading and Cleaning

Loaded datasets in batches to avoid memory issues.

Removed duplicates and handled missing values.

Saved cleaned datasets in parquet format.

Step 2: Batch-wise Aggregation

Aggregated historical data per customer (SK_ID_CURR).

Aggregated features per dataset:

bureau: 15 features

previous_application: 18 features

POS_CASH_balance: 9 features

installments_payments: 11 features

credit_card_balance: 14 features

Step 3: Merging Aggregated Datasets

Merged application_train with all aggregated datasets to create a master dataset.

Resulting dataset: 307,511 rows × 184 features

Step 4: Feature Engineering

Added advanced features such as ratios, counts, sums, and time-based metrics.

Total features after feature engineering: 192

After encoding categorical variables: 332 features

Step 5: Train-Validation Split

Stratified split maintaining target distribution:

Training: 246,008 rows

Validation: 61,503 rows

Step 6: Baseline LightGBM Model

Trained with early stopping and validation set.

Baseline ROC-AUC: 0.778

Step 7: Feature Importance & SHAP Analysis

Computed global feature importance using LightGBM.

Top 3 features:

EXT_SOURCE_3

EXT_SOURCE_2

EXT_SOURCE_1

SHAP plots for global and local feature interpretation.




Step 8: Hyperparameter Tuning (Optuna)

Optimized LightGBM parameters using Optuna:

learning_rate, num_leaves, max_depth, feature_fraction, bagging_fraction, lambda_l1/l2

Improved model performance (~ROC-AUC 0.76 on validation)

Step 9: Model Saving

Saved trained LightGBM model for inference: lgb_model_step32.txt

Step 10: Advanced Model Interpretation

Global and local SHAP explanations highlight feature contributions.

Most predictive features include external credit scores, age, payment behavior.

Project Insights

EXT_SOURCE_1,2,3 are the most predictive features.

Payment history and age significantly influence credit risk.

Aggregating historical data improves model accuracy and interpretability.

Usage

Clone the repository:

git clone https://github.com/MananSapaloke/credit-risk-project.git
cd credit-risk-project


Install required Python packages:

pip install -r requirements.txt


Load cleaned datasets (.parquet) from /data directory.

Run notebooks sequentially:

01_data_cleaning.ipynb

02_aggregation.ipynb

03_feature_engineering.ipynb

04_model_training.ipynb

05_shap_analysis.ipynb

Project Outputs

Cleaned datasets: /data/cleaned/

Aggregated datasets: /data/aggregated/

Feature-engineered dataset: /data/master_train_fe_encoded.parquet

Trained LightGBM model: /models/lgb_model_step32.txt

Feature importance & SHAP plots: /images/

Future Enhancements

Deploy an interactive Streamlit dashboard for applicant risk exploration.

Integrate external socio-economic datasets for improved predictions.

Implement an automated pipeline for periodic model retraining.

Technologies Used

Python: pandas, numpy, scikit-learn, lightgbm, shap, optuna

Data Storage: Parquet for batch processing

Visualization: matplotlib, seaborn, SHAP plots

Optimization: Optuna for hyperparameter tuning
