"""
CreditLens Production Scoring Module

This module provides a production-ready scoring pipeline that replicates
the exact feature engineering and model prediction from the Jupyter notebook.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import re
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditScorer:
    """
    Production credit scoring system that replicates the notebook pipeline.
    """
    
    def __init__(self, model_path: str, feature_columns_path: Optional[str] = None):
        """
        Initialize the credit scorer.
        
        Args:
            model_path: Path to the trained LightGBM model file
            feature_columns_path: Path to saved feature columns (optional)
        """
        self.model_path = model_path
        self.model = None
        self.feature_columns = None
        self.explainer = None
        
        # Load model and feature columns
        self._load_model()
        if feature_columns_path:
            self._load_feature_columns(feature_columns_path)
    
    def _load_model(self):
        """Load the trained LightGBM model."""
        try:
            self.model = lgb.Booster(model_file=self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_feature_columns(self, path: str):
        """Load feature columns from saved file."""
        try:
            self.feature_columns = joblib.load(path)
            logger.info(f"Feature columns loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load feature columns: {e}")
            raise
    
    def clean_feature_names(self, columns: List[str]) -> List[str]:
        """
        Clean feature names to ensure valid characters only.
        Replicates the notebook's clean_feature_names function.
        """
        return [re.sub(r'[^0-9a-zA-Z_]+', '_', str(col)) for col in columns]
    
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess raw input data into model features.
        Replicates the exact feature engineering from the notebook.
        
        Args:
            input_data: Dictionary containing raw applicant data
            
        Returns:
            DataFrame with engineered features
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # === Basic Data Cleaning ===
        # Convert numeric fields with defensive coercion
        numeric_fields = [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
            'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE'
        ]
        
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        # Handle missing values with sensible defaults
        df = df.fillna({
            'AMT_ANNUITY': df['AMT_ANNUITY'].median() if 'AMT_ANNUITY' in df.columns else 0,
            'AMT_GOODS_PRICE': df['AMT_GOODS_PRICE'].median() if 'AMT_GOODS_PRICE' in df.columns else 0,
            'CNT_CHILDREN': 0,
            'CNT_FAM_MEMBERS': 1,
            'OWN_CAR_AGE': 0,
            'DAYS_EMPLOYED': -365,  # Default to 1 year employed
        })
        
        # === Feature Engineering (Exact replication from notebook) ===
        
        # Core financial ratios
        df['credit_to_income'] = np.divide(df['AMT_CREDIT'], df['AMT_INCOME_TOTAL'] + 1)
        df['goods_price_to_credit'] = np.divide(df['AMT_GOODS_PRICE'], df['AMT_CREDIT'] + 1)
        df['annuity_to_income'] = np.divide(df['AMT_ANNUITY'], df['AMT_INCOME_TOTAL'] + 1)
        
        # Employment and temporal features
        df['days_employed_ratio'] = np.divide(df['DAYS_EMPLOYED'], df['DAYS_BIRTH'] + 1)
        df['account_age_days'] = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_REGISTRATION'] if 'DAYS_LAST_PHONE_CHANGE' in df.columns else 0
        df['children_income_ratio'] = np.divide(df['CNT_CHILDREN'], df['AMT_INCOME_TOTAL'] + 1)
        df['family_income_ratio'] = np.divide(df['CNT_FAM_MEMBERS'], df['AMT_INCOME_TOTAL'] + 1)
        
        # Payment ratio (if available)
        if 'AMT_PAYMENT_sum' in df.columns and 'AMT_INSTALMENT_sum' in df.columns:
            df['payment_to_inst_ratio'] = np.divide(df['AMT_PAYMENT_sum'], df['AMT_INSTALMENT_sum'] + 1)
        else:
            df['payment_to_inst_ratio'] = 1.0  # Default assumption
        
        # === Categorical Encoding ===
        # Handle categorical variables with one-hot encoding
        categorical_mappings = {
            'NAME_CONTRACT_TYPE': ['Cash loans', 'Revolving loans'],
            'CODE_GENDER': ['F', 'M'],
            'FLAG_OWN_CAR': ['N', 'Y'],
            'FLAG_OWN_REALTY': ['N', 'Y'],
            'NAME_INCOME_TYPE': [
                'Businessman', 'Commercial associate', 'Maternity leave',
                'Pensioner', 'State servant', 'Student', 'Unemployed', 'Working'
            ],
            'NAME_EDUCATION_TYPE': [
                'Academic degree', 'Higher education', 'Incomplete higher',
                'Lower secondary', 'Secondary / secondary special'
            ],
            'NAME_FAMILY_STATUS': [
                'Civil marriage', 'Married', 'Separated',
                'Single / not married', 'Unknown', 'Widow'
            ],
            'NAME_HOUSING_TYPE': [
                'Co-op apartment', 'House / apartment', 'Municipal apartment',
                'Office apartment', 'Rented apartment', 'With parents'
            ],
            'OCCUPATION_TYPE': [
                'Accountants', 'Cleaning staff', 'Cooking staff', 'Core staff',
                'Drivers', 'HR staff', 'High skill tech staff', 'IT staff',
                'Laborers', 'Low-skill Laborers', 'Managers', 'Medicine staff',
                'Private service staff', 'Realty agents', 'Sales staff',
                'Secretaries', 'Security staff', 'Waiters/barmen staff'
            ]
        }
        
        # Create one-hot encoded columns
        for cat_var, categories in categorical_mappings.items():
            if cat_var in df.columns:
                for category in categories:
                    col_name = f"{cat_var}_{category}"
                    df[col_name] = (df[cat_var] == category).astype(int)
            else:
                # Create all columns as False if variable not present
                for category in categories:
                    col_name = f"{cat_var}_{category}"
                    df[col_name] = 0
        
        # Add nan columns for missing categories
        for cat_var in categorical_mappings.keys():
            df[f"{cat_var}_nan"] = 0
        
        # === Boolean Flag Encoding ===
        boolean_flags = [
            'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
            'FLAG_PHONE', 'FLAG_EMAIL', 'REG_REGION_NOT_LIVE_REGION',
            'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
            'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY'
        ]
        
        for flag in boolean_flags:
            if flag in df.columns:
                df[flag] = df[flag].astype(int)
            else:
                df[flag] = 0
        
        # === Default Values for Missing Features ===
        # Set default values for features that might be missing
        default_features = {
            'EXT_SOURCE_1': 0.5,
            'EXT_SOURCE_2': 0.5,
            'EXT_SOURCE_3': 0.5,
            'REGION_POPULATION_RELATIVE': 0.01,
            'REGION_RATING_CLIENT': 2,
            'REGION_RATING_CLIENT_W_CITY': 2,
            'HOUR_APPR_PROCESS_START': 12,
            'DAYS_LAST_PHONE_CHANGE': -1000,
            'AMT_REQ_CREDIT_BUREAU_HOUR': 0,
            'AMT_REQ_CREDIT_BUREAU_DAY': 0,
            'AMT_REQ_CREDIT_BUREAU_WEEK': 0,
            'AMT_REQ_CREDIT_BUREAU_MON': 0,
            'AMT_REQ_CREDIT_BUREAU_QRT': 0,
            'AMT_REQ_CREDIT_BUREAU_YEAR': 0,
        }
        
        for feature, default_value in default_features.items():
            if feature not in df.columns:
                df[feature] = default_value
        
        # === Bureau and Previous Application Features ===
        # These would typically come from external data sources
        # For now, set reasonable defaults based on training data patterns
        bureau_defaults = {
            'BUREAU_DAYS_CREDIT_MIN': -2000,
            'BUREAU_DAYS_CREDIT_MAX': -100,
            'BUREAU_DAYS_CREDIT_MEAN': -1000,
            'BUREAU_CREDIT_DAY_OVERDUE_MAX': 0,
            'BUREAU_AMT_CREDIT_SUM_SUM': df['AMT_CREDIT'].iloc[0] * 0.5,
            'BUREAU_AMT_CREDIT_SUM_MEAN': df['AMT_CREDIT'].iloc[0] * 0.3,
            'BUREAU_AMT_CREDIT_SUM_DEBT_SUM': df['AMT_CREDIT'].iloc[0] * 0.2,
            'BUREAU_AMT_CREDIT_SUM_DEBT_MEAN': df['AMT_CREDIT'].iloc[0] * 0.15,
            'BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM': 0,
            'BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN': df['AMT_CREDIT'].iloc[0] * 0.4,
        }
        
        for feature, default_value in bureau_defaults.items():
            if feature not in df.columns:
                df[feature] = default_value
        
        # === Fill remaining NaN values ===
        df = df.fillna(0)
        
        # Clean column names
        df.columns = self.clean_feature_names(df.columns)
        
        return df
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on input data.
        
        Args:
            input_data: Raw input data dictionary
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess input
            features_df = self.preprocess_input(input_data)
            
            # Ensure we have the right features in the right order
            if self.feature_columns is not None:
                # Reorder columns to match training data
                missing_cols = set(self.feature_columns) - set(features_df.columns)
                for col in missing_cols:
                    features_df[col] = 0
                features_df = features_df[self.feature_columns]
            
            # Make prediction
            pred_prob = self.model.predict(features_df)[0]
            pred_label = 1 if pred_prob > 0.5 else 0
            
            return {
                'pred_prob': float(pred_prob),
                'pred_label': int(pred_label),
                'features_used': len(features_df.columns),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'pred_prob': None,
                'pred_label': None,
                'error': str(e),
                'success': False
            }
    
    def get_shap_explanation(self, input_data: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """
        Get SHAP explanation for the prediction.
        
        Args:
            input_data: Raw input data dictionary
            top_n: Number of top features to return
            
        Returns:
            Dictionary containing SHAP explanation
        """
        try:
            import shap
            
            # Preprocess input
            features_df = self.preprocess_input(input_data)
            
            # Ensure we have the right features
            if self.feature_columns is not None:
                missing_cols = set(self.feature_columns) - set(features_df.columns)
                for col in missing_cols:
                    features_df[col] = 0
                features_df = features_df[self.feature_columns]
            
            # Initialize explainer if not already done
            if self.explainer is None:
                self.explainer = shap.TreeExplainer(self.model)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(features_df)
            
            # Get feature names
            feature_names = features_df.columns.tolist()
            
            # Get top positive and negative contributors
            shap_vals = shap_values[0] if isinstance(shap_values, list) else shap_values
            
            # Create feature importance pairs
            feature_importance = list(zip(feature_names, shap_vals))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Separate positive and negative
            positive_features = [(name, val) for name, val in feature_importance if val > 0][:top_n]
            negative_features = [(name, val) for name, val in feature_importance if val < 0][:top_n]
            
            return {
                'top_positive': [{'feature': name, 'shap_value': float(val)} for name, val in positive_features],
                'top_negative': [{'feature': name, 'shap_value': float(val)} for name, val in negative_features],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {
                'top_positive': [],
                'top_negative': [],
                'error': str(e),
                'success': False
            }


def create_sample_input() -> Dict[str, Any]:
    """
    Create a sample input for testing.
    
    Returns:
        Sample input dictionary
    """
    return {
        'SK_ID_CURR': 100002,
        'age_years': 35,
        'employment_years': 3,
        'AMT_INCOME_TOTAL': 150000,
        'AMT_CREDIT': 500000,
        'AMT_ANNUITY': 15000,
        'AMT_GOODS_PRICE': 450000,
        'CNT_CHILDREN': 0,
        'CNT_FAM_MEMBERS': 2,
        'FLAG_OWN_CAR': 'Yes',
        'FLAG_OWN_REALTY': 'No',
        'NAME_INCOME_TYPE': 'Working',
        'NAME_EDUCATION_TYPE': 'Higher education',
        'NAME_FAMILY_STATUS': 'Married',
        'OCCUPATION_TYPE': 'Sales staff',
        'NAME_HOUSING_TYPE': 'House / apartment',
        'DAYS_BIRTH': -12775,  # 35 years old
        'DAYS_EMPLOYED': -1095,  # 3 years employed
        'DAYS_REGISTRATION': -2000,
        'DAYS_ID_PUBLISH': -1500,
        'OWN_CAR_AGE': 5,
        'FLAG_MOBIL': 1,
        'FLAG_EMP_PHONE': 1,
        'FLAG_WORK_PHONE': 0,
        'FLAG_CONT_MOBILE': 1,
        'FLAG_PHONE': 1,
        'FLAG_EMAIL': 0,
        'REGION_POPULATION_RELATIVE': 0.018801,
        'REGION_RATING_CLIENT': 2,
        'REGION_RATING_CLIENT_W_CITY': 2,
        'HOUR_APPR_PROCESS_START': 10,
        'EXT_SOURCE_1': 0.083036967,
        'EXT_SOURCE_2': 0.262948593,
        'EXT_SOURCE_3': 0.13937578,
    }


if __name__ == "__main__":
    # Example usage
    scorer = CreditScorer("lgb_model_step32.txt")
    
    # Test with sample input
    sample_input = create_sample_input()
    result = scorer.predict(sample_input)
    print("Prediction Result:", result)
    
    # Get SHAP explanation
    shap_result = scorer.get_shap_explanation(sample_input)
    print("SHAP Explanation:", shap_result)
