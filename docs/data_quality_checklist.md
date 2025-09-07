# CreditLens Data Quality Checklist

## Overview
This checklist ensures the `dashboard_data.csv` file is robust and ready for Power BI consumption, with proper type handling, null management, and column consistency.

## Pre-Export Checklist

### 1. Data Type Validation
- [ ] **Numeric Fields**: All numeric columns are properly typed (int64, float64)
  - `AMT_*` fields: float64
  - `CNT_*` fields: int64
  - `DAYS_*` fields: int64 (negative values preserved)
  - `pred_prob`: float64 (0.0 to 1.0 range)
  - `pred_label`: int64 (0 or 1)

- [ ] **Categorical Fields**: All categorical columns are properly encoded
  - One-hot encoded columns: int64 (0 or 1)
  - Boolean flags: int64 (0 or 1)
  - No string values in numeric columns

- [ ] **Date Fields**: Properly formatted if any date columns exist
  - ISO format (YYYY-MM-DD) or numeric days

### 2. Null Value Handling
- [ ] **No NaN Values**: All NaN values replaced with appropriate defaults
  - Numeric fields: 0 or median value
  - Categorical fields: 0 for one-hot encoded
  - Boolean fields: 0 (False)

- [ ] **No Empty Strings**: All empty strings converted to appropriate values
  - Use `pd.to_numeric(errors='coerce')` for numeric coercion
  - Replace empty strings with 0 or appropriate default

- [ ] **No "NA" Strings**: All "NA" string values converted to numeric 0
  - Check for string representations of missing values
  - Convert using `df.replace('NA', 0)`

### 3. Column Name Standardization
- [ ] **Valid Characters**: Column names contain only alphanumeric and underscores
  - Use `clean_feature_names()` function from notebook
  - Remove special characters: spaces, hyphens, dots, parentheses

- [ ] **Consistent Naming**: Follow consistent naming conventions
  - Prefixes: `AMT_`, `CNT_`, `DAYS_`, `FLAG_`, `NAME_`, `BUREAU_`, `PREV_`
  - Suffixes: `_mean`, `_sum`, `_max`, `_min`, `_Y`, `_N`

- [ ] **No Duplicates**: Ensure no duplicate column names
  - Check with `df.columns.duplicated().any()`

### 4. Feature Engineering Validation
- [ ] **Engineered Features**: All custom features properly calculated
  - `credit_to_income`: AMT_CREDIT / (AMT_INCOME_TOTAL + 1)
  - `goods_price_to_credit`: AMT_GOODS_PRICE / (AMT_CREDIT + 1)
  - `annuity_to_income`: AMT_ANNUITY / (AMT_INCOME_TOTAL + 1)
  - `days_employed_ratio`: DAYS_EMPLOYED / (DAYS_BIRTH + 1)
  - `account_age_days`: DAYS_LAST_PHONE_CHANGE - DAYS_REGISTRATION
  - `children_income_ratio`: CNT_CHILDREN / (AMT_INCOME_TOTAL + 1)
  - `family_income_ratio`: CNT_FAM_MEMBERS / (AMT_INCOME_TOTAL + 1)
  - `payment_to_inst_ratio`: AMT_PAYMENT_sum / (AMT_INSTALMENT_sum + 1)

- [ ] **Bureau Aggregates**: All bureau features properly aggregated
  - `BUREAU_AMT_CREDIT_SUM_*`: sum, mean, max statistics
  - `BUREAU_DAYS_CREDIT_*`: min, max, mean statistics
  - `BUREAU_CREDIT_DAY_OVERDUE_MAX`: maximum overdue days

- [ ] **Previous Application Features**: All prev features properly calculated
  - `PREV_AMT_*`: previous application amounts
  - `PREV_DAYS_DECISION_*`: decision timing statistics
  - `PREV_CNT_PAYMENT_*`: payment count statistics

### 5. Model Output Validation
- [ ] **Prediction Probabilities**: `pred_prob` column properly formatted
  - Range: 0.0 to 1.0
  - No NaN or infinite values
  - Proper float64 type

- [ ] **Prediction Labels**: `pred_label` column properly formatted
  - Values: 0 or 1 only
  - Proper int64 type
  - Consistent with pred_prob (threshold at 0.5)

### 6. Data Range Validation
- [ ] **Income Fields**: Reasonable ranges
  - `AMT_INCOME_TOTAL`: 0 to 10,000,000 (or reasonable upper bound)
  - `AMT_CREDIT`: 0 to 50,000,000 (or reasonable upper bound)
  - `AMT_ANNUITY`: 0 to 5,000,000 (or reasonable upper bound)

- [ ] **Count Fields**: Reasonable ranges
  - `CNT_CHILDREN`: 0 to 20
  - `CNT_FAM_MEMBERS`: 1 to 20
  - `CNT_PAYMENT`: 0 to 1000

- [ ] **Days Fields**: Reasonable ranges
  - `DAYS_BIRTH`: -36500 to -6570 (18 to 100 years old)
  - `DAYS_EMPLOYED`: -18250 to 0 (0 to 50 years employed)
  - `DAYS_REGISTRATION`: -10000 to 0

## Power BI Specific Checklist

### 7. DAX Compatibility
- [ ] **Column Names**: Compatible with DAX naming rules
  - No spaces or special characters
  - Start with letter or underscore
  - No reserved DAX keywords

- [ ] **Data Types**: Properly recognized by Power BI
  - Numeric columns: Whole Number or Decimal Number
  - Text columns: Text (if any)
  - Date columns: Date (if any)

### 8. Performance Optimization
- [ ] **File Size**: Optimized for Power BI performance
  - Consider compression (gzip)
  - Remove unnecessary columns
  - Optimize data types (int32 vs int64)

- [ ] **Column Count**: Reasonable number of columns
  - Power BI handles up to 16,000 columns
  - Current dataset should be well within limits

### 9. Calculated Column Compatibility
- [ ] **Age Calculation**: `AgeYears = ROUND(ABS(DAYS_BIRTH)/365,0)`
  - Ensure DAYS_BIRTH is properly formatted
  - Test calculation in Power BI

- [ ] **Employment Calculation**: `EmploymentYears = INT(-DAYS_EMPLOYED/365.25)`
  - Handle sentinel values (365243)
  - Test calculation in Power BI

- [ ] **Risk Segment**: `RiskSegment = SWITCH(TRUE(), pred_prob < 0.33, "Low", pred_prob < 0.66, "Medium", "High")`
  - Ensure pred_prob is properly formatted
  - Test calculation in Power BI

## Python Code for Data Quality Checks

```python
import pandas as pd
import numpy as np

def validate_dashboard_data(df):
    """Comprehensive data quality validation for dashboard_data.csv"""
    
    issues = []
    
    # 1. Check for NaN values
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        issues.append(f"NaN values found in columns: {nan_cols}")
    
    # 2. Check for empty strings
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        if df[col].eq('').any():
            issues.append(f"Empty strings found in column: {col}")
    
    # 3. Check for "NA" strings
    for col in string_cols:
        if df[col].eq('NA').any():
            issues.append(f"'NA' strings found in column: {col}")
    
    # 4. Check data types
    expected_numeric = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'pred_prob']
    for col in expected_numeric:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column {col} should be numeric but is {df[col].dtype}")
    
    # 5. Check prediction ranges
    if 'pred_prob' in df.columns:
        if df['pred_prob'].min() < 0 or df['pred_prob'].max() > 1:
            issues.append("pred_prob values outside 0-1 range")
    
    if 'pred_label' in df.columns:
        if not df['pred_label'].isin([0, 1]).all():
            issues.append("pred_label contains values other than 0 or 1")
    
    # 6. Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            issues.append(f"Infinite values found in column: {col}")
    
    # 7. Check column names
    invalid_chars = [' ', '-', '.', '(', ')', '/', '\\']
    for col in df.columns:
        if any(char in col for char in invalid_chars):
            issues.append(f"Column name contains invalid characters: {col}")
    
    # 8. Check for duplicate columns
    if df.columns.duplicated().any():
        issues.append("Duplicate column names found")
    
    return issues

def fix_data_quality_issues(df):
    """Fix common data quality issues"""
    
    # 1. Replace NaN values
    df = df.fillna(0)
    
    # 2. Replace empty strings and "NA" strings
    df = df.replace('', 0)
    df = df.replace('NA', 0)
    
    # 3. Convert numeric columns
    numeric_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 4. Clean column names
    def clean_feature_names(columns):
        return [re.sub(r'[^0-9a-zA-Z_]+', '_', str(col)) for col in columns]
    
    df.columns = clean_feature_names(df.columns)
    
    # 5. Ensure prediction columns are proper types
    if 'pred_prob' in df.columns:
        df['pred_prob'] = pd.to_numeric(df['pred_prob'], errors='coerce').fillna(0)
        df['pred_prob'] = df['pred_prob'].clip(0, 1)
    
    if 'pred_label' in df.columns:
        df['pred_label'] = pd.to_numeric(df['pred_label'], errors='coerce').fillna(0)
        df['pred_label'] = df['pred_label'].astype(int)
    
    return df

# Usage example
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('dashboard_data.csv')
    
    # Check for issues
    issues = validate_dashboard_data(df)
    if issues:
        print("Data quality issues found:")
        for issue in issues:
            print(f"- {issue}")
        
        # Fix issues
        df = fix_data_quality_issues(df)
        
        # Re-check
        issues_after = validate_dashboard_data(df)
        if not issues_after:
            print("All issues fixed!")
            df.to_csv('dashboard_data_cleaned.csv', index=False)
        else:
            print("Some issues remain:")
            for issue in issues_after:
                print(f"- {issue}")
    else:
        print("No data quality issues found!")
```

## Final Export Checklist

### 10. Pre-Export Validation
- [ ] Run data quality validation script
- [ ] Fix all identified issues
- [ ] Verify file size is reasonable (< 500MB)
- [ ] Test import into Power BI
- [ ] Verify all calculated columns work
- [ ] Check that all visuals render correctly

### 11. Export Settings
- [ ] Use UTF-8 encoding
- [ ] Include header row
- [ ] Use comma separator
- [ ] No index column
- [ ] Proper date formatting (if applicable)

### 12. Post-Export Verification
- [ ] File opens correctly in Excel
- [ ] File imports correctly in Power BI
- [ ] All columns recognized with correct types
- [ ] Calculated columns work as expected
- [ ] No error messages in Power BI

## Maintenance Checklist

### 13. Regular Monitoring
- [ ] Monitor file size growth
- [ ] Check for new data quality issues
- [ ] Verify model performance metrics
- [ ] Update documentation as needed

### 14. Version Control
- [ ] Tag stable versions
- [ ] Document changes between versions
- [ ] Maintain backup of previous versions
- [ ] Test rollback procedures

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Status**: Ready for Implementation
