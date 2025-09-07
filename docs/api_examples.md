# CreditLens API Examples

## Sample Input JSON (POST /api/v1/score)

### Basic Application
```json
{
  "SK_ID_CURR": 100002,
  "age_years": 35,
  "employment_years": 3,
  "AMT_INCOME_TOTAL": 150000,
  "AMT_CREDIT": 500000,
  "AMT_ANNUITY": 15000,
  "AMT_GOODS_PRICE": 450000,
  "CNT_CHILDREN": 0,
  "CNT_FAM_MEMBERS": 2,
  "FLAG_OWN_CAR": "Yes",
  "FLAG_OWN_REALTY": "No",
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Higher education",
  "NAME_FAMILY_STATUS": "Married",
  "OCCUPATION_TYPE": "Sales staff",
  "NAME_HOUSING_TYPE": "House / apartment",
  "previous_defaults": 0,
  "bureau_overdue_amount": 0,
  "pos_dpd_max": 0
}
```

### Minimal Application (Required Fields Only)
```json
{
  "AMT_INCOME_TOTAL": 75000,
  "AMT_CREDIT": 200000,
  "age_years": 28,
  "employment_years": 2
}
```

### High-Risk Application
```json
{
  "AMT_INCOME_TOTAL": 40000,
  "AMT_CREDIT": 300000,
  "age_years": 22,
  "employment_years": 1,
  "CNT_CHILDREN": 2,
  "CNT_FAM_MEMBERS": 4,
  "FLAG_OWN_CAR": "No",
  "FLAG_OWN_REALTY": "No",
  "NAME_INCOME_TYPE": "Student",
  "NAME_EDUCATION_TYPE": "Incomplete higher",
  "NAME_FAMILY_STATUS": "Single / not married",
  "OCCUPATION_TYPE": "Laborers",
  "NAME_HOUSING_TYPE": "With parents",
  "previous_defaults": 1,
  "bureau_overdue_amount": 5000,
  "pos_dpd_max": 30
}
```

### Low-Risk Application
```json
{
  "AMT_INCOME_TOTAL": 200000,
  "AMT_CREDIT": 400000,
  "age_years": 45,
  "employment_years": 15,
  "CNT_CHILDREN": 1,
  "CNT_FAM_MEMBERS": 3,
  "FLAG_OWN_CAR": "Yes",
  "FLAG_OWN_REALTY": "Yes",
  "NAME_INCOME_TYPE": "Working",
  "NAME_EDUCATION_TYPE": "Academic degree",
  "NAME_FAMILY_STATUS": "Married",
  "OCCUPATION_TYPE": "Managers",
  "NAME_HOUSING_TYPE": "House / apartment",
  "previous_defaults": 0,
  "bureau_overdue_amount": 0,
  "pos_dpd_max": 0
}
```

## Sample Output JSON

### Eligible Application Response
```json
{
  "eligibility": "Eligible",
  "pred_prob": 0.15,
  "pred_label": 0,
  "recommended_interest_rate": 10.75,
  "repayment_schedule": {
    "term_months": 60,
    "monthly_installment": 8665.50,
    "total_payment": 519930.00,
    "total_interest": 119930.00
  },
  "decision_reasoning": [
    {
      "feature": "employment_years",
      "impact_pct_pts": -8.2,
      "description": "Strong employment history reduces default risk"
    },
    {
      "feature": "FLAG_OWN_REALTY_Y",
      "impact_pct_pts": -5.1,
      "description": "Real estate ownership indicates financial stability"
    },
    {
      "feature": "credit_to_income",
      "impact_pct_pts": 3.4,
      "description": "Moderate credit-to-income ratio"
    }
  ],
  "actionable_tips": [
    "Consider a shorter term to reduce total interest paid",
    "Your strong financial profile qualifies for our best rates",
    "Consider adding a co-signer to potentially reduce the interest rate further"
  ],
  "confidence": 0.85,
  "explainability": {
    "top_positive": [
      {
        "feature": "credit_to_income",
        "shap_value": 0.034
      },
      {
        "feature": "annuity_to_income",
        "shap_value": 0.021
      }
    ],
    "top_negative": [
      {
        "feature": "employment_years",
        "shap_value": -0.082
      },
      {
        "feature": "FLAG_OWN_REALTY_Y",
        "shap_value": -0.051
      }
    ],
    "success": true
  },
  "processing_time_ms": 245.67
}
```

### Manual Review Application Response
```json
{
  "eligibility": "Manual Review",
  "pred_prob": 0.42,
  "pred_label": 1,
  "recommended_interest_rate": 14.8,
  "repayment_schedule": {
    "term_months": 60,
    "monthly_installment": 11668.00,
    "total_payment": 700080.00,
    "total_interest": 200080.00
  },
  "decision_reasoning": [
    {
      "feature": "credit_to_income",
      "impact_pct_pts": 12.1,
      "description": "High credit-to-income ratio increases default risk"
    },
    {
      "feature": "employment_years",
      "impact_pct_pts": -3.4,
      "description": "Short employment history is a concern"
    },
    {
      "feature": "annuity_to_income",
      "impact_pct_pts": 4.2,
      "description": "High annuity-to-income ratio"
    }
  ],
  "actionable_tips": [
    "Lower requested loan to 350000 to reduce pred_prob to 0.28",
    "Add a down payment of 20% to reduce rate by up to 1.5%",
    "Provide 12 months of bank statements to improve underwriting confidence",
    "Consider a longer term to reduce monthly payment burden"
  ],
  "confidence": 0.58,
  "explainability": {
    "top_positive": [
      {
        "feature": "credit_to_income",
        "shap_value": 0.121
      },
      {
        "feature": "annuity_to_income",
        "shap_value": 0.042
      },
      {
        "feature": "CNT_CHILDREN",
        "shap_value": 0.028
      }
    ],
    "top_negative": [
      {
        "feature": "employment_years",
        "shap_value": -0.034
      },
      {
        "feature": "NAME_EDUCATION_TYPE_Higher education",
        "shap_value": -0.019
      }
    ],
    "success": true
  },
  "processing_time_ms": 267.34
}
```

### Unlikely Application Response
```json
{
  "eligibility": "Unlikely",
  "pred_prob": 0.78,
  "pred_label": 1,
  "recommended_interest_rate": 20.2,
  "repayment_schedule": {
    "term_months": 60,
    "monthly_installment": 15234.50,
    "total_payment": 914070.00,
    "total_interest": 414070.00
  },
  "decision_reasoning": [
    {
      "feature": "credit_to_income",
      "impact_pct_pts": 25.3,
      "description": "Extremely high credit-to-income ratio"
    },
    {
      "feature": "previous_defaults",
      "impact_pct_pts": 18.7,
      "description": "Previous default history significantly increases risk"
    },
    {
      "feature": "employment_years",
      "impact_pct_pts": 12.4,
      "description": "Very short employment history"
    }
  ],
  "actionable_tips": [
    "Consider a much smaller loan amount (under 100000)",
    "Build employment history for at least 2 years",
    "Address any outstanding debts before applying",
    "Consider a secured loan with collateral",
    "Work with a credit counselor to improve your financial profile"
  ],
  "confidence": 0.78,
  "explainability": {
    "top_positive": [
      {
        "feature": "credit_to_income",
        "shap_value": 0.253
      },
      {
        "feature": "previous_defaults",
        "shap_value": 0.187
      },
      {
        "feature": "employment_years",
        "shap_value": 0.124
      }
    ],
    "top_negative": [
      {
        "feature": "NAME_EDUCATION_TYPE_Higher education",
        "shap_value": -0.045
      }
    ],
    "success": true
  },
  "processing_time_ms": 289.12
}
```

## Scenario Analysis Examples

### Scenario Request
```json
{
  "base_data": {
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 500000,
    "age_years": 35,
    "employment_years": 3,
    "NAME_INCOME_TYPE": "Working",
    "FLAG_OWN_CAR": "Yes"
  },
  "scenario_overrides": {
    "AMT_CREDIT": 350000,
    "AMT_ANNUITY": 12000
  }
}
```

### Scenario Response
```json
{
  "scenario": {
    "AMT_CREDIT": 350000,
    "AMT_ANNUITY": 12000
  },
  "pred_prob": 0.28,
  "pred_label": 0,
  "recommended_interest_rate": 12.7,
  "eligibility": "Manual Review"
}
```

## Error Response Examples

### Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "AMT_INCOME_TOTAL"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt",
      "ctx": {"limit_value": 0}
    },
    {
      "loc": ["body", "age_years"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error.number.not_ge",
      "ctx": {"limit_value": 18}
    }
  ]
}
```

### Model Not Loaded Error
```json
{
  "detail": "Model not loaded"
}
```

### Prediction Error
```json
{
  "detail": "Prediction failed: Feature engineering error"
}
```

## Health Check Response
```json
{
  "status": "healthy",
  "timestamp": 1703123456.789,
  "model_loaded": true,
  "version": "1.0.0"
}
```

## Configuration Response
```json
{
  "eligibility_thresholds": {
    "eligible": 0.20,
    "manual_review": 0.50
  },
  "base_interest_rate": 8.5,
  "risk_premium_multiplier": 15.0,
  "max_loan_to_income_ratio": 10.0
}
```

## Metrics Response
```json
{
  "model_loaded": true,
  "uptime": 86400.5,
  "version": "1.0.0",
  "total_requests": 1250,
  "avg_response_time_ms": 245.67,
  "error_rate": 0.02
}
```

## Field Validation Rules

### Required Fields
- `AMT_INCOME_TOTAL`: Must be > 0
- `AMT_CREDIT`: Must be > 0
- `age_years`: Must be between 18 and 80
- `employment_years`: Must be between 0 and 50

### Optional Fields with Validation
- `AMT_ANNUITY`: Must be >= 0
- `AMT_GOODS_PRICE`: Must be >= 0
- `CNT_CHILDREN`: Must be between 0 and 10
- `CNT_FAM_MEMBERS`: Must be between 1 and 10

### Categorical Fields (must match exact values)
- `FLAG_OWN_CAR`: "Yes" or "No"
- `FLAG_OWN_REALTY`: "Yes" or "No"
- `NAME_INCOME_TYPE`: "Businessman", "Commercial associate", "Maternity leave", "Pensioner", "State servant", "Student", "Unemployed", "Working"
- `NAME_EDUCATION_TYPE`: "Academic degree", "Higher education", "Incomplete higher", "Lower secondary", "Secondary / secondary special"
- `NAME_FAMILY_STATUS`: "Civil marriage", "Married", "Separated", "Single / not married", "Unknown", "Widow"
- `OCCUPATION_TYPE`: "Accountants", "Cleaning staff", "Cooking staff", "Core staff", "Drivers", "HR staff", "High skill tech staff", "IT staff", "Laborers", "Low-skill Laborers", "Managers", "Medicine staff", "Private service staff", "Realty agents", "Sales staff", "Secretaries", "Security staff", "Waiters/barmen staff"
- `NAME_HOUSING_TYPE`: "Co-op apartment", "House / apartment", "Municipal apartment", "Office apartment", "Rented apartment", "With parents"

### Business Rule Validation
- `AMT_CREDIT` cannot exceed `AMT_INCOME_TOTAL * 10` (configurable)
- If loan amount exceeds income ratio, automatic "Unlikely" response
