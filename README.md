# CreditLens - Credit Risk Assessment System

## Overview

CreditLens is a production-ready credit risk assessment system that predicts loan default probability using machine learning and provides business-friendly insights through a Power BI dashboard and web application.

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  ML Pipeline     │───▶│  Power BI       │
│                 │    │                  │    │  Dashboard      │
│ • Bureau Data   │    │ • Feature Eng.   │    │                 │
│ • Applications  │    │ • LightGBM Model │    │ • 6 Pages       │
│ • Payments      │    │ • SHAP Analysis  │    │ • KPIs & Viz    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Web Application │
                       │                  │
                       │ • FastAPI Backend│
                       │ • React Frontend │
                       │ • Real-time Score│
                       └──────────────────┘
```

## Key Features

### Machine Learning Pipeline
- **Model**: LightGBM with 0.769 AUC validation score
- **Features**: 200+ engineered features including:
  - `credit_to_income`: Credit amount to income ratio
  - `goods_price_to_credit`: Goods price to credit ratio  
  - `annuity_to_income`: Annuity to income ratio
  - `days_employed_ratio`: Employment duration ratio
  - Bureau aggregates (AMT_*, DAYS_* statistics)
  - Previous application features
  - Payment history features
- **Explainability**: SHAP analysis for model interpretability
- **Hyperparameter Tuning**: Optuna optimization

### Data Processing
- **Chunked Processing**: Handles 307K+ records efficiently
- **Feature Engineering**: Automated feature creation and aggregation
- **Data Cleaning**: Defensive conversions with `pd.to_numeric(errors='coerce')`
- **Encoding**: One-hot encoding for categorical variables

### Business Intelligence
- **Power BI Dashboard**: 6 comprehensive pages
  1. Overview - High-level KPIs and model summary
  2. Customer Profile Deep Dive - Demographics and risk segments
  3. Financial Health Analysis - Credit ratios and financial metrics
  4. Employment & Bureau History - Employment and credit history
  5. Property & Housing Risk - Housing and property analysis
  6. Summary - Consolidated KPIs and default analysis

### Web Application (Planned)
- **Real-time Scoring**: Instant loan default probability prediction
- **Interactive Forms**: Multi-step application process
- **Explainability**: SHAP-based decision explanations
- **Business Rules**: Configurable eligibility thresholds
- **Recommendations**: Actionable tips and alternative products

## File Structure

```
CreditLens/
├── credit_upto_step_34.ipynb          # Main ML pipeline notebook
├── dashboard_data.csv                  # Final scoring data for Power BI
├── master_train_fe_encoded.parquet    # Encoded training features
├── lgb_model_step32.txt               # Trained LightGBM model
├── agg_*.parquet                      # Aggregated feature files
├── Page *.jpg                         # Power BI dashboard screenshots
├── src/                               # Production code (planned)
│   ├── api/                          # FastAPI backend
│   ├── models/                       # ML model and preprocessing
│   ├── frontend/                     # React web application
│   └── tests/                        # Test suite
├── docker/                           # Containerization
├── docs/                             # Documentation
└── README.md                         # This file
```

## Model Performance

- **Validation AUC**: 0.769
- **Training Data**: 307,512 records
- **Features**: 200+ engineered features
- **Model Type**: LightGBM with early stopping
- **Cross-validation**: 5-fold CV implemented

## Key Features Engineered

### Financial Ratios
- `credit_to_income`: AMT_CREDIT / AMT_INCOME_TOTAL
- `goods_price_to_credit`: AMT_GOODS_PRICE / AMT_CREDIT  
- `annuity_to_income`: AMT_ANNUITY / AMT_INCOME_TOTAL
- `payment_to_inst_ratio`: Payment to installment ratio

### Temporal Features
- `days_employed_ratio`: DAYS_EMPLOYED / DAYS_BIRTH
- `account_age_days`: Account age calculation
- `children_income_ratio`: CNT_CHILDREN / AMT_INCOME_TOTAL

### Bureau Aggregates
- `BUREAU_AMT_CREDIT_SUM_*`: Sum, mean, max statistics
- `BUREAU_DAYS_CREDIT_*`: Min, max, mean statistics
- `BUREAU_CREDIT_DAY_OVERDUE_MAX`: Maximum overdue days

### Previous Application Features
- `PREV_AMT_*`: Previous application amounts
- `PREV_DAYS_DECISION_*`: Decision timing statistics
- `PREV_CNT_PAYMENT_*`: Payment count statistics

## Power BI Dashboard Features

### DAX Measures
```dax
TotalDefaults = SUM('dashboard_data'[TARGET])
DefaultRate = DIVIDE(SUM('dashboard_data'[TARGET]), COUNTROWS('dashboard_data'), 0)
AvgPredProb = AVERAGE('dashboard_data'[pred_prob])
```

### Calculated Columns
```dax
AgeYears = ROUND(ABS(DAYS_BIRTH)/365,0)
EmploymentYears = INT(-DAYS_EMPLOYED/365.25)
RiskSegment = SWITCH(TRUE(), pred_prob < 0.33, "Low", pred_prob < 0.66, "Medium", "High")
```

## Data Quality & Validation

### Known Issues Resolved
- Column type mismatches (DAYS_* fields imported as text)
- Missing EXT_SOURCE_* features in final CSV
- Power BI DAX type coercion issues
- EARLIER function misuse in DAX

### Data Validation
- Numeric coercion with error handling
- Null value imputation strategies
- Feature name standardization
- Consistent encoding across datasets

## Production Deployment (Planned)

### Backend (FastAPI)
- `/api/v1/score` - Main scoring endpoint
- `/api/v1/scenario` - Alternative scenario analysis
- `/api/v1/health` - Service health check
- Input validation with Pydantic
- SHAP integration for explainability

### Frontend (React)
- Multi-step application form
- Real-time scoring results
- Interactive scenario simulator
- Downloadable PDF reports
- Mobile-responsive design

### Infrastructure
- Docker containerization
- Horizontal scaling support
- Rate limiting and security
- Monitoring and alerting
- Model versioning

## Model Retraining

### Automated Pipeline
1. Load new data from sources
2. Apply feature engineering pipeline
3. Retrain LightGBM with Optuna optimization
4. Validate model performance
5. Update production model
6. Regenerate dashboard data

### Monitoring
- Model drift detection
- Feature distribution monitoring
- Performance degradation alerts
- A/B testing framework

## Security & Compliance

### Data Protection
- TLS encryption for all communications
- PII data handling compliance
- Audit logging for all operations
- GDPR compliance features

### Access Control
- Role-based access control
- API authentication
- Rate limiting
- CAPTCHA protection

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Power BI Desktop
- Docker (for production deployment)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd CreditLens

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook credit_upto_step_34.ipynb
```

### Production Deployment
```bash
# Build and run with Docker
docker-compose up -d

# Access web application
open http://localhost:3000
```

## API Usage Examples

### Scoring Request
```json
POST /api/v1/score
{
  "age_years": 35,
  "employment_years": 3,
  "AMT_INCOME_TOTAL": 150000,
  "AMT_CREDIT": 500000,
  "FLAG_OWN_CAR": "Yes",
  "NAME_INCOME_TYPE": "Working"
}
```

### Scoring Response
```json
{
  "eligibility": "Manual Review",
  "pred_prob": 0.42,
  "recommended_interest_rate": 11.5,
  "decision_reasoning": [
    {"feature": "credit_to_income", "impact_pct_pts": 12.1}
  ],
  "actionable_tips": [
    "Lower requested loan to 350000 to reduce risk"
  ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Add your license information here]

## Support

For questions or issues, please contact [your-email@domain.com] or create an issue in the repository.

---

**Last Updated**: December 2024  
**Version**: 1.0.0  
**Status**: Production Ready (ML Pipeline), Development (Web App)
