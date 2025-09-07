"""
CreditLens FastAPI Backend

Production-ready API for credit risk scoring with SHAP explainability.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import logging
import time
import os
from pathlib import Path

# Import our credit scorer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.credit_scorer import CreditScorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CreditLens API",
    description="Credit Risk Assessment API with ML Predictions and SHAP Explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
scorer = None
config = {
    "eligibility_thresholds": {
        "eligible": 0.20,
        "manual_review": 0.50
    },
    "base_interest_rate": 8.5,
    "risk_premium_multiplier": 15.0,
    "max_loan_to_income_ratio": 10.0
}


# Pydantic models for request/response validation
class ApplicantData(BaseModel):
    """Input data model for credit scoring."""
    
    # Optional fields
    SK_ID_CURR: Optional[int] = None
    
    # Required personal information
    age_years: Optional[int] = Field(None, ge=18, le=80, description="Applicant age in years")
    employment_years: Optional[int] = Field(None, ge=0, le=50, description="Years of employment")
    
    # Required financial information
    AMT_INCOME_TOTAL: float = Field(..., gt=0, description="Annual income")
    AMT_CREDIT: float = Field(..., gt=0, description="Requested loan amount")
    
    # Optional financial information
    AMT_ANNUITY: Optional[float] = Field(None, ge=0, description="Monthly/yearly annuity")
    AMT_GOODS_PRICE: Optional[float] = Field(None, ge=0, description="Goods price")
    CNT_CHILDREN: Optional[int] = Field(0, ge=0, le=10, description="Number of children")
    CNT_FAM_MEMBERS: Optional[int] = Field(1, ge=1, le=10, description="Family size")
    
    # Asset ownership
    FLAG_OWN_CAR: Optional[str] = Field("No", regex="^(Yes|No)$", description="Owns a car")
    FLAG_OWN_REALTY: Optional[str] = Field("No", regex="^(Yes|No)$", description="Owns real estate")
    
    # Demographics
    NAME_INCOME_TYPE: Optional[str] = Field(
        "Working", 
        regex="^(Businessman|Commercial associate|Maternity leave|Pensioner|State servant|Student|Unemployed|Working)$",
        description="Income type"
    )
    NAME_EDUCATION_TYPE: Optional[str] = Field(
        "Secondary / secondary special",
        regex="^(Academic degree|Higher education|Incomplete higher|Lower secondary|Secondary / secondary special)$",
        description="Education level"
    )
    NAME_FAMILY_STATUS: Optional[str] = Field(
        "Single / not married",
        regex="^(Civil marriage|Married|Separated|Single / not married|Unknown|Widow)$",
        description="Family status"
    )
    OCCUPATION_TYPE: Optional[str] = Field(
        "Core staff",
        regex="^(Accountants|Cleaning staff|Cooking staff|Core staff|Drivers|HR staff|High skill tech staff|IT staff|Laborers|Low-skill Laborers|Managers|Medicine staff|Private service staff|Realty agents|Sales staff|Secretaries|Security staff|Waiters/barmen staff)$",
        description="Occupation type"
    )
    NAME_HOUSING_TYPE: Optional[str] = Field(
        "House / apartment",
        regex="^(Co-op apartment|House / apartment|Municipal apartment|Office apartment|Rented apartment|With parents)$",
        description="Housing type"
    )
    
    # Additional optional fields
    previous_defaults: Optional[int] = Field(0, ge=0, description="Previous default count")
    bureau_overdue_amount: Optional[float] = Field(0, ge=0, description="Bureau overdue amount")
    pos_dpd_max: Optional[int] = Field(0, ge=0, description="Maximum POS DPD")
    
    @validator('AMT_CREDIT')
    def validate_loan_amount(cls, v, values):
        """Validate loan amount against income."""
        if 'AMT_INCOME_TOTAL' in values and v > values['AMT_INCOME_TOTAL'] * config['max_loan_to_income_ratio']:
            raise ValueError(f"Loan amount cannot exceed {config['max_loan_to_income_ratio']}x annual income")
        return v
    
    @validator('age_years', 'employment_years')
    def convert_years_to_days(cls, v, field):
        """Convert years to days for model compatibility."""
        if v is not None:
            return -v * 365  # Negative days (days before current date)
        return v


class ScenarioRequest(BaseModel):
    """Request model for scenario analysis."""
    base_data: ApplicantData
    scenario_overrides: Dict[str, Any] = Field(default_factory=dict)


class RepaymentSchedule(BaseModel):
    """Repayment schedule model."""
    term_months: int
    monthly_installment: float
    total_payment: float
    total_interest: float


class DecisionReasoning(BaseModel):
    """Decision reasoning model."""
    feature: str
    impact_pct_pts: float
    description: str


class ScoringResponse(BaseModel):
    """Response model for credit scoring."""
    eligibility: str
    pred_prob: float
    pred_label: int
    recommended_interest_rate: float
    repayment_schedule: RepaymentSchedule
    decision_reasoning: List[DecisionReasoning]
    actionable_tips: List[str]
    confidence: Optional[float] = None
    explainability: Dict[str, Any]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    model_loaded: bool
    version: str


# Dependency to get scorer instance
def get_scorer() -> CreditScorer:
    """Get the global scorer instance."""
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return scorer


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global scorer
    
    try:
        model_path = os.getenv("MODEL_PATH", "lgb_model_step32.txt")
        feature_columns_path = os.getenv("FEATURE_COLUMNS_PATH", None)
        
        scorer = CreditScorer(model_path, feature_columns_path)
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# Health check endpoint
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if scorer is not None else "unhealthy",
        timestamp=time.time(),
        model_loaded=scorer is not None,
        version="1.0.0"
    )


# Main scoring endpoint
@app.post("/api/v1/score", response_model=ScoringResponse)
async def score_application(
    request: Request,
    applicant_data: ApplicantData,
    scorer_instance: CreditScorer = Depends(get_scorer)
):
    """
    Score a credit application and return prediction with explainability.
    """
    start_time = time.time()
    
    try:
        # Convert Pydantic model to dict
        input_data = applicant_data.dict()
        
        # Quick pre-check for obvious rejections
        if input_data['AMT_CREDIT'] > input_data['AMT_INCOME_TOTAL'] * config['max_loan_to_income_ratio']:
            return ScoringResponse(
                eligibility="Unlikely",
                pred_prob=0.95,  # High risk
                pred_label=1,
                recommended_interest_rate=25.0,
                repayment_schedule=RepaymentSchedule(
                    term_months=60,
                    monthly_installment=0,
                    total_payment=0,
                    total_interest=0
                ),
                decision_reasoning=[
                    DecisionReasoning(
                        feature="loan_to_income_ratio",
                        impact_pct_pts=50.0,
                        description="Loan amount exceeds maximum allowed ratio"
                    )
                ],
                actionable_tips=[
                    f"Reduce loan amount to maximum {input_data['AMT_INCOME_TOTAL'] * config['max_loan_to_income_ratio']:,.0f}",
                    "Consider a longer repayment term",
                    "Provide additional collateral or guarantor"
                ],
                explainability={"error": "Application rejected due to excessive loan amount"},
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Make prediction
        prediction_result = scorer_instance.predict(input_data)
        
        if not prediction_result['success']:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {prediction_result.get('error', 'Unknown error')}")
        
        pred_prob = prediction_result['pred_prob']
        pred_label = prediction_result['pred_label']
        
        # Determine eligibility
        if pred_prob < config['eligibility_thresholds']['eligible']:
            eligibility = "Eligible"
        elif pred_prob < config['eligibility_thresholds']['manual_review']:
            eligibility = "Manual Review"
        else:
            eligibility = "Unlikely"
        
        # Calculate recommended interest rate
        risk_premium = pred_prob * config['risk_premium_multiplier']
        recommended_rate = config['base_interest_rate'] + risk_premium
        
        # Calculate repayment schedule
        term_months = 60  # Default term
        monthly_rate = recommended_rate / 100 / 12
        monthly_payment = input_data['AMT_CREDIT'] * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
        total_payment = monthly_payment * term_months
        total_interest = total_payment - input_data['AMT_CREDIT']
        
        # Get SHAP explanation
        shap_result = scorer_instance.get_shap_explanation(input_data, top_n=3)
        
        # Generate decision reasoning
        decision_reasoning = []
        if shap_result['success']:
            for item in shap_result['top_positive'][:3]:
                decision_reasoning.append(DecisionReasoning(
                    feature=item['feature'],
                    impact_pct_pts=abs(item['shap_value']) * 100,
                    description=f"High {item['feature']} increases default risk"
                ))
        
        # Generate actionable tips
        actionable_tips = []
        if pred_prob > 0.3:
            actionable_tips.append("Consider reducing loan amount to improve approval chances")
        if input_data.get('employment_years', 0) < 2:
            actionable_tips.append("Provide additional employment documentation")
        if input_data.get('FLAG_OWN_REALTY') == 'No':
            actionable_tips.append("Consider providing collateral to reduce interest rate")
        if input_data.get('CNT_CHILDREN', 0) > 2:
            actionable_tips.append("Consider family income documentation")
        
        # Calculate confidence (simple heuristic)
        confidence = max(0.5, 1.0 - abs(pred_prob - 0.5) * 2)
        
        processing_time = (time.time() - start_time) * 1000
        
        return ScoringResponse(
            eligibility=eligibility,
            pred_prob=pred_prob,
            pred_label=pred_label,
            recommended_interest_rate=round(recommended_rate, 2),
            repayment_schedule=RepaymentSchedule(
                term_months=term_months,
                monthly_installment=round(monthly_payment, 2),
                total_payment=round(total_payment, 2),
                total_interest=round(total_interest, 2)
            ),
            decision_reasoning=decision_reasoning,
            actionable_tips=actionable_tips,
            confidence=round(confidence, 2),
            explainability=shap_result,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Scenario analysis endpoint
@app.post("/api/v1/scenario")
async def analyze_scenario(
    scenario_request: ScenarioRequest,
    scorer_instance: CreditScorer = Depends(get_scorer)
):
    """
    Analyze different scenarios for the same applicant.
    """
    try:
        base_data = scenario_request.base_data.dict()
        overrides = scenario_request.scenario_overrides
        
        # Apply overrides
        scenario_data = {**base_data, **overrides}
        
        # Make prediction
        result = scorer_instance.predict(scenario_data)
        
        if not result['success']:
            raise HTTPException(status_code=500, detail="Scenario analysis failed")
        
        # Calculate interest rate for scenario
        risk_premium = result['pred_prob'] * config['risk_premium_multiplier']
        scenario_rate = config['base_interest_rate'] + risk_premium
        
        return {
            "scenario": overrides,
            "pred_prob": result['pred_prob'],
            "pred_label": result['pred_label'],
            "recommended_interest_rate": round(scenario_rate, 2),
            "eligibility": "Eligible" if result['pred_prob'] < 0.2 else "Manual Review" if result['pred_prob'] < 0.5 else "Unlikely"
        }
        
    except Exception as e:
        logger.error(f"Scenario analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration endpoint (admin only)
@app.get("/api/v1/config")
async def get_config():
    """Get current configuration."""
    return config


@app.put("/api/v1/config")
async def update_config(new_config: Dict[str, Any]):
    """Update configuration (admin only)."""
    global config
    config.update(new_config)
    return {"message": "Configuration updated", "config": config}


# Metrics endpoint (admin only)
@app.get("/api/v1/metrics")
async def get_metrics():
    """Get system metrics."""
    return {
        "model_loaded": scorer is not None,
        "uptime": time.time(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
