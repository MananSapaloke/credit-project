"""
Integration tests for FastAPI endpoints.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app, get_scorer


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_scorer(self):
        """Create mock scorer."""
        scorer = Mock()
        scorer.predict.return_value = {
            'success': True,
            'pred_prob': 0.35,
            'pred_label': 0
        }
        scorer.get_shap_explanation.return_value = {
            'success': True,
            'top_positive': [{'feature': 'credit_to_income', 'shap_value': 0.1}],
            'top_negative': [{'feature': 'employment_years', 'shap_value': -0.05}]
        }
        return scorer
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "version" in data
    
    def test_score_endpoint_success(self, client, mock_scorer):
        """Test successful scoring request."""
        with patch('api.main.get_scorer', return_value=mock_scorer):
            payload = {
                "AMT_INCOME_TOTAL": 150000,
                "AMT_CREDIT": 300000,
                "age_years": 35,
                "employment_years": 5,
                "NAME_INCOME_TYPE": "Working",
                "FLAG_OWN_CAR": "Yes"
            }
            
            response = client.post("/api/v1/score", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert "eligibility" in data
            assert "pred_prob" in data
            assert "pred_label" in data
            assert "recommended_interest_rate" in data
            assert "repayment_schedule" in data
            assert "decision_reasoning" in data
            assert "actionable_tips" in data
            assert "explainability" in data
            assert "processing_time_ms" in data
    
    def test_score_endpoint_validation_error(self, client):
        """Test scoring request with validation errors."""
        payload = {
            "AMT_INCOME_TOTAL": -1000,  # Invalid negative income
            "AMT_CREDIT": 300000
        }
        
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_score_endpoint_high_loan_ratio(self, client, mock_scorer):
        """Test scoring request with excessive loan-to-income ratio."""
        with patch('api.main.get_scorer', return_value=mock_scorer):
            payload = {
                "AMT_INCOME_TOTAL": 50000,
                "AMT_CREDIT": 600000,  # 12x income (exceeds 10x limit)
                "age_years": 35,
                "employment_years": 5
            }
            
            response = client.post("/api/v1/score", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert data["eligibility"] == "Unlikely"
            assert data["pred_prob"] == 0.95
    
    def test_scenario_endpoint(self, client, mock_scorer):
        """Test scenario analysis endpoint."""
        with patch('api.main.get_scorer', return_value=mock_scorer):
            payload = {
                "base_data": {
                    "AMT_INCOME_TOTAL": 150000,
                    "AMT_CREDIT": 300000,
                    "age_years": 35,
                    "employment_years": 5
                },
                "scenario_overrides": {
                    "AMT_CREDIT": 250000
                }
            }
            
            response = client.post("/api/v1/scenario", json=payload)
            assert response.status_code == 200
            
            data = response.json()
            assert "scenario" in data
            assert "pred_prob" in data
            assert "pred_label" in data
            assert "recommended_interest_rate" in data
            assert "eligibility" in data
    
    def test_config_endpoints(self, client):
        """Test configuration endpoints."""
        # Get config
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        
        # Update config
        new_config = {"base_interest_rate": 9.0}
        response = client.put("/api/v1/config", json=new_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["config"]["base_interest_rate"] == 9.0
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_loaded" in data
        assert "uptime" in data
        assert "version" in data


class TestInputValidation:
    """Test input validation logic."""
    
    def test_age_validation(self):
        """Test age field validation."""
        client = TestClient(app)
        
        # Valid age
        payload = {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000,
            "age_years": 35
        }
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code in [200, 503]  # 503 if model not loaded
        
        # Invalid age (too young)
        payload["age_years"] = 15
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 422
        
        # Invalid age (too old)
        payload["age_years"] = 85
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 422
    
    def test_income_validation(self):
        """Test income field validation."""
        client = TestClient(app)
        
        # Valid income
        payload = {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000
        }
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code in [200, 503]
        
        # Invalid income (negative)
        payload["AMT_INCOME_TOTAL"] = -1000
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 422
        
        # Invalid income (zero)
        payload["AMT_INCOME_TOTAL"] = 0
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 422
    
    def test_categorical_validation(self):
        """Test categorical field validation."""
        client = TestClient(app)
        
        # Valid categorical values
        payload = {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 200000,
            "NAME_INCOME_TYPE": "Working",
            "FLAG_OWN_CAR": "Yes"
        }
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code in [200, 503]
        
        # Invalid categorical value
        payload["NAME_INCOME_TYPE"] = "InvalidType"
        response = client.post("/api/v1/score", json=payload)
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_model_not_loaded(self, client):
        """Test behavior when model is not loaded."""
        with patch('api.main.get_scorer', side_effect=Exception("Model not loaded")):
            payload = {
                "AMT_INCOME_TOTAL": 100000,
                "AMT_CREDIT": 200000
            }
            
            response = client.post("/api/v1/score", json=payload)
            assert response.status_code == 503
    
    def test_prediction_failure(self, client):
        """Test handling of prediction failures."""
        mock_scorer = Mock()
        mock_scorer.predict.return_value = {
            'success': False,
            'error': 'Prediction failed'
        }
        
        with patch('api.main.get_scorer', return_value=mock_scorer):
            payload = {
                "AMT_INCOME_TOTAL": 100000,
                "AMT_CREDIT": 200000
            }
            
            response = client.post("/api/v1/score", json=payload)
            assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__])
