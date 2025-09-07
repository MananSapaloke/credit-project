"""
Unit tests for CreditScorer module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.credit_scorer import CreditScorer, create_sample_input


class TestCreditScorer:
    """Test cases for CreditScorer class."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock LightGBM model."""
        model = Mock()
        model.predict.return_value = np.array([0.35])
        return model
    
    @pytest.fixture
    def scorer(self, mock_model, tmp_path):
        """Create a CreditScorer instance with mock model."""
        model_path = tmp_path / "test_model.txt"
        model_path.write_text("mock model")
        
        with patch('lightgbm.Booster') as mock_booster:
            mock_booster.return_value = mock_model
            scorer = CreditScorer(str(model_path))
            return scorer
    
    def test_clean_feature_names(self, scorer):
        """Test feature name cleaning."""
        dirty_names = ["feature with spaces", "feature-with-dashes", "feature.with.dots"]
        clean_names = scorer.clean_feature_names(dirty_names)
        expected = ["feature_with_spaces", "feature_with_dashes", "feature_with_dots"]
        assert clean_names == expected
    
    def test_preprocess_input_basic(self, scorer):
        """Test basic input preprocessing."""
        input_data = {
            'AMT_INCOME_TOTAL': 100000,
            'AMT_CREDIT': 200000,
            'AMT_ANNUITY': 5000,
            'CNT_CHILDREN': 1,
            'CNT_FAM_MEMBERS': 3,
            'FLAG_OWN_CAR': 'Yes',
            'NAME_INCOME_TYPE': 'Working'
        }
        
        result = scorer.preprocess_input(input_data)
        
        # Check that it returns a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        
        # Check that key features are created
        assert 'credit_to_income' in result.columns
        assert 'annuity_to_income' in result.columns
        assert 'NAME_INCOME_TYPE_Working' in result.columns
        assert 'FLAG_OWN_CAR_Y' in result.columns
        
        # Check feature values
        assert result['credit_to_income'].iloc[0] == 200000 / (100000 + 1)
        assert result['annuity_to_income'].iloc[0] == 5000 / (100000 + 1)
        assert result['NAME_INCOME_TYPE_Working'].iloc[0] == 1
        assert result['FLAG_OWN_CAR_Y'].iloc[0] == 1
    
    def test_preprocess_input_missing_values(self, scorer):
        """Test preprocessing with missing values."""
        input_data = {
            'AMT_INCOME_TOTAL': 100000,
            'AMT_CREDIT': 200000,
            # Missing AMT_ANNUITY, CNT_CHILDREN, etc.
        }
        
        result = scorer.preprocess_input(input_data)
        
        # Should not raise an error and should fill missing values
        assert isinstance(result, pd.DataFrame)
        assert not result.isnull().any().any()
    
    def test_predict_success(self, scorer):
        """Test successful prediction."""
        input_data = create_sample_input()
        
        result = scorer.predict(input_data)
        
        assert result['success'] is True
        assert 'pred_prob' in result
        assert 'pred_label' in result
        assert result['pred_prob'] == 0.35
        assert result['pred_label'] == 0  # 0.35 < 0.5
    
    def test_predict_failure(self, scorer):
        """Test prediction failure handling."""
        # Mock model to raise an exception
        scorer.model.predict.side_effect = Exception("Model error")
        
        input_data = create_sample_input()
        result = scorer.predict(input_data)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['error'] == "Model error"
    
    def test_get_shap_explanation_success(self, scorer):
        """Test successful SHAP explanation."""
        with patch('shap.TreeExplainer') as mock_explainer_class:
            mock_explainer = Mock()
            mock_explainer_class.return_value = mock_explainer
            mock_explainer.shap_values.return_value = np.array([[0.1, -0.05, 0.08, -0.02]])
            
            input_data = create_sample_input()
            result = scorer.get_shap_explanation(input_data)
            
            assert result['success'] is True
            assert 'top_positive' in result
            assert 'top_negative' in result
    
    def test_get_shap_explanation_failure(self, scorer):
        """Test SHAP explanation failure handling."""
        with patch('shap.TreeExplainer') as mock_explainer_class:
            mock_explainer_class.side_effect = ImportError("SHAP not available")
            
            input_data = create_sample_input()
            result = scorer.get_shap_explanation(input_data)
            
            assert result['success'] is False
            assert 'error' in result
    
    def test_create_sample_input(self):
        """Test sample input creation."""
        sample = create_sample_input()
        
        assert isinstance(sample, dict)
        assert 'AMT_INCOME_TOTAL' in sample
        assert 'AMT_CREDIT' in sample
        assert 'NAME_INCOME_TYPE' in sample
        assert sample['NAME_INCOME_TYPE'] == 'Working'


class TestFeatureEngineering:
    """Test feature engineering logic."""
    
    def test_credit_to_income_ratio(self):
        """Test credit to income ratio calculation."""
        input_data = {
            'AMT_INCOME_TOTAL': 100000,
            'AMT_CREDIT': 200000
        }
        
        scorer = CreditScorer("dummy_path")
        result = scorer.preprocess_input(input_data)
        
        expected_ratio = 200000 / (100000 + 1)
        assert abs(result['credit_to_income'].iloc[0] - expected_ratio) < 1e-6
    
    def test_annuity_to_income_ratio(self):
        """Test annuity to income ratio calculation."""
        input_data = {
            'AMT_INCOME_TOTAL': 120000,
            'AMT_ANNUITY': 6000
        }
        
        scorer = CreditScorer("dummy_path")
        result = scorer.preprocess_input(input_data)
        
        expected_ratio = 6000 / (120000 + 1)
        assert abs(result['annuity_to_income'].iloc[0] - expected_ratio) < 1e-6
    
    def test_categorical_encoding(self):
        """Test categorical variable encoding."""
        input_data = {
            'NAME_INCOME_TYPE': 'Working',
            'FLAG_OWN_CAR': 'Yes',
            'FLAG_OWN_REALTY': 'No'
        }
        
        scorer = CreditScorer("dummy_path")
        result = scorer.preprocess_input(input_data)
        
        # Check that correct categories are encoded as 1
        assert result['NAME_INCOME_TYPE_Working'].iloc[0] == 1
        assert result['FLAG_OWN_CAR_Y'].iloc[0] == 1
        assert result['FLAG_OWN_REALTY_N'].iloc[0] == 1
        
        # Check that other categories are encoded as 0
        assert result['NAME_INCOME_TYPE_Pensioner'].iloc[0] == 0
        assert result['FLAG_OWN_CAR_N'].iloc[0] == 0
        assert result['FLAG_OWN_REALTY_Y'].iloc[0] == 0


if __name__ == "__main__":
    pytest.main([__file__])
