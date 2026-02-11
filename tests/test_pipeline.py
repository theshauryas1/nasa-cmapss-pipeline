"""
Unit Tests for NASA C-MAPSS Pipeline

This module contains tests to verify the data pipeline components work correctly.

Run with: pytest tests/ -v
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDataIngestion:
    """Tests for data ingestion module."""
    
    def test_column_names_length(self):
        """Verify correct number of columns."""
        from src.data.ingestion import COLUMN_NAMES
        assert len(COLUMN_NAMES) == 26  # engine_id + cycle + 3 settings + 21 sensors
    
    def test_sensor_columns(self):
        """Verify sensor column names."""
        from src.data.ingestion import get_sensor_columns
        sensors = get_sensor_columns()
        assert len(sensors) == 21
        assert sensors[0] == 'sensor_1'
        assert sensors[-1] == 'sensor_21'
    
    def test_feature_columns(self):
        """Verify feature column names."""
        from src.data.ingestion import get_feature_columns
        features = get_feature_columns()
        assert len(features) == 24  # 3 settings + 21 sensors


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'engine_id': np.repeat([1, 2], 50),
            'cycle': np.tile(np.arange(1, 51), 2),
            'op_setting_1': np.random.randn(n_samples),
            'op_setting_2': np.random.randn(n_samples),
            'op_setting_3': np.random.randn(n_samples),
        }
        
        # Add sensor columns
        for i in range(1, 22):
            if i in [1, 5, 6]:  # Constant sensors
                data[f'sensor_{i}'] = np.ones(n_samples) * 0.5
            else:
                data[f'sensor_{i}'] = np.random.randn(n_samples) + np.linspace(0, 1, n_samples)
        
        return pd.DataFrame(data)
    
    def test_preprocessor_fit(self, sample_data):
        """Test preprocessor fitting."""
        from src.data.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(normalization='minmax')
        preprocessor.fit(sample_data)
        
        assert preprocessor.is_fitted_
        assert preprocessor.norm_params_ is not None
    
    def test_normalization_range(self, sample_data):
        """Verify normalized values are in [0, 1]."""
        from src.data.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(normalization='minmax', rolling_windows=[])
        result = preprocessor.fit_transform(sample_data, add_features=False)
        
        sensor_cols = [c for c in result.columns if c.startswith('sensor')]
        for col in sensor_cols:
            assert result[col].min() >= -0.01  # Small tolerance
            assert result[col].max() <= 1.01
    
    def test_constant_sensor_detection(self, sample_data):
        """Test that constant sensors are detected."""
        from src.data.preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(drop_constant_sensors=True)
        preprocessor.fit(sample_data)
        
        # Sensors 1, 5, 6 should be detected as constant
        assert 'sensor_1' in preprocessor.constant_sensors_
        assert 'sensor_5' in preprocessor.constant_sensors_
        assert 'sensor_6' in preprocessor.constant_sensors_


class TestRULComputation:
    """Tests for RUL computation."""
    
    def test_training_rul_computation(self):
        """Test RUL computation for training data."""
        from src.data.ingestion import compute_training_rul
        
        # Create simple test case
        df = pd.DataFrame({
            'engine_id': [1, 1, 1, 2, 2],
            'cycle': [1, 2, 3, 1, 2]
        })
        
        result = compute_training_rul(df, cap_rul=125)
        
        # Engine 1: cycles 1, 2, 3 -> RUL should be 2, 1, 0
        engine1 = result[result['engine_id'] == 1]['RUL'].values
        assert list(engine1) == [2, 1, 0]
        
        # Engine 2: cycles 1, 2 -> RUL should be 1, 0
        engine2 = result[result['engine_id'] == 2]['RUL'].values
        assert list(engine2) == [1, 0]
    
    def test_rul_capping(self):
        """Test that RUL is capped correctly."""
        from src.data.ingestion import compute_training_rul
        
        df = pd.DataFrame({
            'engine_id': [1] * 200,
            'cycle': list(range(1, 201))
        })
        
        result = compute_training_rul(df, cap_rul=125)
        
        # Maximum RUL should be 125 (capped)
        assert result['RUL'].max() == 125
        # At cycle 1, RUL should be 125 (capped from 199)
        assert result[result['cycle'] == 1]['RUL'].values[0] == 125


class TestEvaluation:
    """Tests for evaluation metrics."""
    
    def test_rmse_computation(self):
        """Test RMSE calculation."""
        from src.models.evaluation import compute_rmse
        
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        assert compute_rmse(y_true, y_pred) == 0.0
    
    def test_nasa_score(self):
        """Test NASA scoring function."""
        from src.models.evaluation import compute_nasa_score
        
        # Perfect predictions
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        assert compute_nasa_score(y_true, y_pred) == 0.0
        
        # Early prediction (less penalty)
        y_true = np.array([10])
        y_pred = np.array([5])  # 5 cycles early
        early_score = compute_nasa_score(y_true, y_pred)
        
        # Late prediction (more penalty)
        y_pred = np.array([15])  # 5 cycles late
        late_score = compute_nasa_score(y_true, y_pred)
        
        # Late should have higher penalty
        assert late_score > early_score


class TestLinearModel:
    """Tests for linear baseline model."""
    
    def test_linear_model_fit_predict(self):
        """Test linear model fitting and prediction."""
        from src.models.baseline import LinearRULPredictor
        
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.sum(X, axis=1) + np.random.randn(100) * 0.1
        
        model = LinearRULPredictor(regularization='ridge', alpha=1.0)
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert model.is_fitted_
        assert len(predictions) == 100
        assert predictions.min() >= 0  # RUL should be non-negative
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        from src.models.baseline import LinearRULPredictor
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + 0.5 * X[:, 1]  # First feature is 4x more important
        
        model = LinearRULPredictor(regularization='none')
        model.fit(X, y, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])
        
        importance = model.get_feature_importance()
        
        # First feature should have highest importance
        assert importance.iloc[0]['feature'] == 'f1'


class TestStatisticalAnalysis:
    """Tests for statistical analysis functions."""
    
    @pytest.fixture
    def sample_data_with_rul(self):
        """Create sample data with RUL for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'engine_id': np.repeat([1, 2], 50),
            'cycle': np.tile(np.arange(1, 51), 2),
            'RUL': np.tile(np.arange(49, -1, -1), 2),
        }
        
        # Create correlated sensor
        data['sensor_1'] = -data['RUL'] + np.random.randn(n_samples) * 5
        data['sensor_2'] = np.random.randn(n_samples)  # Uncorrelated
        
        return pd.DataFrame(data)
    
    def test_sensor_rul_correlation(self, sample_data_with_rul):
        """Test correlation computation."""
        from src.analysis.statistical import compute_sensor_rul_correlation
        
        # Mock sensor columns function for this test
        import src.data.ingestion as ingestion
        original_func = ingestion.get_sensor_columns
        ingestion.get_sensor_columns = lambda: ['sensor_1', 'sensor_2']
        
        try:
            result = compute_sensor_rul_correlation(sample_data_with_rul, method='pearson')
            
            # sensor_1 should have higher correlation (it's designed to correlate)
            sensor1_corr = result[result['sensor'] == 'sensor_1']['pearson_corr'].abs().values[0]
            sensor2_corr = result[result['sensor'] == 'sensor_2']['pearson_corr'].abs().values[0]
            
            assert sensor1_corr > sensor2_corr
        finally:
            ingestion.get_sensor_columns = original_func


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
