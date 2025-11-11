"""
Tests for time-based feature engineering.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.time_features import TimeFeatureEngineer


class TestTimeFeatureEngineer:
    """Test suite for TimeFeatureEngineer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', '2023-01-07', freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'market': ['NYC'] * len(dates),
            'subscriber_count': np.random.randint(1000, 2000, len(dates))
        })
        return df
    
    def test_add_time_components(self, sample_data):
        """Test time component extraction."""
        engineer = TimeFeatureEngineer()
        df = engineer.add_time_components(sample_data)
        
        # Check new columns exist
        assert 'hour' in df.columns
        assert 'day_of_week' in df.columns
        assert 'month' in df.columns
        assert 'year' in df.columns
        assert 'is_weekend' in df.columns
        
        # Check value ranges
        assert (df['hour'] >= 0).all() and (df['hour'] <= 23).all()
        assert (df['day_of_week'] >= 0).all() and (df['day_of_week'] <= 6).all()
        assert (df['month'] >= 1).all() and (df['month'] <= 12).all()
        
        # Check cyclical encoding
        assert 'hour_sin' in df.columns
        assert 'hour_cos' in df.columns
        assert (df['hour_sin'] >= -1).all() and (df['hour_sin'] <= 1).all()
    
    def test_add_holiday_flags(self, sample_data):
        """Test holiday flag generation."""
        engineer = TimeFeatureEngineer(country='US')
        df = engineer.add_holiday_flags(sample_data)
        
        # Check new columns exist
        assert 'is_holiday' in df.columns
        assert 'days_to_holiday' in df.columns
        assert 'days_from_holiday' in df.columns
        
        # Check binary values
        assert df['is_holiday'].isin([0, 1]).all()
    
    def test_add_rolling_features(self, sample_data):
        """Test rolling feature generation."""
        engineer = TimeFeatureEngineer()
        df = engineer.add_rolling_features(
            sample_data,
            target_col='subscriber_count',
            windows=[3, 6]
        )
        
        # Check new columns exist
        assert 'subscriber_count_rolling_mean_3h' in df.columns
        assert 'subscriber_count_rolling_std_3h' in df.columns
        assert 'subscriber_count_rolling_mean_6h' in df.columns
        
        # Check no NaN in rolling features (should use min_periods=1)
        assert not df['subscriber_count_rolling_mean_3h'].isnull().any()
    
    def test_add_lag_features(self, sample_data):
        """Test lag feature generation."""
        engineer = TimeFeatureEngineer()
        df = engineer.add_lag_features(
            sample_data,
            target_col='subscriber_count',
            lags=[1, 6]
        )
        
        # Check new columns exist
        assert 'subscriber_count_lag_1h' in df.columns
        assert 'subscriber_count_lag_6h' in df.columns
    
    def test_create_all_time_features(self, sample_data):
        """Test complete time feature pipeline."""
        engineer = TimeFeatureEngineer()
        df = engineer.create_all_time_features(
            sample_data,
            target_col='subscriber_count',
            rolling_windows=[3, 6],
            lags=[1, 6]
        )
        
        # Check that multiple feature types exist
        assert 'hour' in df.columns
        assert 'is_holiday' in df.columns
        assert 'subscriber_count_rolling_mean_3h' in df.columns
        assert 'subscriber_count_lag_1h' in df.columns
        
        # Check original columns preserved
        assert 'timestamp' in df.columns
        assert 'market' in df.columns
        assert 'subscriber_count' in df.columns
