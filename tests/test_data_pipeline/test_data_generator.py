"""
Tests for synthetic data generation.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_pipeline.data_generator import SyntheticDataGenerator


class TestSyntheticDataGenerator:
    """Test suite for SyntheticDataGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = SyntheticDataGenerator(
            start_date='2023-01-01',
            end_date='2023-01-31',
            markets=['NYC', 'LA']
        )
        assert generator.start_date == pd.to_datetime('2023-01-01')
        assert generator.end_date == pd.to_datetime('2023-01-31')
        assert generator.markets == ['NYC', 'LA']
    
    def test_generate_subscriber_data(self):
        """Test subscriber data generation."""
        generator = SyntheticDataGenerator(
            start_date='2023-01-01',
            end_date='2023-01-02',
            markets=['NYC']
        )
        
        df = generator.generate_subscriber_data()
        
        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert 'timestamp' in df.columns
        assert 'market' in df.columns
        assert 'subscriber_count' in df.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        assert df['market'].dtype == object
        assert pd.api.types.is_numeric_dtype(df['subscriber_count'])
        
        # Check values
        assert len(df) > 0
        assert (df['subscriber_count'] >= 0).all()
        assert df['market'].unique().tolist() == ['NYC']
    
    def test_generate_weather_data(self):
        """Test weather data generation."""
        generator = SyntheticDataGenerator(
            start_date='2023-01-01',
            end_date='2023-01-02',
            markets=['LA']
        )
        
        df = generator.generate_weather_data()
        
        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert 'timestamp' in df.columns
        assert 'market' in df.columns
        assert 'temperature' in df.columns
        assert 'precipitation' in df.columns
        assert 'humidity' in df.columns
        assert 'wind_speed' in df.columns
        assert 'severe_weather' in df.columns
        
        # Check value ranges
        assert (df['humidity'] >= 0).all()
        assert (df['humidity'] <= 100).all()
        assert (df['precipitation'] >= 0).all()
        assert (df['wind_speed'] >= 0).all()
        assert df['severe_weather'].isin([0, 1]).all()
    
    def test_generate_event_data(self):
        """Test event data generation."""
        generator = SyntheticDataGenerator(
            start_date='2023-01-01',
            end_date='2023-01-02',
            markets=['CHI']
        )
        
        df = generator.generate_event_data()
        
        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert 'timestamp' in df.columns
        assert 'market' in df.columns
        assert 'outage' in df.columns
        assert 'campaign' in df.columns
        
        # Check binary values
        assert df['outage'].isin([0, 1]).all()
        assert df['campaign'].isin([0, 1]).all()
    
    def test_reproducibility(self):
        """Test that same seed produces same data."""
        generator1 = SyntheticDataGenerator(
            start_date='2023-01-01',
            end_date='2023-01-02'
        )
        generator2 = SyntheticDataGenerator(
            start_date='2023-01-01',
            end_date='2023-01-02'
        )
        
        df1 = generator1.generate_subscriber_data(seed=42)
        df2 = generator2.generate_subscriber_data(seed=42)
        
        pd.testing.assert_frame_equal(df1, df2)
