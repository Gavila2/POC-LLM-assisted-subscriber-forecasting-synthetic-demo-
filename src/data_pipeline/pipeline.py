"""
Main pipeline orchestration for data processing.
Combines data generation, merging, cleaning, feature engineering, and normalization.
"""
import pandas as pd
import os
from .data_generator import SyntheticDataGenerator
from .data_merger import DataMerger
from .data_cleaner import DataCleaner
from ..features.time_features import TimeFeatureEngineer
from ..features.event_features import EventFeatureEngineer
from ..features.normalization import DataNormalizer


class DataPipeline:
    """Complete data processing pipeline for subscriber forecasting."""
    
    def __init__(self, start_date='2023-01-01', end_date='2023-12-31', 
                 markets=None, country='US'):
        """
        Initialize the data pipeline.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            markets: List of market identifiers
            country: Country code for holiday calendar
        """
        self.start_date = start_date
        self.end_date = end_date
        self.markets = markets or ['NYC', 'LA', 'CHI']
        self.country = country
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator(start_date, end_date, markets)
        self.data_merger = DataMerger()
        self.data_cleaner = DataCleaner()
        self.time_features = TimeFeatureEngineer(country)
        self.event_features = EventFeatureEngineer()
        self.normalizer = DataNormalizer(method='standard')
        
        self.processed_data = None
        
    def run_full_pipeline(self, raw_data_dir='data/raw', 
                         processed_data_dir='data/processed',
                         generate_new_data=True):
        """
        Run the complete data processing pipeline.
        
        Args:
            raw_data_dir: Directory for raw data
            processed_data_dir: Directory for processed data
            generate_new_data: Whether to generate new synthetic data
            
        Returns:
            Processed DataFrame ready for modeling
        """
        print("=" * 80)
        print("SUBSCRIBER FORECASTING DATA PIPELINE")
        print("=" * 80)
        
        # Ensure directories exist
        os.makedirs(raw_data_dir, exist_ok=True)
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Step 1: Generate or load raw data
        if generate_new_data:
            print("\n[1/7] Generating synthetic data...")
            self.data_generator.save_data(raw_data_dir)
        else:
            print("\n[1/7] Using existing raw data...")
        
        # Step 2: Merge datasets
        print("\n[2/7] Merging subscriber, weather, and event data...")
        merged_data = self.data_merger.load_and_merge(
            f'{raw_data_dir}/subscriber_data.csv',
            f'{raw_data_dir}/weather_data.csv',
            f'{raw_data_dir}/event_data.csv'
        )
        print(f"Merged data shape: {merged_data.shape}")
        
        # Save merged data
        self.data_merger.save_merged_data(f'{processed_data_dir}/merged_data.csv')
        
        # Step 3: Clean data
        print("\n[3/7] Cleaning data...")
        cleaned_data = self.data_cleaner.clean_pipeline(
            merged_data,
            smooth=True,
            interpolate=True,
            handle_outliers_flag=True,
            window_size=3
        )
        
        # Step 4: Create time features
        print("\n[4/7] Engineering time features...")
        time_featured_data = self.time_features.create_all_time_features(
            cleaned_data,
            target_col='subscriber_count',
            rolling_windows=[3, 6, 12, 24, 168],
            lags=[1, 6, 12, 24, 168]
        )
        
        # Step 5: Create event features
        print("\n[5/7] Engineering event features...")
        event_featured_data = self.event_features.create_all_event_features(
            time_featured_data
        )
        
        # Step 6: Normalize data by market
        print("\n[6/7] Normalizing data by market...")
        normalized_data = self.normalizer.fit_transform_by_market(
            event_featured_data,
            exclude_cols=['timestamp', 'market', 'subscriber_count']
        )
        
        # Save normalizer
        self.normalizer.save_scalers(f'{processed_data_dir}/scalers.pkl')
        
        # Step 7: Save final processed data
        print("\n[7/7] Saving processed data...")
        output_file = f'{processed_data_dir}/processed_data.csv'
        normalized_data.to_csv(output_file, index=False)
        print(f"Final processed data saved to {output_file}")
        print(f"Final shape: {normalized_data.shape}")
        
        self.processed_data = normalized_data
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Markets: {', '.join(self.markets)}")
        print(f"Total rows: {len(normalized_data):,}")
        print(f"Total features: {len(normalized_data.columns)}")
        print(f"Missing values: {normalized_data.isnull().sum().sum()}")
        print("=" * 80)
        
        return normalized_data
    
    def get_feature_list(self):
        """
        Get list of all generated features.
        
        Returns:
            List of feature column names
        """
        if self.processed_data is None:
            raise ValueError("Pipeline not run yet. Call run_full_pipeline first.")
        
        exclude_cols = ['timestamp', 'market', 'subscriber_count']
        features = [col for col in self.processed_data.columns if col not in exclude_cols]
        
        return features
    
    def get_feature_groups(self):
        """
        Get features organized by type.
        
        Returns:
            Dictionary of feature groups
        """
        if self.processed_data is None:
            raise ValueError("Pipeline not run yet. Call run_full_pipeline first.")
        
        all_features = self.get_feature_list()
        
        feature_groups = {
            'time_components': [f for f in all_features if any(
                x in f for x in ['hour', 'day', 'week', 'month', 'quarter', 'year', 
                               'weekend', 'sin', 'cos', 'holiday']
            )],
            'rolling_features': [f for f in all_features if 'rolling' in f],
            'lag_features': [f for f in all_features if 'lag' in f],
            'weather_features': [f for f in all_features if any(
                x in f for x in ['temperature', 'precipitation', 'humidity', 
                               'wind_speed', 'severe_weather']
            )],
            'event_features': [f for f in all_features if any(
                x in f for x in ['outage', 'campaign']
            )],
            'interaction_features': [f for f in all_features if 'interaction' in f or 
                                   any(x in f for x in ['_weekend', '_holiday', 'multiple_events'])]
        }
        
        return feature_groups
