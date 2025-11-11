"""
Data merging module for combining subscriber, weather, and event data.
"""
import pandas as pd
import numpy as np


class DataMerger:
    """Merge multiple data sources for model training."""
    
    def __init__(self):
        """Initialize the data merger."""
        self.merged_data = None
        
    def merge_datasets(self, subscriber_df, weather_df, event_df):
        """
        Merge subscriber, weather, and event datasets on timestamp and market.
        
        Args:
            subscriber_df: DataFrame with subscriber data
            weather_df: DataFrame with weather data
            event_df: DataFrame with event data
            
        Returns:
            Merged DataFrame
        """
        # Ensure timestamp columns are datetime
        subscriber_df['timestamp'] = pd.to_datetime(subscriber_df['timestamp'])
        weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
        event_df['timestamp'] = pd.to_datetime(event_df['timestamp'])
        
        # Merge on timestamp and market
        merged = subscriber_df.merge(
            weather_df,
            on=['timestamp', 'market'],
            how='left'
        )
        
        merged = merged.merge(
            event_df,
            on=['timestamp', 'market'],
            how='left'
        )
        
        # Fill any missing values from merge
        merged = merged.fillna(0)
        
        self.merged_data = merged
        return merged
    
    def load_and_merge(self, subscriber_path, weather_path, event_path):
        """
        Load CSV files and merge them.
        
        Args:
            subscriber_path: Path to subscriber data CSV
            weather_path: Path to weather data CSV
            event_path: Path to event data CSV
            
        Returns:
            Merged DataFrame
        """
        subscriber_df = pd.read_csv(subscriber_path)
        weather_df = pd.read_csv(weather_path)
        event_df = pd.read_csv(event_path)
        
        return self.merge_datasets(subscriber_df, weather_df, event_df)
    
    def save_merged_data(self, output_path):
        """
        Save merged data to CSV.
        
        Args:
            output_path: Path to save the merged data
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Run merge_datasets first.")
        
        self.merged_data.to_csv(output_path, index=False)
        print(f"Merged data saved to {output_path}")
        print(f"Shape: {self.merged_data.shape}")
