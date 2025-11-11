"""
Data cleaning module for handling missing values, duplicates, and noise.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d


class DataCleaner:
    """Clean and preprocess data for model training."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        pass
    
    def remove_duplicates(self, df):
        """
        Remove duplicate rows based on timestamp and market.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df_clean = df.drop_duplicates(subset=['timestamp', 'market'], keep='first')
        removed = initial_rows - len(df_clean)
        
        if removed > 0:
            print(f"Removed {removed} duplicate rows")
        
        return df_clean
    
    def interpolate_missing_values(self, df, columns=None, method='linear'):
        """
        Interpolate missing values in specified columns.
        
        Args:
            df: DataFrame to clean
            columns: List of columns to interpolate (default: all numeric columns)
            method: Interpolation method ('linear', 'time', 'polynomial')
            
        Returns:
            DataFrame with interpolated values
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in df_clean.columns:
                missing_before = df_clean[col].isna().sum()
                
                if missing_before > 0:
                    # Group by market and interpolate within each market
                    if 'market' in df_clean.columns:
                        df_clean[col] = df_clean.groupby('market')[col].transform(
                            lambda x: x.interpolate(method=method, limit_direction='both')
                        )
                    else:
                        df_clean[col] = df_clean[col].interpolate(method=method, limit_direction='both')
                    
                    missing_after = df_clean[col].isna().sum()
                    filled = missing_before - missing_after
                    
                    if filled > 0:
                        print(f"Interpolated {filled} missing values in {col}")
                    
                    # Fill any remaining NaN with forward/backward fill
                    df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        return df_clean
    
    def smooth_noisy_series(self, df, columns=None, window_size=3):
        """
        Apply smoothing to noisy time series columns.
        
        Args:
            df: DataFrame to smooth
            columns: List of columns to smooth (default: numeric columns)
            window_size: Size of the smoothing window
            
        Returns:
            DataFrame with smoothed values
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = ['subscriber_count', 'temperature', 'precipitation', 
                      'humidity', 'wind_speed']
            columns = [col for col in columns if col in df_clean.columns]
        
        for col in columns:
            if col in df_clean.columns:
                # Group by market and smooth within each market
                if 'market' in df_clean.columns:
                    df_clean[col] = df_clean.groupby('market')[col].transform(
                        lambda x: uniform_filter1d(x, size=window_size, mode='nearest')
                    )
                else:
                    df_clean[col] = uniform_filter1d(
                        df_clean[col].values, size=window_size, mode='nearest'
                    )
        
        return df_clean
    
    def handle_outliers(self, df, columns=None, method='clip', n_std=3):
        """
        Handle outliers in specified columns.
        
        Args:
            df: DataFrame to clean
            columns: List of columns to check for outliers
            method: Method to handle outliers ('clip', 'remove')
            n_std: Number of standard deviations for outlier threshold
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col in df_clean.columns:
                # Calculate outlier thresholds by market
                if 'market' in df_clean.columns:
                    for market in df_clean['market'].unique():
                        mask = df_clean['market'] == market
                        data = df_clean.loc[mask, col]
                        
                        mean = data.mean()
                        std = data.std()
                        
                        lower_bound = mean - n_std * std
                        upper_bound = mean + n_std * std
                        
                        if method == 'clip':
                            df_clean.loc[mask, col] = data.clip(lower_bound, upper_bound)
                        elif method == 'remove':
                            outliers = (data < lower_bound) | (data > upper_bound)
                            df_clean.loc[mask & outliers, col] = np.nan
                else:
                    mean = df_clean[col].mean()
                    std = df_clean[col].std()
                    
                    lower_bound = mean - n_std * std
                    upper_bound = mean + n_std * std
                    
                    if method == 'clip':
                        df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    elif method == 'remove':
                        outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                        df_clean.loc[outliers, col] = np.nan
        
        return df_clean
    
    def clean_pipeline(self, df, smooth=True, interpolate=True, 
                       handle_outliers_flag=True, window_size=3):
        """
        Run complete cleaning pipeline.
        
        Args:
            df: DataFrame to clean
            smooth: Whether to apply smoothing
            interpolate: Whether to interpolate missing values
            handle_outliers_flag: Whether to handle outliers
            window_size: Window size for smoothing
            
        Returns:
            Cleaned DataFrame
        """
        print("Starting data cleaning pipeline...")
        
        # Remove duplicates
        df_clean = self.remove_duplicates(df)
        
        # Handle outliers (before interpolation)
        if handle_outliers_flag:
            df_clean = self.handle_outliers(df_clean, method='clip')
        
        # Interpolate missing values
        if interpolate:
            df_clean = self.interpolate_missing_values(df_clean)
        
        # Smooth noisy series
        if smooth:
            df_clean = self.smooth_noisy_series(df_clean, window_size=window_size)
        
        print("Data cleaning complete.")
        print(f"Final shape: {df_clean.shape}")
        
        return df_clean
