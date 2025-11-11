"""
Time-based feature engineering module.
Creates hour, day, week, month features, rolling averages, and holiday flags.
"""
import pandas as pd
import numpy as np
import holidays


class TimeFeatureEngineer:
    """Generate time-based features for forecasting."""
    
    def __init__(self, country='US'):
        """
        Initialize the feature engineer.
        
        Args:
            country: Country code for holiday calendar (default: 'US')
        """
        self.country = country
        self.holiday_calendar = holidays.country_holidays(country)
    
    def add_time_components(self, df):
        """
        Add basic time components (hour, day, week, month).
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with time component features
        """
        df_features = df.copy()
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        
        # Extract time components
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['day_of_month'] = df_features['timestamp'].dt.day
        df_features['day_of_year'] = df_features['timestamp'].dt.dayofyear
        df_features['week_of_year'] = df_features['timestamp'].dt.isocalendar().week
        df_features['month'] = df_features['timestamp'].dt.month
        df_features['quarter'] = df_features['timestamp'].dt.quarter
        df_features['year'] = df_features['timestamp'].dt.year
        
        # Boolean flags
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        df_features['is_month_start'] = df_features['timestamp'].dt.is_month_start.astype(int)
        df_features['is_month_end'] = df_features['timestamp'].dt.is_month_end.astype(int)
        df_features['is_quarter_start'] = df_features['timestamp'].dt.is_quarter_start.astype(int)
        df_features['is_quarter_end'] = df_features['timestamp'].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for periodic features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        return df_features
    
    def add_holiday_flags(self, df):
        """
        Add holiday flags based on country calendar.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with holiday features
        """
        df_features = df.copy()
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        
        # Check if date is a holiday
        df_features['is_holiday'] = df_features['timestamp'].dt.date.apply(
            lambda x: 1 if x in self.holiday_calendar else 0
        )
        
        # Days before/after holiday
        df_features['days_to_holiday'] = 0
        df_features['days_from_holiday'] = 0
        
        for idx, row in df_features.iterrows():
            date = row['timestamp'].date()
            
            # Check next 7 days for upcoming holiday
            for days_ahead in range(1, 8):
                future_date = date + pd.Timedelta(days=days_ahead)
                if future_date in self.holiday_calendar:
                    df_features.loc[idx, 'days_to_holiday'] = days_ahead
                    break
            
            # Check previous 7 days for recent holiday
            for days_ago in range(1, 8):
                past_date = date - pd.Timedelta(days=days_ago)
                if past_date in self.holiday_calendar:
                    df_features.loc[idx, 'days_from_holiday'] = days_ago
                    break
        
        return df_features
    
    def add_rolling_features(self, df, target_col='subscriber_count', 
                           windows=[3, 6, 12, 24, 168]):
        """
        Add rolling average features for target column.
        
        Args:
            df: DataFrame with data
            target_col: Column to create rolling features for
            windows: List of window sizes (in hours)
            
        Returns:
            DataFrame with rolling features
        """
        df_features = df.copy()
        
        # Sort by timestamp and market
        df_features = df_features.sort_values(['market', 'timestamp'])
        
        if target_col in df_features.columns:
            for window in windows:
                # Rolling mean
                df_features[f'{target_col}_rolling_mean_{window}h'] = (
                    df_features.groupby('market')[target_col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                )
                
                # Rolling std
                df_features[f'{target_col}_rolling_std_{window}h'] = (
                    df_features.groupby('market')[target_col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).std())
                )
                
                # Rolling min/max
                df_features[f'{target_col}_rolling_min_{window}h'] = (
                    df_features.groupby('market')[target_col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).min())
                )
                
                df_features[f'{target_col}_rolling_max_{window}h'] = (
                    df_features.groupby('market')[target_col]
                    .transform(lambda x: x.rolling(window=window, min_periods=1).max())
                )
        
        return df_features
    
    def add_lag_features(self, df, target_col='subscriber_count', 
                        lags=[1, 6, 12, 24, 168]):
        """
        Add lagged features for target column.
        
        Args:
            df: DataFrame with data
            target_col: Column to create lag features for
            lags: List of lag periods (in hours)
            
        Returns:
            DataFrame with lag features
        """
        df_features = df.copy()
        df_features = df_features.sort_values(['market', 'timestamp'])
        
        if target_col in df_features.columns:
            for lag in lags:
                df_features[f'{target_col}_lag_{lag}h'] = (
                    df_features.groupby('market')[target_col]
                    .shift(lag)
                )
        
        # Fill NaN values from lagging with forward fill
        lag_cols = [col for col in df_features.columns if '_lag_' in col]
        for col in lag_cols:
            df_features[col] = df_features.groupby('market')[col].fillna(method='bfill')
        
        return df_features
    
    def create_all_time_features(self, df, target_col='subscriber_count',
                                 rolling_windows=[3, 6, 12, 24, 168],
                                 lags=[1, 6, 12, 24, 168]):
        """
        Create all time-based features.
        
        Args:
            df: DataFrame with data
            target_col: Target column for rolling and lag features
            rolling_windows: Window sizes for rolling features
            lags: Lag periods for lag features
            
        Returns:
            DataFrame with all time features
        """
        print("Creating time features...")
        
        df_features = df.copy()
        
        # Add basic time components
        df_features = self.add_time_components(df_features)
        print("  - Added time components")
        
        # Add holiday flags
        df_features = self.add_holiday_flags(df_features)
        print("  - Added holiday flags")
        
        # Add rolling features
        df_features = self.add_rolling_features(
            df_features, target_col=target_col, windows=rolling_windows
        )
        print("  - Added rolling features")
        
        # Add lag features
        df_features = self.add_lag_features(
            df_features, target_col=target_col, lags=lags
        )
        print("  - Added lag features")
        
        print(f"Time feature engineering complete. Shape: {df_features.shape}")
        
        return df_features
