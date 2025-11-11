"""
Data normalization and transformation module.
Normalizes features by market and applies transformations.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle


class DataNormalizer:
    """Normalize and transform data by market."""
    
    def __init__(self, method='standard'):
        """
        Initialize the data normalizer.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scalers = {}
        self.feature_columns = []
        
    def fit_transform_by_market(self, df, exclude_cols=None):
        """
        Fit scalers and transform data separately for each market.
        
        Args:
            df: DataFrame with features
            exclude_cols: Columns to exclude from normalization
            
        Returns:
            Normalized DataFrame
        """
        df_normalized = df.copy()
        
        # Default columns to exclude
        if exclude_cols is None:
            exclude_cols = ['timestamp', 'market', 'subscriber_count']
        
        # Get numeric columns to normalize
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns
        self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]
        
        # Convert feature columns to float to avoid type issues
        for col in self.feature_columns:
            df_normalized[col] = df_normalized[col].astype(float)
        
        print(f"Normalizing {len(self.feature_columns)} features by market...")
        
        # Normalize each market separately
        for market in df_normalized['market'].unique():
            market_mask = df_normalized['market'] == market
            
            # Select scaler
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'minmax':
                scaler = MinMaxScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
            
            # Fit and transform
            df_normalized.loc[market_mask, self.feature_columns] = scaler.fit_transform(
                df_normalized.loc[market_mask, self.feature_columns]
            )
            
            # Store scaler for this market
            self.scalers[market] = scaler
            
            print(f"  - Normalized market: {market}")
        
        return df_normalized
    
    def transform_by_market(self, df):
        """
        Transform data using fitted scalers.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Normalized DataFrame
        """
        if not self.scalers:
            raise ValueError("Scalers not fitted. Call fit_transform_by_market first.")
        
        df_normalized = df.copy()
        
        for market in df_normalized['market'].unique():
            if market not in self.scalers:
                print(f"Warning: No scaler found for market {market}. Skipping.")
                continue
            
            market_mask = df_normalized['market'] == market
            scaler = self.scalers[market]
            
            df_normalized.loc[market_mask, self.feature_columns] = scaler.transform(
                df_normalized.loc[market_mask, self.feature_columns]
            )
        
        return df_normalized
    
    def inverse_transform_by_market(self, df):
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            df: Normalized DataFrame
            
        Returns:
            DataFrame in original scale
        """
        if not self.scalers:
            raise ValueError("Scalers not fitted. Call fit_transform_by_market first.")
        
        df_original = df.copy()
        
        for market in df_original['market'].unique():
            if market not in self.scalers:
                continue
            
            market_mask = df_original['market'] == market
            scaler = self.scalers[market]
            
            df_original.loc[market_mask, self.feature_columns] = scaler.inverse_transform(
                df_original.loc[market_mask, self.feature_columns]
            )
        
        return df_original
    
    def apply_log_transform(self, df, columns=None):
        """
        Apply log transformation to specified columns.
        
        Args:
            df: DataFrame with features
            columns: Columns to transform (default: ['subscriber_count'])
            
        Returns:
            DataFrame with log-transformed columns
        """
        df_transformed = df.copy()
        
        if columns is None:
            columns = ['subscriber_count']
        
        for col in columns:
            if col in df_transformed.columns:
                # Add small constant to avoid log(0)
                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                print(f"Applied log transform to {col}")
        
        return df_transformed
    
    def apply_box_cox_transform(self, df, columns=None):
        """
        Apply Box-Cox transformation to specified columns.
        
        Args:
            df: DataFrame with features
            columns: Columns to transform
            
        Returns:
            DataFrame with Box-Cox transformed columns
        """
        from scipy.stats import boxcox
        
        df_transformed = df.copy()
        
        if columns is None:
            columns = ['subscriber_count']
        
        for col in columns:
            if col in df_transformed.columns:
                # Box-Cox requires positive values
                if (df_transformed[col] > 0).all():
                    transformed, lambda_param = boxcox(df_transformed[col])
                    df_transformed[f'{col}_boxcox'] = transformed
                    print(f"Applied Box-Cox transform to {col} (lambda={lambda_param:.3f})")
                else:
                    print(f"Warning: {col} contains non-positive values. Skipping Box-Cox.")
        
        return df_transformed
    
    def save_scalers(self, filepath):
        """
        Save fitted scalers to disk.
        
        Args:
            filepath: Path to save the scalers
        """
        if not self.scalers:
            raise ValueError("No scalers to save. Fit scalers first.")
        
        scaler_data = {
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'method': self.method
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        print(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath):
        """
        Load fitted scalers from disk.
        
        Args:
            filepath: Path to load the scalers from
        """
        with open(filepath, 'rb') as f:
            scaler_data = pickle.load(f)
        
        self.scalers = scaler_data['scalers']
        self.feature_columns = scaler_data['feature_columns']
        self.method = scaler_data['method']
        
        print(f"Scalers loaded from {filepath}")
        print(f"  - Method: {self.method}")
        print(f"  - Markets: {list(self.scalers.keys())}")
        print(f"  - Features: {len(self.feature_columns)}")
