"""
Event-based feature engineering module.
Creates features for severe weather, outages, campaigns, and interaction terms.
"""
import pandas as pd
import numpy as np


class EventFeatureEngineer:
    """Generate event-based features for forecasting."""
    
    def __init__(self):
        """Initialize the event feature engineer."""
        pass
    
    def add_severe_weather_features(self, df):
        """
        Add features related to severe weather events.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with severe weather features
        """
        df_features = df.copy()
        
        # Severe weather already exists, but add derived features
        if 'severe_weather' in df_features.columns:
            # Count of severe weather events in past 24 hours
            df_features = df_features.sort_values(['market', 'timestamp'])
            df_features['severe_weather_count_24h'] = (
                df_features.groupby('market')['severe_weather']
                .transform(lambda x: x.rolling(window=24, min_periods=1).sum())
            )
            
            # Days since last severe weather event
            df_features['days_since_severe_weather'] = 0.0
            
            for market in df_features['market'].unique():
                market_mask = df_features['market'] == market
                severe_events = df_features.loc[market_mask, 'severe_weather'] == 1
                
                days_since = []
                last_event_idx = None
                
                for idx in df_features[market_mask].index:
                    if df_features.loc[idx, 'severe_weather'] == 1:
                        last_event_idx = idx
                        days_since.append(0)
                    elif last_event_idx is not None:
                        hours_diff = (df_features.loc[idx, 'timestamp'] - 
                                    df_features.loc[last_event_idx, 'timestamp']).total_seconds() / 3600
                        days_since.append(hours_diff / 24)
                    else:
                        days_since.append(999)  # No previous event
                
                df_features.loc[market_mask, 'days_since_severe_weather'] = days_since
        
        return df_features
    
    def add_outage_features(self, df):
        """
        Add features related to service outages.
        
        Args:
            df: DataFrame with outage data
            
        Returns:
            DataFrame with outage features
        """
        df_features = df.copy()
        
        if 'outage' in df_features.columns:
            # Count of outages in past 24 hours
            df_features = df_features.sort_values(['market', 'timestamp'])
            df_features['outage_count_24h'] = (
                df_features.groupby('market')['outage']
                .transform(lambda x: x.rolling(window=24, min_periods=1).sum())
            )
            
            # Hours since last outage
            df_features['hours_since_outage'] = 0.0
            
            for market in df_features['market'].unique():
                market_mask = df_features['market'] == market
                
                hours_since = []
                last_outage_idx = None
                
                for idx in df_features[market_mask].index:
                    if df_features.loc[idx, 'outage'] == 1:
                        last_outage_idx = idx
                        hours_since.append(0)
                    elif last_outage_idx is not None:
                        hours_diff = (df_features.loc[idx, 'timestamp'] - 
                                    df_features.loc[last_outage_idx, 'timestamp']).total_seconds() / 3600
                        hours_since.append(hours_diff)
                    else:
                        hours_since.append(999)  # No previous outage
                
                df_features.loc[market_mask, 'hours_since_outage'] = hours_since
            
            # Outage impact flag (affected for next 12 hours)
            df_features['outage_impact'] = 0
            for market in df_features['market'].unique():
                market_mask = df_features['market'] == market
                market_data = df_features[market_mask].copy()
                
                for idx in market_data.index:
                    if df_features.loc[idx, 'outage'] == 1:
                        # Mark next 12 hours as impacted
                        timestamp = df_features.loc[idx, 'timestamp']
                        impact_mask = (
                            (df_features['market'] == market) &
                            (df_features['timestamp'] >= timestamp) &
                            (df_features['timestamp'] <= timestamp + pd.Timedelta(hours=12))
                        )
                        df_features.loc[impact_mask, 'outage_impact'] = 1
        
        return df_features
    
    def add_campaign_features(self, df):
        """
        Add features related to marketing campaigns.
        
        Args:
            df: DataFrame with campaign data
            
        Returns:
            DataFrame with campaign features
        """
        df_features = df.copy()
        
        if 'campaign' in df_features.columns:
            # Count of campaigns in past 7 days
            df_features = df_features.sort_values(['market', 'timestamp'])
            df_features['campaign_count_7d'] = (
                df_features.groupby('market')['campaign']
                .transform(lambda x: x.rolling(window=168, min_periods=1).sum())
            )
            
            # Days since last campaign
            df_features['days_since_campaign'] = 0.0
            
            for market in df_features['market'].unique():
                market_mask = df_features['market'] == market
                
                days_since = []
                last_campaign_idx = None
                
                for idx in df_features[market_mask].index:
                    if df_features.loc[idx, 'campaign'] == 1:
                        last_campaign_idx = idx
                        days_since.append(0)
                    elif last_campaign_idx is not None:
                        hours_diff = (df_features.loc[idx, 'timestamp'] - 
                                    df_features.loc[last_campaign_idx, 'timestamp']).total_seconds() / 3600
                        days_since.append(hours_diff / 24)
                    else:
                        days_since.append(999)  # No previous campaign
                
                df_features.loc[market_mask, 'days_since_campaign'] = days_since
            
            # Campaign momentum (campaigns in past 30 days)
            df_features['campaign_momentum'] = (
                df_features.groupby('market')['campaign']
                .transform(lambda x: x.rolling(window=720, min_periods=1).sum())
            )
        
        return df_features
    
    def add_interaction_terms(self, df):
        """
        Add interaction terms between different features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with interaction terms
        """
        df_features = df.copy()
        
        # Weather and time interactions
        if 'temperature' in df_features.columns and 'is_weekend' in df_features.columns:
            df_features['temp_weekend_interaction'] = (
                df_features['temperature'] * df_features['is_weekend']
            )
        
        if 'severe_weather' in df_features.columns and 'is_weekend' in df_features.columns:
            df_features['severe_weather_weekend'] = (
                df_features['severe_weather'] * df_features['is_weekend']
            )
        
        # Campaign and time interactions
        if 'campaign' in df_features.columns and 'is_weekend' in df_features.columns:
            df_features['campaign_weekend'] = (
                df_features['campaign'] * df_features['is_weekend']
            )
        
        if 'campaign' in df_features.columns and 'is_holiday' in df_features.columns:
            df_features['campaign_holiday'] = (
                df_features['campaign'] * df_features['is_holiday']
            )
        
        # Outage and weather interactions
        if 'outage' in df_features.columns and 'severe_weather' in df_features.columns:
            df_features['outage_severe_weather'] = (
                df_features['outage'] * df_features['severe_weather']
            )
        
        # Temperature and precipitation interaction
        if 'temperature' in df_features.columns and 'precipitation' in df_features.columns:
            df_features['temp_precip_interaction'] = (
                df_features['temperature'] * df_features['precipitation']
            )
        
        # Multiple event flag (campaign + severe weather)
        if 'campaign' in df_features.columns and 'severe_weather' in df_features.columns:
            df_features['multiple_events'] = (
                (df_features['campaign'] == 1) & (df_features['severe_weather'] == 1)
            ).astype(int)
        
        return df_features
    
    def create_all_event_features(self, df):
        """
        Create all event-based features.
        
        Args:
            df: DataFrame with data
            
        Returns:
            DataFrame with all event features
        """
        print("Creating event features...")
        
        df_features = df.copy()
        
        # Add severe weather features
        df_features = self.add_severe_weather_features(df_features)
        print("  - Added severe weather features")
        
        # Add outage features
        df_features = self.add_outage_features(df_features)
        print("  - Added outage features")
        
        # Add campaign features
        df_features = self.add_campaign_features(df_features)
        print("  - Added campaign features")
        
        # Add interaction terms
        df_features = self.add_interaction_terms(df_features)
        print("  - Added interaction terms")
        
        print(f"Event feature engineering complete. Shape: {df_features.shape}")
        
        return df_features
