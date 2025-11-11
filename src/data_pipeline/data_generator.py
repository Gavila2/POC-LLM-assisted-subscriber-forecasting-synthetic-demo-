"""
Synthetic data generation for subscriber forecasting.
Generates subscriber counts and weather data for demonstration purposes.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SyntheticDataGenerator:
    """Generate synthetic subscriber and weather data for testing."""
    
    def __init__(self, start_date='2023-01-01', end_date='2023-12-31', markets=None):
        """
        Initialize the data generator.
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            markets: List of market identifiers (default: ['NYC', 'LA', 'CHI'])
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.markets = markets or ['NYC', 'LA', 'CHI']
        
    def generate_subscriber_data(self, seed=42):
        """
        Generate synthetic subscriber data with trends and seasonality.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with subscriber counts by timestamp and market
        """
        np.random.seed(seed)
        
        # Generate hourly timestamps
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        
        data = []
        for market in self.markets:
            # Base subscriber count varies by market
            base_subscribers = {'NYC': 10000, 'LA': 8000, 'CHI': 6000}.get(market, 5000)
            
            for timestamp in date_range:
                # Add trend (growing over time)
                days_elapsed = (timestamp - self.start_date).days
                trend = days_elapsed * 5
                
                # Add daily seasonality
                hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * timestamp.hour / 24)
                
                # Add weekly seasonality (weekends are different)
                day_of_week = timestamp.dayofweek
                weekly_factor = 1.2 if day_of_week >= 5 else 1.0
                
                # Add monthly seasonality
                monthly_factor = 1.0 + 0.1 * np.sin(2 * np.pi * timestamp.month / 12)
                
                # Random noise
                noise = np.random.normal(0, 100)
                
                # Calculate subscribers
                subscribers = int(base_subscribers * hour_factor * weekly_factor * 
                                monthly_factor + trend + noise)
                
                data.append({
                    'timestamp': timestamp,
                    'market': market,
                    'subscriber_count': max(0, subscribers)
                })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_weather_data(self, seed=43):
        """
        Generate synthetic weather data.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with weather metrics by timestamp and market
        """
        np.random.seed(seed)
        
        # Generate hourly timestamps
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        
        data = []
        for market in self.markets:
            # Base temperature varies by market and season
            base_temp = {'NYC': 50, 'LA': 70, 'CHI': 45}.get(market, 60)
            
            for timestamp in date_range:
                # Seasonal temperature variation
                season_temp = 20 * np.sin(2 * np.pi * (timestamp.dayofyear - 80) / 365)
                
                # Daily variation
                daily_temp = 10 * np.sin(2 * np.pi * timestamp.hour / 24)
                
                # Random variation
                temp_noise = np.random.normal(0, 5)
                temperature = base_temp + season_temp + daily_temp + temp_noise
                
                # Precipitation (more likely in certain seasons)
                precip_prob = 0.1 + 0.1 * np.sin(2 * np.pi * timestamp.dayofyear / 365)
                precipitation = np.random.exponential(0.5) if np.random.random() < precip_prob else 0
                
                # Humidity (correlated with precipitation)
                humidity = min(100, max(0, 50 + 30 * (precipitation > 0) + np.random.normal(0, 10)))
                
                # Wind speed
                wind_speed = max(0, np.random.gamma(2, 5))
                
                # Severe weather flag (rare events)
                severe_weather = 1 if (precipitation > 1.5 or wind_speed > 30) else 0
                
                data.append({
                    'timestamp': timestamp,
                    'market': market,
                    'temperature': temperature,
                    'precipitation': precipitation,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'severe_weather': severe_weather
                })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_event_data(self, seed=44):
        """
        Generate synthetic event data (outages, campaigns).
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with event information
        """
        np.random.seed(seed)
        
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='H')
        
        data = []
        for market in self.markets:
            for timestamp in date_range:
                # Outages (rare events)
                outage = 1 if np.random.random() < 0.001 else 0
                
                # Marketing campaigns (scheduled events)
                # More likely at beginning of months and quarters
                campaign = 1 if (timestamp.day <= 7 and timestamp.hour == 10) else 0
                
                data.append({
                    'timestamp': timestamp,
                    'market': market,
                    'outage': outage,
                    'campaign': campaign
                })
        
        df = pd.DataFrame(data)
        return df
    
    def save_data(self, output_dir='data/raw'):
        """
        Generate and save all synthetic datasets.
        
        Args:
            output_dir: Directory to save the data files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate and save subscriber data
        subscriber_df = self.generate_subscriber_data()
        subscriber_df.to_csv(f'{output_dir}/subscriber_data.csv', index=False)
        
        # Generate and save weather data
        weather_df = self.generate_weather_data()
        weather_df.to_csv(f'{output_dir}/weather_data.csv', index=False)
        
        # Generate and save event data
        event_df = self.generate_event_data()
        event_df.to_csv(f'{output_dir}/event_data.csv', index=False)
        
        print(f"Generated data saved to {output_dir}/")
        print(f"  - subscriber_data.csv: {len(subscriber_df)} rows")
        print(f"  - weather_data.csv: {len(weather_df)} rows")
        print(f"  - event_data.csv: {len(event_df)} rows")
