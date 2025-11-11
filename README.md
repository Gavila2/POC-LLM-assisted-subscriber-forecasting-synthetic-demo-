# Subscriber Forecasting with Synthetic Data

Proof of concept for building a comprehensive subscriber forecasting system with synthetic data generation, feature engineering, and time-based backtesting.

## Overview

This project demonstrates a complete end-to-end pipeline for subscriber forecasting that includes:

- **Data Generation**: Synthetic subscriber, weather, and event data
- **Data Merging**: Combining multiple data sources
- **Data Cleaning**: Handling missing values, duplicates, and outliers
- **Feature Engineering**: Time-based and event-based features
- **Normalization**: Market-specific data transformation
- **Backtesting**: Time-based model evaluation
- **Monitoring**: Model performance tracking

## Features

### Data Pipeline

1. **Synthetic Data Generation**
   - Subscriber counts with trends and seasonality
   - Weather data (temperature, precipitation, humidity, wind speed)
   - Event data (outages, marketing campaigns)

2. **Data Merging**
   - Combines subscriber, weather, and event data on timestamp and market
   - Handles missing data from merges

3. **Data Cleaning**
   - De-duplication of records
   - Interpolation of missing values
   - Smoothing of noisy time series
   - Outlier handling (clipping or removal)

### Feature Engineering

1. **Time Features**
   - Basic components: hour, day of week, month, quarter, year
   - Cyclical encoding: sin/cos transformations for periodic features
   - Boolean flags: weekends, month start/end, quarter start/end
   - Holiday flags: holiday detection and proximity features

2. **Rolling Features**
   - Rolling means, standard deviations, min, max
   - Configurable window sizes (default: 3h, 6h, 12h, 24h, 168h)
   - Computed by market

3. **Lag Features**
   - Historical values at different time lags
   - Configurable lag periods (default: 1h, 6h, 12h, 24h, 168h)

4. **Event Features**
   - Severe weather event counts and recency
   - Outage impact zones and recency
   - Campaign momentum and recency
   - Interaction terms between events and time features

### Normalization & Transformation

- Market-specific normalization (StandardScaler, MinMaxScaler, RobustScaler)
- Log and Box-Cox transformations
- Serializable scalers for production use

### Model Evaluation

1. **Time-Based Backtesting**
   - Rolling window backtest
   - Expanding window backtest
   - Time series cross-validation
   - Metrics: MAE, RMSE, R², MAPE

2. **Model Monitoring**
   - Performance tracking over time
   - Market-level performance analysis
   - Drift detection
   - Error distribution analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/Gavila2/POC-LLM-assisted-subscriber-forecasting-synthetic-demo-.git
cd POC-LLM-assisted-subscriber-forecasting-synthetic-demo-

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python run_pipeline.py
```

This will:
1. Generate synthetic data for 3 months
2. Merge subscriber, weather, and event data
3. Clean and engineer features
4. Normalize by market
5. Run time-based backtesting
6. Generate monitoring report

### Custom Pipeline

```python
from src.data_pipeline.pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline(
    start_date='2023-01-01',
    end_date='2023-12-31',
    markets=['NYC', 'LA', 'CHI'],
    country='US'
)

# Run pipeline
processed_data = pipeline.run_full_pipeline(
    raw_data_dir='data/raw',
    processed_data_dir='data/processed',
    generate_new_data=True
)

# Get feature groups
feature_groups = pipeline.get_feature_groups()
```

### Individual Components

```python
# Generate data only
from src.data_pipeline.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    start_date='2023-01-01',
    end_date='2023-12-31'
)
subscriber_df = generator.generate_subscriber_data()
weather_df = generator.generate_weather_data()
event_df = generator.generate_event_data()

# Feature engineering only
from src.features.time_features import TimeFeatureEngineer

engineer = TimeFeatureEngineer(country='US')
df_with_features = engineer.create_all_time_features(
    df,
    target_col='subscriber_count'
)

# Backtesting only
from src.models.backtest import TimeBasedBacktest
from sklearn.ensemble import RandomForestRegressor

backtester = TimeBasedBacktest(n_splits=5)
model = RandomForestRegressor(n_estimators=100)

results = backtester.cross_validate(
    df,
    model,
    target_col='subscriber_count'
)
```

## Project Structure

```
.
├── src/
│   ├── data_pipeline/
│   │   ├── data_generator.py    # Synthetic data generation
│   │   ├── data_merger.py       # Data merging logic
│   │   ├── data_cleaner.py      # Data cleaning utilities
│   │   └── pipeline.py          # Main pipeline orchestration
│   ├── features/
│   │   ├── time_features.py     # Time-based feature engineering
│   │   ├── event_features.py    # Event-based feature engineering
│   │   └── normalization.py     # Data normalization
│   └── models/
│       ├── backtest.py          # Time-based backtesting
│       └── monitoring.py        # Model monitoring
├── tests/
│   ├── test_data_pipeline/
│   └── test_features/
├── data/
│   ├── raw/                     # Raw generated data
│   └── processed/               # Processed data and artifacts
├── requirements.txt             # Python dependencies
├── run_pipeline.py              # Main execution script
└── README.md                    # This file
```

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Data Schema

### Subscriber Data
- `timestamp`: DateTime of measurement
- `market`: Market identifier (NYC, LA, CHI)
- `subscriber_count`: Number of subscribers

### Weather Data
- `timestamp`: DateTime of measurement
- `market`: Market identifier
- `temperature`: Temperature in Fahrenheit
- `precipitation`: Precipitation in inches
- `humidity`: Relative humidity (0-100%)
- `wind_speed`: Wind speed in mph
- `severe_weather`: Binary flag for severe weather events

### Event Data
- `timestamp`: DateTime of event
- `market`: Market identifier
- `outage`: Binary flag for service outages
- `campaign`: Binary flag for marketing campaigns

## Generated Features

After processing, the data includes:

- **Time Components**: 20+ features including hour, day, week, month, cyclical encodings
- **Holiday Features**: Holiday flags and proximity metrics
- **Rolling Features**: 20+ rolling statistics (mean, std, min, max)
- **Lag Features**: 5+ lag features at different time periods
- **Weather Features**: 5 base + 3 derived severe weather features
- **Event Features**: 8+ outage and campaign features
- **Interaction Features**: 7+ interaction terms

**Total**: 80+ engineered features

## Performance Metrics

The pipeline calculates the following metrics:

- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Root mean squared prediction error
- **R² (R-squared)**: Coefficient of determination
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error

## Monitoring

The monitoring system tracks:

- Prediction accuracy over time
- Performance by market
- Model drift detection
- Error distribution analysis

## Contributing

This is a proof-of-concept project. For contributions:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is provided as-is for demonstration purposes.
