# POC-LLM-assisted-subscriber-forecasting-synthetic-demo-
Proof of concept using LLMs to build a subscriber forecast.

## Overview
This repository contains tools for generating synthetic subscriber and weather data for multiple markets (DMAs) to support forecasting model development.

## Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation
```bash
pip install -r requirements.txt
```

### Usage
Generate synthetic data:
```bash
python generate_data.py
```

This will create a `data/` directory with two CSV files:
- `subscribers.csv`: Hourly subscriber metrics (activations, cancellations, net_adds) for each market
- `weather.csv`: Hourly weather data (temperature, precipitation) for each market

### Data Description

#### Subscribers Data
- **market**: Market identifier (DMA1, DMA2, DMA3)
- **ts**: Timestamp (hourly)
- **activations**: Number of new subscriber activations
- **cancellations**: Number of subscriber cancellations
- **net_adds**: Net change in subscribers (activations - cancellations + noise)

The data includes:
- Diurnal patterns (hourly variations)
- Weekly patterns (weekday vs. weekend)
- Drift over time
- Random noise

#### Weather Data
- **market**: Market identifier (DMA1, DMA2, DMA3)
- **ts**: Timestamp (hourly)
- **temp**: Temperature in Fahrenheit
- **precip**: Precipitation amount
- **weather_flag**: 1 if precipitation occurred, 0 otherwise

The data spans from January 1, 2024 to March 31, 2024 (Q1 2024) with hourly granularity.
