#!/usr/bin/env python3
"""
Main script to run the subscriber forecasting data pipeline.
Demonstrates the complete workflow from data generation to model training.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_pipeline.pipeline import DataPipeline
from src.models.backtest import TimeBasedBacktest
from src.models.monitoring import ModelMonitor
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


def main():
    """Run the complete pipeline."""
    print("Subscriber Forecasting Pipeline Demo")
    print("=" * 80)
    
    # Step 1: Initialize and run data pipeline
    print("\n[STEP 1] Running data processing pipeline...\n")
    pipeline = DataPipeline(
        start_date='2023-01-01',
        end_date='2023-03-31',  # 3 months of data
        markets=['NYC', 'LA', 'CHI'],
        country='US'
    )
    
    # Run full pipeline
    processed_data = pipeline.run_full_pipeline(
        raw_data_dir='data/raw',
        processed_data_dir='data/processed',
        generate_new_data=True
    )
    
    # Display feature summary
    print("\n[STEP 2] Feature Summary\n")
    feature_groups = pipeline.get_feature_groups()
    for group_name, features in feature_groups.items():
        print(f"{group_name}: {len(features)} features")
    
    # Step 3: Prepare data for modeling
    print("\n[STEP 3] Preparing data for modeling...\n")
    
    # Get feature columns
    feature_cols = pipeline.get_feature_list()
    print(f"Total features for modeling: {len(feature_cols)}")
    
    # Filter out rows with NaN (from lag features at beginning)
    model_data = processed_data.dropna()
    print(f"Data points after removing NaN: {len(model_data):,}")
    
    # Split by market for demonstration
    nyc_data = model_data[model_data['market'] == 'NYC'].copy()
    print(f"NYC market data points: {len(nyc_data):,}")
    
    # Step 4: Run backtest
    print("\n[STEP 4] Running time-based backtest...\n")
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Initialize backtester
    backtester = TimeBasedBacktest(n_splits=3)
    
    # Run cross-validation
    cv_results = backtester.cross_validate(
        nyc_data,
        model,
        target_col='subscriber_count',
        feature_cols=feature_cols
    )
    
    # Step 5: Demonstrate monitoring
    print("\n[STEP 5] Setting up model monitoring...\n")
    
    monitor = ModelMonitor()
    
    # Simulate some predictions for monitoring
    print("Simulating predictions for monitoring...")
    sample_predictions = cv_results['fold_results'][0]
    
    # Log a few sample predictions
    for i in range(min(100, len(nyc_data))):
        row = nyc_data.iloc[i]
        # Simulate prediction (using actual for demo)
        predicted = row['subscriber_count'] + np.random.normal(0, 100)
        monitor.log_prediction(
            timestamp=row['timestamp'],
            market=row['market'],
            actual=row['subscriber_count'],
            predicted=predicted
        )
    
    # Generate monitoring report
    report = monitor.generate_report(window='30D')
    print("\n" + report)
    
    # Save monitoring history
    monitor.save_history('data/processed/monitoring_history.csv')
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - data/raw/subscriber_data.csv")
    print("  - data/raw/weather_data.csv")
    print("  - data/raw/event_data.csv")
    print("  - data/processed/merged_data.csv")
    print("  - data/processed/processed_data.csv")
    print("  - data/processed/scalers.pkl")
    print("  - data/processed/monitoring_history.csv")
    print("\nNext steps:")
    print("  1. Review the processed data and features")
    print("  2. Experiment with different models")
    print("  3. Tune hyperparameters")
    print("  4. Deploy model with monitoring")
    print("=" * 80)


if __name__ == '__main__':
    import numpy as np
    main()
