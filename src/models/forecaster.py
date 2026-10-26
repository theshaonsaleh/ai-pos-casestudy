"""
Sales Forecasting Module
Predicts hourly and daily sales using time series techniques.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SalesForecaster:
    """Time series forecaster for POS sales data."""
    
    def __init__(self):
        """Initialize the forecaster."""
        self.hourly_patterns = {}
        self.daily_patterns = {}
        self.is_trained = False
    
    def train(self, transactions_df: pd.DataFrame):
        """
        Train the forecaster on historical transaction data.
        
        Args:
            transactions_df: DataFrame with transaction data including timestamp and total
        """
        df = transactions_df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_of_week
        df['date'] = df['timestamp'].dt.date
        
        # Aggregate to hourly sales
        hourly_sales = df.groupby(['date', 'hour', 'day_of_week'])['total'].sum().reset_index()
        
        # Calculate average sales by hour of day
        self.hourly_patterns = hourly_sales.groupby('hour')['total'].mean().to_dict()
        
        # Calculate average sales by day of week
        daily_sales = df.groupby(['date', 'day_of_week'])['total'].sum().reset_index()
        self.daily_patterns = daily_sales.groupby('day_of_week')['total'].mean().to_dict()
        
        # Calculate rolling averages for more sophisticated forecasting
        hourly_sales_sorted = hourly_sales.sort_values('date')
        hourly_sales_sorted['rolling_7d_avg'] = hourly_sales_sorted.groupby('hour')['total'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        self.hourly_rolling = hourly_sales_sorted.groupby('hour')['rolling_7d_avg'].last().to_dict()
        
        self.is_trained = True
    
    def forecast_hourly(self, forecast_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Forecast hourly sales for a given date.
        
        Args:
            forecast_date: Date to forecast (defaults to next day)
            
        Returns:
            DataFrame with hourly forecasts
        """
        if not self.is_trained:
            raise ValueError("Forecaster must be trained before forecasting")
        
        if forecast_date is None:
            forecast_date = pd.Timestamp.now().normalize()
        
        day_of_week = forecast_date.day_of_week
        
        # Get daily multiplier
        daily_avg = self.daily_patterns.get(day_of_week, 
                                            np.mean(list(self.daily_patterns.values())))
        overall_daily_avg = np.mean(list(self.daily_patterns.values()))
        daily_multiplier = daily_avg / overall_daily_avg if overall_daily_avg > 0 else 1.0
        
        forecasts = []
        for hour in range(6, 21):  # Operating hours
            # Use rolling average if available, else pattern
            base_forecast = self.hourly_rolling.get(hour, self.hourly_patterns.get(hour, 0))
            
            # Apply day-of-week adjustment
            forecast = base_forecast * daily_multiplier
            
            forecasts.append({
                'hour': hour,
                'forecast_sales': max(0, forecast),
                'day_of_week': day_of_week,
                'date': forecast_date.date()
            })
        
        return pd.DataFrame(forecasts)
    
    def forecast_daily(self, num_days: int = 7) -> pd.DataFrame:
        """
        Forecast daily sales for the next N days.
        
        Args:
            num_days: Number of days to forecast
            
        Returns:
            DataFrame with daily forecasts
        """
        if not self.is_trained:
            raise ValueError("Forecaster must be trained before forecasting")
        
        start_date = pd.Timestamp.now().normalize()
        forecasts = []
        
        for day_offset in range(num_days):
            forecast_date = start_date + pd.Timedelta(days=day_offset)
            day_of_week = forecast_date.day_of_week
            
            # Sum hourly forecasts for daily total
            hourly_forecast = self.forecast_hourly(forecast_date)
            daily_total = hourly_forecast['forecast_sales'].sum()
            
            forecasts.append({
                'date': forecast_date.date(),
                'day_of_week': day_of_week,
                'day_name': forecast_date.strftime('%A'),
                'forecast_sales': daily_total
            })
        
        return pd.DataFrame(forecasts)
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate forecaster performance on test data.
        
        Args:
            test_df: DataFrame with actual sales to compare against
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Forecaster must be trained before evaluation")
        
        test_df = test_df.copy()
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
        test_df['date'] = test_df['timestamp'].dt.date
        test_df['hour'] = test_df['timestamp'].dt.hour
        
        # Get unique dates in test set
        test_dates = sorted(test_df['date'].unique())
        
        # Generate forecasts for each date
        all_forecasts = []
        for date in test_dates:
            forecast = self.forecast_hourly(pd.Timestamp(date))
            all_forecasts.append(forecast)
        
        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        
        # Aggregate actual sales by date and hour
        actual_hourly = test_df.groupby(['date', 'hour'])['total'].sum().reset_index()
        actual_hourly.columns = ['date', 'hour', 'actual_sales']
        
        # Merge forecasts with actuals
        comparison = forecast_df.merge(actual_hourly, on=['date', 'hour'], how='inner')
        
        # Calculate metrics
        mae = mean_absolute_error(comparison['actual_sales'], comparison['forecast_sales'])
        rmse = np.sqrt(mean_squared_error(comparison['actual_sales'], comparison['forecast_sales']))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((comparison['actual_sales'] - comparison['forecast_sales']) / 
                              (comparison['actual_sales'] + 1e-10))) * 100
        
        # Daily-level metrics
        actual_daily = test_df.groupby('date')['total'].sum().reset_index()
        actual_daily.columns = ['date', 'actual_sales']
        
        forecast_daily_agg = forecast_df.groupby('date')['forecast_sales'].sum().reset_index()
        
        daily_comparison = forecast_daily_agg.merge(actual_daily, on='date', how='inner')
        
        daily_mae = mean_absolute_error(daily_comparison['actual_sales'], 
                                       daily_comparison['forecast_sales'])
        daily_rmse = np.sqrt(mean_squared_error(daily_comparison['actual_sales'], 
                                                daily_comparison['forecast_sales']))
        daily_mape = np.mean(np.abs((daily_comparison['actual_sales'] - daily_comparison['forecast_sales']) / 
                                   (daily_comparison['actual_sales'] + 1e-10))) * 100
        
        return {
            'hourly_mae': mae,
            'hourly_rmse': rmse,
            'hourly_mape': mape,
            'daily_mae': daily_mae,
            'daily_rmse': daily_rmse,
            'daily_mape': daily_mape,
            'num_hourly_predictions': len(comparison),
            'num_daily_predictions': len(daily_comparison)
        }
    
    def get_peak_hours(self, top_n: int = 5) -> pd.DataFrame:
        """
        Identify peak sales hours.
        
        Args:
            top_n: Number of top hours to return
            
        Returns:
            DataFrame with peak hours and average sales
        """
        if not self.is_trained:
            raise ValueError("Forecaster must be trained before analysis")
        
        hourly_df = pd.DataFrame([
            {'hour': hour, 'avg_sales': sales}
            for hour, sales in self.hourly_patterns.items()
        ]).sort_values('avg_sales', ascending=False).head(top_n)
        
        return hourly_df.reset_index(drop=True)
    
    def get_busiest_days(self) -> pd.DataFrame:
        """
        Identify busiest days of the week.
        
        Returns:
            DataFrame with days ranked by average sales
        """
        if not self.is_trained:
            raise ValueError("Forecaster must be trained before analysis")
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        daily_df = pd.DataFrame([
            {'day_of_week': dow, 'day_name': day_names[dow], 'avg_sales': sales}
            for dow, sales in self.daily_patterns.items()
        ]).sort_values('avg_sales', ascending=False)
        
        return daily_df.reset_index(drop=True)


if __name__ == "__main__":
    # Demo usage
    from src.data.generator import generate_pos_data
    
    print("Generating synthetic data...")
    transactions, _ = generate_pos_data(num_days=90)
    
    # Split into train and test
    split_date = pd.Timestamp('2024-03-01')
    train_df = transactions[transactions['timestamp'] < split_date]
    test_df = transactions[transactions['timestamp'] >= split_date]
    
    print(f"\nTraining on {len(train_df)} transactions...")
    forecaster = SalesForecaster()
    forecaster.train(train_df)
    
    print("\nPeak hours:")
    print(forecaster.get_peak_hours())
    
    print("\nBusiest days:")
    print(forecaster.get_busiest_days())
    
    print("\nForecasting next 7 days:")
    daily_forecast = forecaster.forecast_daily(num_days=7)
    print(daily_forecast)
    
    print("\nEvaluating on test set...")
    metrics = forecaster.evaluate(test_df)
    print(f"Hourly MAE: ${metrics['hourly_mae']:.2f}")
    print(f"Hourly RMSE: ${metrics['hourly_rmse']:.2f}")
    print(f"Hourly MAPE: {metrics['hourly_mape']:.1f}%")
    print(f"Daily MAE: ${metrics['daily_mae']:.2f}")
    print(f"Daily RMSE: ${metrics['daily_rmse']:.2f}")
    print(f"Daily MAPE: {metrics['daily_mape']:.1f}%")
