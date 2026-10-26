"""
Tests for ML models.
"""

import pytest
import pandas as pd
from src.data.generator import generate_pos_data
from src.models.forecaster import SalesForecaster
from src.models.basket_analyzer import BasketAnalyzer
from src.models.anomaly_detector import RefundAnomalyDetector
from src.models.llm_query import LLMQueryEngine


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    transactions, inventory = generate_pos_data(num_days=30, random_seed=42)
    return transactions, inventory


def test_forecaster_training(sample_data):
    """Test sales forecaster training."""
    transactions, _ = sample_data
    
    forecaster = SalesForecaster()
    forecaster.train(transactions)
    
    assert forecaster.is_trained
    assert len(forecaster.hourly_patterns) > 0
    assert len(forecaster.daily_patterns) > 0


def test_forecaster_hourly_forecast(sample_data):
    """Test hourly forecasting."""
    transactions, _ = sample_data
    
    forecaster = SalesForecaster()
    forecaster.train(transactions)
    
    forecast = forecaster.forecast_hourly()
    
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) > 0
    assert 'hour' in forecast.columns
    assert 'forecast_sales' in forecast.columns
    assert all(forecast['forecast_sales'] >= 0)


def test_forecaster_daily_forecast(sample_data):
    """Test daily forecasting."""
    transactions, _ = sample_data
    
    forecaster = SalesForecaster()
    forecaster.train(transactions)
    
    forecast = forecaster.forecast_daily(num_days=7)
    
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == 7
    assert 'date' in forecast.columns
    assert 'forecast_sales' in forecast.columns


def test_basket_analyzer_training(sample_data):
    """Test basket analyzer training."""
    transactions, _ = sample_data
    
    analyzer = BasketAnalyzer()
    analyzer.train(transactions)
    
    assert analyzer.is_trained


def test_basket_analyzer_recommendations(sample_data):
    """Test upsell recommendations."""
    transactions, _ = sample_data
    
    analyzer = BasketAnalyzer(min_support=0.01, min_confidence=0.3)
    analyzer.train(transactions)
    
    if not analyzer.rules.empty:
        recs = analyzer.get_upsell_recommendations(['Latte'], top_n=3)
        assert isinstance(recs, pd.DataFrame)


def test_basket_analyzer_rules(sample_data):
    """Test association rules generation."""
    transactions, _ = sample_data
    
    analyzer = BasketAnalyzer(min_support=0.01, min_confidence=0.3)
    analyzer.train(transactions)
    
    stats = analyzer.get_statistics()
    assert 'num_frequent_itemsets' in stats
    assert 'num_rules' in stats


def test_anomaly_detector_training(sample_data):
    """Test anomaly detector training."""
    transactions, _ = sample_data
    
    detector = RefundAnomalyDetector()
    detector.train(transactions)
    
    assert detector.is_trained


def test_anomaly_detector_detection(sample_data):
    """Test anomaly detection."""
    transactions, _ = sample_data
    
    detector = RefundAnomalyDetector(contamination=0.1)
    detector.train(transactions)
    
    anomalies = detector.detect_anomalies(transactions)
    
    assert isinstance(anomalies, pd.DataFrame)
    if len(anomalies) > 0:
        assert 'is_anomaly' in anomalies.columns
        assert 'anomaly_score' in anomalies.columns


def test_anomaly_detector_statistics(sample_data):
    """Test refund statistics."""
    transactions, _ = sample_data
    
    detector = RefundAnomalyDetector()
    detector.train(transactions)
    
    stats = detector.get_refund_statistics(transactions)
    
    assert 'total_transactions' in stats
    assert 'refund_transactions' in stats
    assert 'refund_rate' in stats


def test_llm_query_mock_mode(sample_data):
    """Test LLM query engine in mock mode."""
    transactions, _ = sample_data
    
    engine = LLMQueryEngine(use_mock=True)
    
    assert engine.use_mock
    
    # Test simple query
    response = engine.query(transactions, "What were the top 5 products by revenue?")
    
    assert 'success' in response
    assert 'code' in response
    assert 'method' in response
    assert response['method'] == 'mock'


def test_llm_query_various_questions(sample_data):
    """Test various question patterns."""
    transactions, _ = sample_data
    
    engine = LLMQueryEngine(use_mock=True)
    
    questions = [
        "What were the top 5 products by revenue?",
        "Show me sales by hour",
        "How many refunds did we have?",
        "What was the average transaction value?",
    ]
    
    for question in questions:
        response = engine.query(transactions, question)
        assert 'success' in response
        assert 'code' in response


def test_forecaster_evaluation(sample_data):
    """Test forecaster evaluation metrics."""
    transactions, _ = sample_data
    
    # Split data
    split_date = pd.Timestamp(transactions['date'].min()) + pd.Timedelta(days=20)
    train_df = transactions[transactions['timestamp'] < split_date]
    test_df = transactions[transactions['timestamp'] >= split_date]
    
    if len(test_df) > 0:
        forecaster = SalesForecaster()
        forecaster.train(train_df)
        
        metrics = forecaster.evaluate(test_df)
        
        assert 'hourly_mae' in metrics
        assert 'daily_mae' in metrics
        assert metrics['hourly_mae'] >= 0
        assert metrics['daily_mae'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
