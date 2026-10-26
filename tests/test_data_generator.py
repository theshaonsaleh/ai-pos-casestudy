"""
Tests for the synthetic data generator.
"""

import pytest
import pandas as pd
from src.data.generator import generate_pos_data, POSDataGenerator


def test_generate_pos_data():
    """Test basic data generation."""
    transactions, inventory = generate_pos_data(num_days=7, random_seed=42)
    
    assert isinstance(transactions, pd.DataFrame)
    assert isinstance(inventory, pd.DataFrame)
    assert len(transactions) > 0
    assert len(inventory) > 0


def test_transaction_columns():
    """Test that generated transactions have required columns."""
    transactions, _ = generate_pos_data(num_days=1, random_seed=42)
    
    required_cols = [
        'transaction_id', 'timestamp', 'date', 'hour', 'day_of_week',
        'product_name', 'category', 'quantity', 'unit_price', 'total',
        'staff_id', 'staff_name', 'payment_method', 'is_refund'
    ]
    
    for col in required_cols:
        assert col in transactions.columns, f"Missing column: {col}"


def test_refund_generation():
    """Test that refunds are generated correctly."""
    transactions, _ = generate_pos_data(num_days=30, random_seed=42)
    
    refunds = transactions[transactions['is_refund'] == True]
    
    assert len(refunds) > 0, "Should generate some refunds"
    assert all(refunds['total'] < 0), "Refunds should have negative totals"
    assert all(refunds['quantity'] < 0), "Refunds should have negative quantities"


def test_temporal_patterns():
    """Test that data has temporal patterns."""
    transactions, _ = generate_pos_data(num_days=7, random_seed=42)
    
    # Check operating hours
    hours = transactions['hour'].unique()
    assert min(hours) >= 6, "Should start at 6 AM"
    assert max(hours) <= 20, "Should end by 8 PM"
    
    # Check date range
    date_range = (transactions['date'].max() - transactions['date'].min()).days
    assert date_range >= 6, "Should cover at least 6 days"


def test_product_variety():
    """Test that multiple products are generated."""
    transactions, _ = generate_pos_data(num_days=7, random_seed=42)
    
    products = transactions['product_name'].unique()
    assert len(products) >= 10, "Should have variety of products"
    
    categories = transactions['category'].unique()
    assert len(categories) >= 3, "Should have multiple categories"


def test_staff_assignment():
    """Test that staff are assigned correctly."""
    transactions, _ = generate_pos_data(num_days=7, random_seed=42)
    
    staff = transactions['staff_id'].unique()
    assert len(staff) >= 3, "Should have multiple staff members"
    
    # Check staff names match IDs
    for staff_id in staff:
        staff_trans = transactions[transactions['staff_id'] == staff_id]
        assert staff_trans['staff_name'].nunique() == 1, "Each staff_id should have one name"


def test_inventory_generation():
    """Test inventory data generation."""
    _, inventory = generate_pos_data(num_days=1, random_seed=42)
    
    required_cols = [
        'product_name', 'category', 'current_stock', 'reorder_point',
        'estimated_daily_demand', 'unit_cost', 'unit_price'
    ]
    
    for col in required_cols:
        assert col in inventory.columns, f"Missing inventory column: {col}"
    
    assert all(inventory['current_stock'] > 0), "Stock should be positive"
    assert all(inventory['reorder_point'] > 0), "Reorder points should be positive"


def test_reproducibility():
    """Test that same seed produces same data."""
    trans1, inv1 = generate_pos_data(num_days=7, random_seed=42)
    trans2, inv2 = generate_pos_data(num_days=7, random_seed=42)
    
    assert len(trans1) == len(trans2), "Same seed should produce same number of transactions"
    assert trans1['transaction_id'].tolist() == trans2['transaction_id'].tolist()


def test_basket_correlations():
    """Test that basket items are generated together."""
    generator = POSDataGenerator(random_seed=42)
    transactions = generator.generate_transactions(num_days=30)
    
    # Group by transaction to see baskets
    baskets = transactions.groupby('transaction_id')['product_name'].apply(list)
    
    # Check that some transactions have multiple items
    multi_item_baskets = [b for b in baskets if len(b) > 1]
    assert len(multi_item_baskets) > 0, "Should generate multi-item baskets"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
