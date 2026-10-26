"""
Synthetic POS Data Generator for Maple Corner Café
Generates realistic transaction data with temporal patterns, basket correlations, and staff assignments.
All prices in GBP (£).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class POSDataGenerator:
    """Generate synthetic POS transaction data for a fictional café."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Fictional café: Maple Corner Café
        self.store_name = "Maple Corner Café"
        
        # Product catalogue with categories and prices (UK pricing in GBP)
        self.products = {
            # Hot Beverages
            'Espresso': {'category': 'Hot Beverage', 'price': 2.80, 'cost': 0.65},
            'Americano': {'category': 'Hot Beverage', 'price': 3.00, 'cost': 0.70},
            'Flat White': {'category': 'Hot Beverage', 'price': 3.50, 'cost': 0.95},
            'Latte': {'category': 'Hot Beverage', 'price': 3.60, 'cost': 0.98},
            'Cappuccino': {'category': 'Hot Beverage', 'price': 3.50, 'cost': 0.95},
            'Mocha': {'category': 'Hot Beverage', 'price': 3.95, 'cost': 1.20},
            'Hot Chocolate': {'category': 'Hot Beverage', 'price': 3.30, 'cost': 0.90},
            'English Breakfast Tea': {'category': 'Hot Beverage', 'price': 2.50, 'cost': 0.50},
            'Earl Grey Tea': {'category': 'Hot Beverage', 'price': 2.50, 'cost': 0.50},
            
            # Cold Beverages
            'Iced Coffee': {'category': 'Cold Beverage', 'price': 3.40, 'cost': 0.80},
            'Cold Brew': {'category': 'Cold Beverage', 'price': 3.80, 'cost': 1.05},
            'Iced Latte': {'category': 'Cold Beverage', 'price': 3.95, 'cost': 1.10},
            'Iced Tea': {'category': 'Cold Beverage', 'price': 2.80, 'cost': 0.55},
            'Fruit Smoothie': {'category': 'Cold Beverage', 'price': 4.50, 'cost': 1.60},
            
            # Pastries
            'Croissant': {'category': 'Pastry', 'price': 2.80, 'cost': 0.95},
            'Pain au Chocolat': {'category': 'Pastry', 'price': 2.95, 'cost': 1.00},
            'Blueberry Muffin': {'category': 'Pastry', 'price': 2.90, 'cost': 1.00},
            'Scone with Jam': {'category': 'Pastry', 'price': 3.20, 'cost': 1.10},
            'Cinnamon Bun': {'category': 'Pastry', 'price': 3.50, 'cost': 1.20},
            
            # Food
            'Cheese Toastie': {'category': 'Food', 'price': 5.50, 'cost': 2.20},
            'Bacon Butty': {'category': 'Food', 'price': 4.95, 'cost': 2.00},
            'Ploughman\'s Sandwich': {'category': 'Food', 'price': 6.50, 'cost': 2.80},
            'Coronation Chicken Wrap': {'category': 'Food', 'price': 6.95, 'cost': 2.95},
            'Jacket Potato with Beans': {'category': 'Food', 'price': 5.75, 'cost': 2.10},
            'Seasonal Salad': {'category': 'Food', 'price': 7.25, 'cost': 3.10},
            
            # Snacks
            'Shortbread Biscuit': {'category': 'Snack', 'price': 1.80, 'cost': 0.60},
            'Chocolate Brownie': {'category': 'Snack', 'price': 2.95, 'cost': 1.00},
            'Flapjack': {'category': 'Snack', 'price': 2.50, 'cost': 0.85},
            'Crisps': {'category': 'Snack', 'price': 1.50, 'cost': 0.55},
        }
        
        # Staff (fictional names)
        self.staff = [
            {'id': 'S001', 'name': 'Alex River', 'shift': 'morning'},
            {'id': 'S002', 'name': 'Jordan Sky', 'shift': 'morning'},
            {'id': 'S003', 'name': 'Casey Moon', 'shift': 'afternoon'},
            {'id': 'S004', 'name': 'Morgan Sun', 'shift': 'afternoon'},
            {'id': 'S005', 'name': 'Taylor Star', 'shift': 'evening'},
            {'id': 'S006', 'name': 'Riley Cloud', 'shift': 'evening'},
        ]
        
        # Market basket associations (products that go together)
        self.basket_rules = {
            'Latte': ['Croissant', 'Blueberry Muffin', 'Shortbread Biscuit'],
            'Flat White': ['Croissant', 'Pain au Chocolat'],
            'Cappuccino': ['Blueberry Muffin', 'Scone with Jam', 'Flapjack'],
            'Espresso': ['Shortbread Biscuit', 'Chocolate Brownie'],
            'English Breakfast Tea': ['Scone with Jam', 'Shortbread Biscuit'],
            'Cold Brew': ['Chocolate Brownie', 'Flapjack'],
            'Bacon Butty': ['Americano', 'English Breakfast Tea'],
            'Ploughman\'s Sandwich': ['Iced Tea', 'Crisps'],
            'Coronation Chicken Wrap': ['Iced Tea', 'Fruit Smoothie'],
            'Seasonal Salad': ['Iced Tea', 'Iced Latte'],
        }
    
    def _get_hourly_traffic_multiplier(self, hour: int, day_of_week: int) -> float:
        """
        Get traffic multiplier based on hour of day and day of week.
        
        Args:
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            Traffic multiplier (0.1 to 3.0)
        """
        # Base pattern: morning rush, lunch peak, afternoon slowdown
        hourly_pattern = {
            6: 0.3, 7: 1.5, 8: 2.5, 9: 2.0, 10: 1.2,
            11: 1.8, 12: 2.2, 13: 1.8, 14: 1.0, 15: 0.9,
            16: 1.1, 17: 1.3, 18: 0.8, 19: 0.5, 20: 0.3
        }
        base = hourly_pattern.get(hour, 0.1)
        
        # Weekly pattern: weekends busier, Monday slowest
        weekly_multiplier = {
            0: 0.7,   # Monday
            1: 0.85,  # Tuesday
            2: 0.9,   # Wednesday
            3: 0.95,  # Thursday
            4: 1.0,   # Friday
            5: 1.3,   # Saturday
            6: 1.25,  # Sunday
        }
        
        return base * weekly_multiplier.get(day_of_week, 1.0)
    
    def _get_staff_for_time(self, hour: int) -> dict:
        """Get appropriate staff member for a given hour."""
        if 6 <= hour < 12:
            shift_staff = [s for s in self.staff if s['shift'] == 'morning']
        elif 12 <= hour < 18:
            shift_staff = [s for s in self.staff if s['shift'] == 'afternoon']
        else:
            shift_staff = [s for s in self.staff if s['shift'] == 'evening']
        
        return np.random.choice(shift_staff) if shift_staff else self.staff[0]
    
    def _generate_basket(self, hour: int) -> List[Tuple[str, int]]:
        """
        Generate a realistic shopping basket based on time of day.
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            List of (product_name, quantity) tuples
        """
        basket = []
        
        # Time-based primary product selection
        if 6 <= hour < 10:  # Morning: hot drinks + breakfast
            beverage_probs = {'Latte': 0.2, 'Flat White': 0.2, 'Cappuccino': 0.15, 
                            'Americano': 0.15, 'English Breakfast Tea': 0.15, 
                            'Espresso': 0.08, 'Hot Chocolate': 0.07}
            food_probs = {'Croissant': 0.25, 'Pain au Chocolat': 0.2, 'Blueberry Muffin': 0.2,
                         'Bacon Butty': 0.2, 'Scone with Jam': 0.15}
        elif 10 <= hour < 14:  # Midday: varied drinks + lunch items
            beverage_probs = {'Iced Coffee': 0.18, 'Latte': 0.15, 'Cold Brew': 0.15,
                            'Iced Tea': 0.15, 'Cappuccino': 0.1, 'Fruit Smoothie': 0.12,
                            'Americano': 0.1, 'English Breakfast Tea': 0.05}
            food_probs = {'Ploughman\'s Sandwich': 0.25, 'Coronation Chicken Wrap': 0.2, 
                         'Seasonal Salad': 0.2, 'Cheese Toastie': 0.18,
                         'Jacket Potato with Beans': 0.17}
        else:  # Afternoon/evening: cold drinks + snacks
            beverage_probs = {'Iced Coffee': 0.2, 'Cold Brew': 0.18, 'Iced Latte': 0.15,
                            'Latte': 0.15, 'English Breakfast Tea': 0.12, 
                            'Fruit Smoothie': 0.1, 'Espresso': 0.1}
            food_probs = {'Shortbread Biscuit': 0.25, 'Chocolate Brownie': 0.2, 
                         'Blueberry Muffin': 0.2, 'Flapjack': 0.18, 
                         'Croissant': 0.12, 'Cinnamon Bun': 0.05}
        
        # Select primary beverage
        beverage = np.random.choice(list(beverage_probs.keys()), 
                                   p=list(beverage_probs.values()))
        basket.append((beverage, 1))
        
        # 60% chance of adding food/snack
        if np.random.random() < 0.6:
            food = np.random.choice(list(food_probs.keys()), 
                                   p=list(food_probs.values()))
            basket.append((food, 1))
            
            # Check for basket associations (upsell opportunity)
            if beverage in self.basket_rules:
                associated_items = self.basket_rules[beverage]
                if food in associated_items:
                    pass
                elif np.random.random() < 0.3:  # 30% chance to add associated item
                    extra = np.random.choice(associated_items)
                    basket.append((extra, 1))
        
        # 10% chance of buying multiple of the same beverage
        if np.random.random() < 0.1:
            quantity = np.random.choice([2, 3], p=[0.8, 0.2])
            basket[0] = (basket[0][0], quantity)
        
        return basket
    
    def _should_generate_refund(self, transaction_count: int) -> bool:
        """Determine if this transaction should be a refund (~1.5% rate)."""
        return np.random.random() < 0.015
    
    def _inject_anomalous_refund(self) -> bool:
        """Occasionally inject an anomalous refund (very rare: ~0.1%)."""
        return np.random.random() < 0.001
    
    def generate_transactions(self, 
                            start_date: str = "2024-01-01",
                            num_days: int = 90) -> pd.DataFrame:
        """
        Generate synthetic POS transactions for the specified period.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            num_days: Number of days to generate
            
        Returns:
            DataFrame with transaction data
        """
        transactions = []
        transaction_id = 1
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        
        for day_offset in range(num_days):
            current_date = start + timedelta(days=day_offset)
            day_of_week = current_date.weekday()
            
            # Operating hours: 6 AM to 8 PM
            for hour in range(6, 21):
                # Calculate expected transactions this hour
                traffic_multiplier = self._get_hourly_traffic_multiplier(hour, day_of_week)
                base_transactions_per_hour = 15
                num_transactions = int(np.random.poisson(
                    base_transactions_per_hour * traffic_multiplier
                ))
                
                for _ in range(num_transactions):
                    # Random minute within the hour
                    minute = np.random.randint(0, 60)
                    timestamp = current_date.replace(hour=hour, minute=minute, second=0)
                    
                    # Get staff member
                    staff = self._get_staff_for_time(hour)
                    
                    # Generate basket
                    basket = self._generate_basket(hour)
                    
                    # Check if this should be a refund
                    is_refund = self._should_generate_refund(transaction_id)
                    is_anomalous = False
                    
                    if is_refund:
                        # Normal refund: pick a random past item
                        if np.random.random() < 0.2:  # 20% of refunds are anomalous
                            is_anomalous = True
                            # Anomalous: very large amount or odd timing
                            basket = [('Espresso', np.random.randint(5, 15))]  # Unusual quantity
                    
                    # Calculate total
                    total = sum(self.products[item]['price'] * qty 
                              for item, qty in basket)
                    
                    if is_refund:
                        total = -total
                    
                    # Payment method
                    payment_method = np.random.choice(['card', 'cash'], p=[0.75, 0.25])
                    
                    # Record each item in the basket
                    for product_name, quantity in basket:
                        transactions.append({
                            'transaction_id': f'TXN{transaction_id:06d}',
                            'timestamp': timestamp,
                            'date': timestamp.date(),
                            'hour': hour,
                            'day_of_week': day_of_week,
                            'day_name': timestamp.strftime('%A'),
                            'product_name': product_name,
                            'category': self.products[product_name]['category'],
                            'quantity': quantity if not is_refund else -quantity,
                            'unit_price': self.products[product_name]['price'],
                            'total': self.products[product_name]['price'] * quantity * (-1 if is_refund else 1),
                            'cost': self.products[product_name]['cost'] * quantity,
                            'staff_id': staff['id'],
                            'staff_name': staff['name'],
                            'payment_method': payment_method,
                            'is_refund': is_refund,
                            'is_anomalous_refund': is_anomalous,
                        })
                    
                    transaction_id += 1
        
        df = pd.DataFrame(transactions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def generate_inventory_data(self) -> pd.DataFrame:
        """Generate initial inventory levels and reorder points."""
        inventory = []
        
        for product_name, details in self.products.items():
            # Estimate daily demand based on category
            if details['category'] in ['Hot Beverage', 'Cold Beverage']:
                daily_demand = np.random.randint(30, 80)
            elif details['category'] in ['Pastry', 'Snack']:
                daily_demand = np.random.randint(15, 40)
            else:  # Food
                daily_demand = np.random.randint(10, 30)
            
            # Calculate reorder point (lead time = 2 days, safety stock = 1 day)
            reorder_point = daily_demand * 3
            current_stock = np.random.randint(reorder_point, reorder_point * 2)
            
            inventory.append({
                'product_name': product_name,
                'category': details['category'],
                'current_stock': current_stock,
                'reorder_point': reorder_point,
                'estimated_daily_demand': daily_demand,
                'unit_cost': details['cost'],
                'unit_price': details['price'],
            })
        
        return pd.DataFrame(inventory)


def generate_pos_data(start_date: str = "2024-01-01", 
                     num_days: int = 90,
                     random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to generate POS transaction and inventory data.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        num_days: Number of days to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (transactions_df, inventory_df)
    """
    generator = POSDataGenerator(random_seed=random_seed)
    transactions = generator.generate_transactions(start_date, num_days)
    inventory = generator.generate_inventory_data()
    
    return transactions, inventory


if __name__ == "__main__":
    # Generate sample data
    print("Generating synthetic POS data for Maple Corner Café...")
    transactions, inventory = generate_pos_data(num_days=90)
    
    print(f"\nGenerated {len(transactions)} transaction line items")
    print(f"Unique transactions: {transactions['transaction_id'].nunique()}")
    print(f"Date range: {transactions['date'].min()} to {transactions['date'].max()}")
    print(f"\nSample transactions:")
    print(transactions.head(10))
    
    print(f"\nInventory items: {len(inventory)}")
    print(inventory.head())
