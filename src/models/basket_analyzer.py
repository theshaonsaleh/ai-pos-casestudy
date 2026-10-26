"""
Market Basket Analysis Module
Discovers frequent item combinations and generates upsell recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


class BasketAnalyzer:
    """Market basket analysis using association rule mining."""
    
    def __init__(self, min_support: float = 0.05, min_confidence: float = 0.5, min_lift: float = 1.0):
        """
        Initialize the basket analyzer.
        
        Args:
            min_support: Minimum support threshold for frequent itemsets
            min_confidence: Minimum confidence threshold for rules
            min_lift: Minimum lift threshold (1.0 = no correlation)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = None
        self.frequent_itemsets = None
        self.is_trained = False
    
    def train(self, transactions_df: pd.DataFrame):
        """
        Train the basket analyzer on transaction data.
        
        Args:
            transactions_df: DataFrame with transaction_id and product_name columns
        """
        df = transactions_df.copy()
        
        # Filter out refunds for basket analysis
        df = df[df['is_refund'] == False]
        
        # Create basket format: transaction_id -> list of products
        baskets = df.groupby('transaction_id')['product_name'].apply(list).values
        
        # One-hot encode baskets
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Find frequent itemsets
        self.frequent_itemsets = apriori(basket_df, 
                                         min_support=self.min_support, 
                                         use_colnames=True)
        
        if len(self.frequent_itemsets) == 0:
            print(f"Warning: No frequent itemsets found with min_support={self.min_support}")
            self.rules = pd.DataFrame()
            self.is_trained = True
            return
        
        # Generate association rules
        try:
            self.rules = association_rules(self.frequent_itemsets, 
                                          metric="confidence", 
                                          min_threshold=self.min_confidence)
            
            # Filter by lift
            self.rules = self.rules[self.rules['lift'] >= self.min_lift]
            
            # Sort by lift (strongest associations first)
            self.rules = self.rules.sort_values('lift', ascending=False)
            
        except ValueError:
            # No rules generated
            print(f"Warning: No association rules found with current thresholds")
            self.rules = pd.DataFrame()
        
        self.is_trained = True
    
    def get_upsell_recommendations(self, current_basket: List[str], top_n: int = 3) -> pd.DataFrame:
        """
        Get product recommendations based on current basket contents.
        
        Args:
            current_basket: List of product names currently in the basket
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended products and their association metrics
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before generating recommendations")
        
        if self.rules.empty:
            return pd.DataFrame(columns=['recommended_product', 'confidence', 'lift', 'support'])
        
        recommendations = []
        
        for product in current_basket:
            # Find rules where this product is in the antecedent
            matching_rules = self.rules[
                self.rules['antecedents'].apply(lambda x: product in x)
            ]
            
            for _, rule in matching_rules.iterrows():
                # Get consequent items not already in basket
                consequent_items = list(rule['consequents'])
                for item in consequent_items:
                    if item not in current_basket:
                        recommendations.append({
                            'recommended_product': item,
                            'because_of': product,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'support': rule['support']
                        })
        
        if not recommendations:
            return pd.DataFrame(columns=['recommended_product', 'confidence', 'lift', 'support'])
        
        # Convert to DataFrame and aggregate by recommended product
        rec_df = pd.DataFrame(recommendations)
        
        # If multiple rules lead to the same recommendation, take the strongest
        rec_df = rec_df.sort_values('lift', ascending=False).drop_duplicates(
            subset=['recommended_product'], keep='first'
        )
        
        return rec_df.head(top_n)[['recommended_product', 'because_of', 'confidence', 'lift', 'support']]
    
    def get_top_rules(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get the top association rules by lift.
        
        Args:
            top_n: Number of rules to return
            
        Returns:
            DataFrame with top rules
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before retrieving rules")
        
        if self.rules.empty:
            return pd.DataFrame()
        
        # Format for readability
        top_rules = self.rules.head(top_n).copy()
        top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        return top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    
    def get_product_affinities(self, product: str, top_n: int = 5) -> pd.DataFrame:
        """
        Get products with highest affinity to a given product.
        
        Args:
            product: Product name to analyze
            top_n: Number of affiliated products to return
            
        Returns:
            DataFrame with affiliated products and metrics
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before analysis")
        
        if self.rules.empty:
            return pd.DataFrame()
        
        # Rules where product is in antecedent
        forward_rules = self.rules[
            self.rules['antecedents'].apply(lambda x: product in x)
        ].copy()
        
        # Rules where product is in consequent
        backward_rules = self.rules[
            self.rules['consequents'].apply(lambda x: product in x)
        ].copy()
        
        affinities = []
        
        # Forward: product -> other
        for _, rule in forward_rules.iterrows():
            for item in rule['consequents']:
                affinities.append({
                    'affiliated_product': item,
                    'direction': f'{product} → {item}',
                    'lift': rule['lift'],
                    'confidence': rule['confidence'],
                    'support': rule['support']
                })
        
        # Backward: other -> product
        for _, rule in backward_rules.iterrows():
            for item in rule['antecedents']:
                affinities.append({
                    'affiliated_product': item,
                    'direction': f'{item} → {product}',
                    'lift': rule['lift'],
                    'confidence': rule['confidence'],
                    'support': rule['support']
                })
        
        if not affinities:
            return pd.DataFrame()
        
        aff_df = pd.DataFrame(affinities)
        aff_df = aff_df.sort_values('lift', ascending=False).drop_duplicates(
            subset=['affiliated_product'], keep='first'
        )
        
        return aff_df.head(top_n)
    
    def get_frequent_pairs(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get most frequent product pairs.
        
        Args:
            top_n: Number of pairs to return
            
        Returns:
            DataFrame with frequent pairs
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before analysis")
        
        if self.frequent_itemsets.empty:
            return pd.DataFrame()
        
        # Filter to pairs only (length = 2)
        pairs = self.frequent_itemsets[
            self.frequent_itemsets['itemsets'].apply(lambda x: len(x) == 2)
        ].copy()
        
        if pairs.empty:
            return pd.DataFrame()
        
        pairs['product_1'] = pairs['itemsets'].apply(lambda x: list(x)[0])
        pairs['product_2'] = pairs['itemsets'].apply(lambda x: list(x)[1])
        pairs = pairs[['product_1', 'product_2', 'support']].sort_values(
            'support', ascending=False
        ).head(top_n)
        
        return pairs.reset_index(drop=True)
    
    def get_statistics(self) -> Dict:
        """
        Get summary statistics about the analysis.
        
        Returns:
            Dictionary with statistics
        """
        if not self.is_trained:
            raise ValueError("Analyzer must be trained before retrieving statistics")
        
        return {
            'num_frequent_itemsets': len(self.frequent_itemsets) if self.frequent_itemsets is not None else 0,
            'num_rules': len(self.rules) if self.rules is not None else 0,
            'min_support': self.min_support,
            'min_confidence': self.min_confidence,
            'min_lift': self.min_lift,
            'max_lift': self.rules['lift'].max() if not self.rules.empty else 0,
            'avg_confidence': self.rules['confidence'].mean() if not self.rules.empty else 0,
        }


if __name__ == "__main__":
    # Demo usage
    from src.data.generator import generate_pos_data
    
    print("Generating synthetic data...")
    transactions, _ = generate_pos_data(num_days=90)
    
    print(f"\nAnalyzing {transactions['transaction_id'].nunique()} transactions...")
    analyzer = BasketAnalyzer(min_support=0.05, min_confidence=0.5, min_lift=1.0)
    analyzer.train(transactions)
    
    print("\nAnalysis Statistics:")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nTop 10 Association Rules:")
    top_rules = analyzer.get_top_rules(top_n=10)
    if not top_rules.empty:
        print(top_rules.to_string(index=False))
    else:
        print("  No rules found")
    
    print("\nMost Frequent Product Pairs:")
    pairs = analyzer.get_frequent_pairs(top_n=10)
    if not pairs.empty:
        print(pairs.to_string(index=False))
    else:
        print("  No pairs found")
    
    print("\nUpsell Recommendations for ['Latte', 'Croissant']:")
    recs = analyzer.get_upsell_recommendations(['Latte', 'Croissant'], top_n=3)
    if not recs.empty:
        print(recs.to_string(index=False))
    else:
        print("  No recommendations found")
    
    print("\nProduct Affinities for 'Latte':")
    affinities = analyzer.get_product_affinities('Latte', top_n=5)
    if not affinities.empty:
        print(affinities.to_string(index=False))
    else:
        print("  No affinities found")
