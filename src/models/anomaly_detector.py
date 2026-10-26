"""
Anomaly Detection Module
Detects unusual refund patterns that might indicate fraud or operational issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder


class RefundAnomalyDetector:
    """Anomaly detector for identifying unusual refund patterns."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.1 = 10%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.staff_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_cols = []
    
    def train(self, transactions_df: pd.DataFrame):
        """
        Train the anomaly detector on refund transactions.
        
        Args:
            transactions_df: DataFrame with transaction data
        """
        df = transactions_df.copy()
        
        # Filter to refunds only
        refunds = df[df['is_refund'] == True].copy()
        
        if len(refunds) == 0:
            print("Warning: No refunds found in training data")
            self.is_trained = False
            return
        
        # Engineer features for anomaly detection
        refunds['abs_total'] = refunds['total'].abs()
        refunds['hour'] = pd.to_datetime(refunds['timestamp']).dt.hour
        
        # Encode staff_id
        self.staff_encoder.fit(df['staff_id'].unique())
        refunds['staff_id_encoded'] = self.staff_encoder.transform(refunds['staff_id'])
        
        # Count refunds per staff member
        staff_refund_counts = refunds.groupby('staff_id').size()
        refunds['staff_refund_count'] = refunds['staff_id'].map(staff_refund_counts)
        
        # Time-based features
        refunds['is_late_night'] = (refunds['hour'] < 6) | (refunds['hour'] > 20)
        refunds['is_early_morning'] = (refunds['hour'] >= 6) & (refunds['hour'] < 9)
        
        # Select features for anomaly detection
        self.feature_cols = [
            'abs_total',
            'hour',
            'staff_id_encoded',
            'staff_refund_count',
            'is_late_night',
            'is_early_morning'
        ]
        
        X = refunds[self.feature_cols].values
        
        # Train isolation forest
        self.model.fit(X)
        self.is_trained = True
    
    def detect_anomalies(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalous refunds in the given transactions.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            DataFrame with refunds flagged as anomalous or normal
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained before detecting anomalies")
        
        df = transactions_df.copy()
        
        # Filter to refunds
        refunds = df[df['is_refund'] == True].copy()
        
        if len(refunds) == 0:
            return pd.DataFrame(columns=df.columns.tolist() + ['is_anomaly', 'anomaly_score'])
        
        # Engineer same features
        refunds['abs_total'] = refunds['total'].abs()
        refunds['hour'] = pd.to_datetime(refunds['timestamp']).dt.hour
        
        # Encode staff_id (handle unknown staff)
        staff_mapping = {staff: idx for idx, staff in enumerate(self.staff_encoder.classes_)}
        refunds['staff_id_encoded'] = refunds['staff_id'].map(staff_mapping).fillna(-1).astype(int)
        
        # Count refunds per staff
        staff_refund_counts = refunds.groupby('staff_id').size()
        refunds['staff_refund_count'] = refunds['staff_id'].map(staff_refund_counts)
        
        # Time features
        refunds['is_late_night'] = (refunds['hour'] < 6) | (refunds['hour'] > 20)
        refunds['is_early_morning'] = (refunds['hour'] >= 6) & (refunds['hour'] < 9)
        
        # Prepare features
        X = refunds[self.feature_cols].values
        
        # Predict anomalies (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        refunds['is_anomaly'] = predictions == -1
        refunds['anomaly_score'] = scores
        
        return refunds
    
    def get_anomalous_refunds(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get only the anomalous refunds with relevant details.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            DataFrame with anomalous refunds and explanation
        """
        anomalies = self.detect_anomalies(transactions_df)
        
        if len(anomalies) == 0:
            return pd.DataFrame()
        
        anomalous = anomalies[anomalies['is_anomaly'] == True].copy()
        
        if len(anomalous) == 0:
            return pd.DataFrame()
        
        # Add explanation
        def explain_anomaly(row):
            reasons = []
            
            if row['abs_total'] > anomalies['abs_total'].quantile(0.9):
                reasons.append(f"Unusually large amount (${row['abs_total']:.2f})")
            
            if row['is_late_night']:
                reasons.append(f"Late night refund (hour {row['hour']})")
            
            if row['staff_refund_count'] > anomalies['staff_refund_count'].quantile(0.75):
                reasons.append(f"Staff has high refund count ({row['staff_refund_count']})")
            
            if not reasons:
                reasons.append("Pattern deviates from normal refund behavior")
            
            return "; ".join(reasons)
        
        anomalous['anomaly_explanation'] = anomalous.apply(explain_anomaly, axis=1)
        
        # Select relevant columns
        result_cols = [
            'transaction_id', 'timestamp', 'product_name', 'total', 
            'staff_id', 'staff_name', 'anomaly_score', 'anomaly_explanation'
        ]
        
        return anomalous[result_cols].sort_values('anomaly_score')
    
    def get_refund_statistics(self, transactions_df: pd.DataFrame) -> Dict:
        """
        Get statistics about refunds and anomalies.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            Dictionary with refund statistics
        """
        df = transactions_df.copy()
        
        total_transactions = df['transaction_id'].nunique()
        refund_transactions = df[df['is_refund'] == True]['transaction_id'].nunique()
        refund_rate = refund_transactions / total_transactions if total_transactions > 0 else 0
        
        refund_amount = df[df['is_refund'] == True]['total'].sum()
        total_sales = df[df['is_refund'] == False]['total'].sum()
        refund_amount_rate = abs(refund_amount) / total_sales if total_sales > 0 else 0
        
        stats = {
            'total_transactions': total_transactions,
            'refund_transactions': refund_transactions,
            'refund_rate': refund_rate,
            'total_refund_amount': abs(refund_amount),
            'refund_amount_rate': refund_amount_rate,
        }
        
        if self.is_trained:
            anomalies = self.detect_anomalies(df)
            if len(anomalies) > 0:
                num_anomalies = anomalies['is_anomaly'].sum()
                anomaly_rate = num_anomalies / len(anomalies) if len(anomalies) > 0 else 0
                
                stats['anomalous_refunds'] = int(num_anomalies)
                stats['anomaly_rate'] = anomaly_rate
        
        return stats
    
    def get_staff_refund_analysis(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze refund patterns by staff member.
        
        Args:
            transactions_df: DataFrame with transaction data
            
        Returns:
            DataFrame with staff refund analysis
        """
        df = transactions_df.copy()
        
        # Total transactions per staff
        staff_totals = df.groupby(['staff_id', 'staff_name']).agg({
            'transaction_id': 'nunique',
            'total': 'sum'
        }).reset_index()
        staff_totals.columns = ['staff_id', 'staff_name', 'total_transactions', 'total_sales']
        
        # Refund transactions per staff
        refunds = df[df['is_refund'] == True]
        staff_refunds = refunds.groupby(['staff_id', 'staff_name']).agg({
            'transaction_id': 'nunique',
            'total': lambda x: abs(x.sum())
        }).reset_index()
        staff_refunds.columns = ['staff_id', 'staff_name', 'refund_transactions', 'refund_amount']
        
        # Merge
        staff_analysis = staff_totals.merge(staff_refunds, on=['staff_id', 'staff_name'], how='left')
        staff_analysis = staff_analysis.fillna(0)
        
        # Calculate rates
        staff_analysis['refund_rate'] = (
            staff_analysis['refund_transactions'] / staff_analysis['total_transactions']
        )
        staff_analysis['refund_amount_rate'] = (
            staff_analysis['refund_amount'] / staff_analysis['total_sales']
        )
        
        # If trained, add anomaly counts
        if self.is_trained:
            anomalies = self.detect_anomalies(df)
            if len(anomalies) > 0:
                anomalous = anomalies[anomalies['is_anomaly'] == True]
                staff_anomalies = anomalous.groupby('staff_id').size().reset_index()
                staff_anomalies.columns = ['staff_id', 'anomalous_refunds']
                
                staff_analysis = staff_analysis.merge(staff_anomalies, on='staff_id', how='left')
                staff_analysis['anomalous_refunds'] = staff_analysis['anomalous_refunds'].fillna(0).astype(int)
        
        staff_analysis = staff_analysis.sort_values('refund_rate', ascending=False)
        
        return staff_analysis


if __name__ == "__main__":
    # Demo usage
    from src.data.generator import generate_pos_data
    
    print("Generating synthetic data...")
    transactions, _ = generate_pos_data(num_days=90)
    
    print(f"\nTraining anomaly detector on {len(transactions)} transactions...")
    detector = RefundAnomalyDetector(contamination=0.1)
    detector.train(transactions)
    
    print("\nRefund Statistics:")
    stats = detector.get_refund_statistics(transactions)
    for key, value in stats.items():
        if 'rate' in key:
            print(f"  {key}: {value:.2%}")
        elif 'amount' in key:
            print(f"  {key}: ${value:,.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nStaff Refund Analysis:")
    staff_analysis = detector.get_staff_refund_analysis(transactions)
    print(staff_analysis.to_string(index=False))
    
    print("\nAnomalous Refunds Detected:")
    anomalous = detector.get_anomalous_refunds(transactions)
    if len(anomalous) > 0:
        print(f"Found {len(anomalous)} anomalous refunds:")
        print(anomalous.to_string(index=False))
    else:
        print("  No anomalies detected")
