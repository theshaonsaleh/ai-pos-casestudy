"""
AI-Powered POS Analytics Demo
Demonstrates forecasting, basket analysis, anomaly detection, and LLM queries.
All amounts in GBP (£).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data.generator import generate_pos_data
from src.models.forecaster import SalesForecaster
from src.models.basket_analyzer import BasketAnalyzer
from src.models.anomaly_detector import RefundAnomalyDetector
from src.models.llm_query import LLMQueryEngine

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_sales_forecasting(transactions: pd.DataFrame):
    """Demonstrate sales forecasting capabilities."""
    print_section("1. SALES FORECASTING")
    
    # Split data for training and testing
    split_date = pd.Timestamp('2024-03-01')
    train_df = transactions[transactions['timestamp'] < split_date]
    test_df = transactions[transactions['timestamp'] >= split_date]
    
    print(f"Training on {train_df['date'].nunique()} days of data...")
    print(f"Testing on {test_df['date'].nunique()} days of data...")
    
    # Train forecaster
    forecaster = SalesForecaster()
    forecaster.train(train_df)
    
    # Show peak hours
    print("\n📊 Peak Sales Hours:")
    peak_hours = forecaster.get_peak_hours(top_n=5)
    for idx, row in peak_hours.iterrows():
        print(f"  {int(row['hour']):02d}:00 - £{row['avg_sales']:,.2f} average sales")
    
    # Show busiest days
    print("\n📊 Busiest Days of Week:")
    busiest_days = forecaster.get_busiest_days()
    for idx, row in busiest_days.iterrows():
        print(f"  {row['day_name']:10s} - £{row['avg_sales']:,.2f} average sales")
    
    # Evaluate on test set
    print("\n📈 Forecast Performance on Test Set:")
    metrics = forecaster.evaluate(test_df)
    print(f"  Hourly Forecast:")
    print(f"    MAE:  £{metrics['hourly_mae']:.2f}")
    print(f"    RMSE: £{metrics['hourly_rmse']:.2f}")
    print(f"    MAPE: {metrics['hourly_mape']:.1f}%")
    print(f"\n  Daily Forecast:")
    print(f"    MAE:  £{metrics['daily_mae']:.2f}")
    print(f"    RMSE: £{metrics['daily_rmse']:.2f}")
    print(f"    MAPE: {metrics['daily_mape']:.1f}%")
    
    # Forecast next 7 days
    print("\n🔮 7-Day Sales Forecast:")
    daily_forecast = forecaster.forecast_daily(num_days=7)
    for idx, row in daily_forecast.iterrows():
        print(f"  {row['day_name']:10s} - £{row['forecast_sales']:,.2f}")
    
    # Visualize hourly patterns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    peak_hours.plot(x='hour', y='avg_sales', kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('Average Sales by Hour', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Average Sales (£)')
    ax1.set_xticklabels([f"{h:02d}:00" for h in peak_hours['hour']], rotation=45)
    
    busiest_days.plot(x='day_name', y='avg_sales', kind='bar', ax=ax2, color='coral')
    ax2.set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Average Sales (£)')
    ax2.set_xticklabels(busiest_days['day_name'], rotation=45)
    
    plt.tight_layout()
    plt.savefig('forecast_patterns.png', dpi=100, bbox_inches='tight')
    print("\n✅ Saved visualization: forecast_patterns.png")


def demo_basket_analysis(transactions: pd.DataFrame):
    """Demonstrate market basket analysis."""
    print_section("2. MARKET BASKET ANALYSIS")
    
    print("Analyzing purchase patterns...")
    analyzer = BasketAnalyzer(min_support=0.02, min_confidence=0.3, min_lift=1.0)
    analyzer.train(transactions)
    
    # Statistics
    stats = analyzer.get_statistics()
    print(f"\n📊 Analysis Statistics:")
    print(f"  Frequent Itemsets: {stats['num_frequent_itemsets']}")
    print(f"  Association Rules: {stats['num_rules']}")
    if stats['num_rules'] > 0:
        print(f"  Max Lift: {stats['max_lift']:.2f}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2%}")
    
    # Top rules
    print(f"\n🔗 Top 10 Association Rules:")
    top_rules = analyzer.get_top_rules(top_n=10)
    if not top_rules.empty:
        for idx, row in top_rules.iterrows():
            print(f"  {row['antecedents']:20s} → {row['consequents']:20s}")
            print(f"    Support: {row['support']:.2%}, Confidence: {row['confidence']:.2%}, Lift: {row['lift']:.2f}")
    else:
        print("  No rules found with current thresholds")
    
    # Frequent pairs
    print(f"\n👥 Most Frequent Product Pairs:")
    pairs = analyzer.get_frequent_pairs(top_n=10)
    if not pairs.empty:
        for idx, row in pairs.iterrows():
            print(f"  {row['product_1']} + {row['product_2']} (support: {row['support']:.2%})")
    else:
        print("  No pairs found")
    
    # Upsell recommendations
    test_baskets = [
        ['Latte'],
        ['Latte', 'Croissant'],
        ['Turkey Sandwich'],
        ['Cold Brew'],
    ]
    
    print(f"\n💡 Upsell Recommendations:")
    for basket in test_baskets:
        print(f"\n  Current basket: {', '.join(basket)}")
        recs = analyzer.get_upsell_recommendations(basket, top_n=3)
        if not recs.empty:
            for idx, row in recs.iterrows():
                print(f"    → Suggest: {row['recommended_product']} "
                      f"(confidence: {row['confidence']:.1%}, lift: {row['lift']:.2f})")
        else:
            print(f"    → No recommendations available")
    
    # Visualize top pairs
    if not pairs.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        pairs_top5 = pairs.head(5).copy()
        pairs_top5['pair'] = pairs_top5['product_1'] + '\n+\n' + pairs_top5['product_2']
        
        ax.barh(range(len(pairs_top5)), pairs_top5['support'], color='teal')
        ax.set_yticks(range(len(pairs_top5)))
        ax.set_yticklabels(pairs_top5['pair'])
        ax.set_xlabel('Support (Frequency)')
        ax.set_title('Top 5 Product Pairs', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('basket_pairs.png', dpi=100, bbox_inches='tight')
        print("\n✅ Saved visualization: basket_pairs.png")


def demo_anomaly_detection(transactions: pd.DataFrame):
    """Demonstrate refund anomaly detection."""
    print_section("3. REFUND ANOMALY DETECTION")
    
    print("Training anomaly detector...")
    detector = RefundAnomalyDetector(contamination=0.1)
    detector.train(transactions)
    
    # Statistics
    stats = detector.get_refund_statistics(transactions)
    print(f"\n📊 Refund Statistics:")
    print(f"  Total Transactions: {stats['total_transactions']:,}")
    print(f"  Refund Transactions: {stats['refund_transactions']:,}")
    print(f"  Refund Rate: {stats['refund_rate']:.2%}")
    print(f"  Total Refund Amount: £{stats['total_refund_amount']:,.2f}")
    print(f"  Refund Amount Rate: {stats['refund_amount_rate']:.2%}")
    
    if 'anomalous_refunds' in stats:
        print(f"\n🚨 Anomalies Detected:")
        print(f"  Anomalous Refunds: {stats['anomalous_refunds']}")
        print(f"  Anomaly Rate: {stats['anomaly_rate']:.2%}")
    
    # Staff analysis
    print(f"\n👥 Staff Refund Analysis:")
    staff_analysis = detector.get_staff_refund_analysis(transactions)
    for idx, row in staff_analysis.iterrows():
        print(f"  {row['staff_name']:15s} - {int(row['refund_transactions']):3d} refunds "
              f"({row['refund_rate']:.1%} rate)", end='')
        if 'anomalous_refunds' in row:
            if row['anomalous_refunds'] > 0:
                print(f" ⚠️  {int(row['anomalous_refunds'])} anomalous")
            else:
                print()
        else:
            print()
    
    # Get anomalous refunds
    anomalous = detector.get_anomalous_refunds(transactions)
    if len(anomalous) > 0:
        print(f"\n🔍 Top 5 Anomalous Refunds:")
        for idx, row in anomalous.head(5).iterrows():
            print(f"\n  Transaction: {row['transaction_id']}")
            print(f"    Time: {row['timestamp']}")
            print(f"    Staff: {row['staff_name']}")
            print(f"    Amount: £{abs(row['total']):.2f}")
            print(f"    Reason: {row['anomaly_explanation']}")
    else:
        print("\n✅ No anomalous refunds detected")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Refund rate by staff
    staff_analysis_sorted = staff_analysis.sort_values('refund_rate', ascending=True)
    ax1.barh(staff_analysis_sorted['staff_name'], staff_analysis_sorted['refund_rate'], 
             color='indianred')
    ax1.set_xlabel('Refund Rate')
    ax1.set_title('Refund Rate by Staff', fontsize=14, fontweight='bold')
    ax1.axvline(stats['refund_rate'], color='black', linestyle='--', 
                label=f'Overall: {stats["refund_rate"]:.1%}')
    ax1.legend()
    
    # Anomalies over time
    if len(anomalous) > 0:
        anomalous_daily = anomalous.groupby(anomalous['timestamp'].dt.date).size()
        anomalous_daily.plot(kind='line', ax=ax2, color='red', marker='o')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Anomalous Refunds')
        ax2.set_title('Anomalous Refunds Over Time', fontsize=14, fontweight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No anomalies detected', 
                ha='center', va='center', fontsize=14)
        ax2.set_title('Anomalous Refunds Over Time', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=100, bbox_inches='tight')
    print("\n✅ Saved visualization: anomaly_detection.png")


def demo_llm_query(transactions: pd.DataFrame):
    """Demonstrate natural language query engine."""
    print_section("4. NATURAL LANGUAGE SALES Q&A")
    
    print("Initializing query engine (mock mode - no API key required)...")
    engine = LLMQueryEngine(use_mock=True)
    
    # Test questions
    test_questions = [
        "What were the top 5 products by revenue?",
        "Show me sales by hour of day",
        "Which staff member had the most transactions?",
        "Show me sales by category",
        "How many refunds did we have?",
    ]
    
    print("\n💬 Sample Questions & Answers:\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")
        print("-" * 60)
        
        response = engine.query(transactions, question)
        
        if response['success']:
            print(f"Code: {response['code']}")
            print(f"\nResult:")
            print(response['result_str'])
        else:
            print(f"Error: {response['error']}")
        
        if i < len(test_questions):
            print()


def main():
    """Run all demos."""
    print("\n" + "🍁" * 40)
    print("  Maple Corner Café - AI-Powered POS Analytics Demo")
    print("  Personal learning project - All data is fictional")
    print("🍁" * 40)
    
    # Generate data
    print("\n📦 Generating synthetic transaction data...")
    transactions, inventory = generate_pos_data(start_date="2024-01-01", num_days=90)
    
    print(f"✅ Generated {len(transactions):,} transaction line items")
    print(f"   Transactions: {transactions['transaction_id'].nunique():,}")
    print(f"   Date range: {transactions['date'].min()} to {transactions['date'].max()}")
    print(f"   Products: {transactions['product_name'].nunique()}")
    print(f"   Staff: {transactions['staff_id'].nunique()}")
    
    # Run demos
    demo_sales_forecasting(transactions)
    demo_basket_analysis(transactions)
    demo_anomaly_detection(transactions)
    demo_llm_query(transactions)
    
    # Summary
    print_section("DEMO COMPLETE")
    print("✅ All demonstrations completed successfully!")
    print("\n📊 Generated Files:")
    print("  - forecast_patterns.png    (hourly and daily sales patterns)")
    print("  - basket_pairs.png         (frequent product combinations)")
    print("  - anomaly_detection.png    (refund anomaly analysis)")
    print("\n📖 For more details, see CASE_STUDY.md")
    print("\n🚀 Try the interactive dashboard:")
    print("   streamlit run dashboard.py")
    print("\n" + "🍁" * 40 + "\n")


if __name__ == "__main__":
    main()
