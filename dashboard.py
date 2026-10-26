"""
Interactive Streamlit Dashboard for AI-Powered POS Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.data.generator import generate_pos_data
from src.models.forecaster import SalesForecaster
from src.models.basket_analyzer import BasketAnalyzer
from src.models.anomaly_detector import RefundAnomalyDetector
from src.models.llm_query import LLMQueryEngine

# Page config
st.set_page_config(
    page_title="AI POS Analytics",
    page_icon="🍁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(num_days=90):
    """Load synthetic transaction data."""
    transactions, inventory = generate_pos_data(num_days=num_days)
    return transactions, inventory


@st.cache_resource
def train_models(transactions):
    """Train all ML models."""
    # Forecaster
    forecaster = SalesForecaster()
    forecaster.train(transactions)
    
    # Basket analyzer
    analyzer = BasketAnalyzer(min_support=0.05, min_confidence=0.5, min_lift=1.0)
    analyzer.train(transactions)
    
    # Anomaly detector
    detector = RefundAnomalyDetector(contamination=0.1)
    detector.train(transactions)
    
    return forecaster, analyzer, detector


def show_overview(transactions):
    """Show overview dashboard."""
    st.markdown('<div class="main-header">🍁 Maple Corner Café</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered POS Analytics Dashboard</div>', unsafe_allow_html=True)
    
    st.info("📌 **Personal learning project** - All data is fictional and generated for demonstration purposes.")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_sales = transactions[transactions['is_refund'] == False]['total'].sum()
    num_transactions = transactions['transaction_id'].nunique()
    avg_transaction = total_sales / num_transactions if num_transactions > 0 else 0
    num_products = transactions['product_name'].nunique()
    
    with col1:
        st.metric("Total Sales", f"${total_sales:,.2f}")
    with col2:
        st.metric("Transactions", f"{num_transactions:,}")
    with col3:
        st.metric("Avg Transaction", f"${avg_transaction:.2f}")
    with col4:
        st.metric("Products", num_products)
    
    # Sales over time
    st.subheader("📈 Sales Over Time")
    
    daily_sales = transactions[transactions['is_refund'] == False].groupby('date')['total'].sum().reset_index()
    fig = px.line(daily_sales, x='date', y='total', 
                  title='Daily Sales Trend',
                  labels={'date': 'Date', 'total': 'Sales ($)'})
    fig.update_traces(line_color='#1f77b4', line_width=2)
    st.plotly_chart(fig, use_container_width=True)
    
    # Category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Sales by Category")
        category_sales = transactions[transactions['is_refund'] == False].groupby('category')['total'].sum().reset_index()
        fig = px.pie(category_sales, values='total', names='category', 
                     title='Revenue by Category')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Top 10 Products")
        top_products = transactions[transactions['is_refund'] == False].groupby('product_name')['total'].sum().nlargest(10).reset_index()
        fig = px.bar(top_products, x='total', y='product_name', 
                     orientation='h',
                     title='Top Products by Revenue',
                     labels={'total': 'Revenue ($)', 'product_name': 'Product'})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def show_forecasting(transactions, forecaster):
    """Show sales forecasting dashboard."""
    st.header("🔮 Sales Forecasting")
    
    st.markdown("""
    Predict future sales based on historical patterns to optimize staffing and inventory.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Peak Hours")
        peak_hours = forecaster.get_peak_hours(top_n=10)
        peak_hours['hour_label'] = peak_hours['hour'].apply(lambda x: f"{x:02d}:00")
        fig = px.bar(peak_hours, x='hour_label', y='avg_sales',
                     title='Average Sales by Hour',
                     labels={'hour_label': 'Hour', 'avg_sales': 'Average Sales ($)'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(peak_hours[['hour_label', 'avg_sales']].rename(
            columns={'hour_label': 'Hour', 'avg_sales': 'Avg Sales ($)'}
        ), hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("📅 Busiest Days")
        busiest_days = forecaster.get_busiest_days()
        fig = px.bar(busiest_days, x='day_name', y='avg_sales',
                     title='Average Sales by Day of Week',
                     labels={'day_name': 'Day', 'avg_sales': 'Average Sales ($)'},
                     color='avg_sales',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(busiest_days[['day_name', 'avg_sales']].rename(
            columns={'day_name': 'Day', 'avg_sales': 'Avg Sales ($)'}
        ), hide_index=True, use_container_width=True)
    
    st.subheader("🔮 7-Day Forecast")
    forecast_df = forecaster.forecast_daily(num_days=7)
    
    fig = px.bar(forecast_df, x='day_name', y='forecast_sales',
                 title='Next 7 Days Sales Forecast',
                 labels={'day_name': 'Day', 'forecast_sales': 'Forecast Sales ($)'},
                 color='forecast_sales',
                 color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(forecast_df[['day_name', 'date', 'forecast_sales']].rename(
        columns={'day_name': 'Day', 'date': 'Date', 'forecast_sales': 'Forecast ($)'}
    ), hide_index=True, use_container_width=True)


def show_basket_analysis(transactions, analyzer):
    """Show market basket analysis dashboard."""
    st.header("🛒 Market Basket Analysis")
    
    st.markdown("""
    Discover which products are frequently bought together to optimize upselling and product placement.
    """)
    
    stats = analyzer.get_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Frequent Itemsets", stats['num_frequent_itemsets'])
    with col2:
        st.metric("Association Rules", stats['num_rules'])
    with col3:
        if stats['num_rules'] > 0:
            st.metric("Max Lift", f"{stats['max_lift']:.2f}")
    
    # Top rules
    st.subheader("🔗 Top Association Rules")
    top_rules = analyzer.get_top_rules(top_n=10)
    
    if not top_rules.empty:
        display_rules = top_rules.copy()
        display_rules['support'] = display_rules['support'].apply(lambda x: f"{x:.2%}")
        display_rules['confidence'] = display_rules['confidence'].apply(lambda x: f"{x:.2%}")
        display_rules['lift'] = display_rules['lift'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_rules.rename(columns={
            'antecedents': 'If Customer Buys',
            'consequents': 'Then Suggest',
            'support': 'Support',
            'confidence': 'Confidence',
            'lift': 'Lift'
        }), hide_index=True, use_container_width=True)
    else:
        st.info("No association rules found with current thresholds.")
    
    # Frequent pairs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Frequent Product Pairs")
        pairs = analyzer.get_frequent_pairs(top_n=10)
        if not pairs.empty:
            pairs['support_pct'] = pairs['support'].apply(lambda x: f"{x:.2%}")
            st.dataframe(pairs[['product_1', 'product_2', 'support_pct']].rename(columns={
                'product_1': 'Product 1',
                'product_2': 'Product 2',
                'support_pct': 'Frequency'
            }), hide_index=True, use_container_width=True)
        else:
            st.info("No frequent pairs found.")
    
    with col2:
        st.subheader("💡 Upsell Simulator")
        
        # Get all products for selection
        products = sorted(transactions['product_name'].unique())
        selected_products = st.multiselect(
            "Select products in current basket:",
            options=products,
            default=['Latte'] if 'Latte' in products else []
        )
        
        if selected_products:
            recs = analyzer.get_upsell_recommendations(selected_products, top_n=5)
            if not recs.empty:
                st.success("**Recommended upsells:**")
                for idx, row in recs.iterrows():
                    st.write(f"• **{row['recommended_product']}** "
                            f"(because of {row['because_of']}) - "
                            f"Confidence: {row['confidence']:.1%}, "
                            f"Lift: {row['lift']:.2f}")
            else:
                st.warning("No recommendations available for this basket.")
        else:
            st.info("Select products to see upsell recommendations.")


def show_anomaly_detection(transactions, detector):
    """Show anomaly detection dashboard."""
    st.header("🚨 Refund Anomaly Detection")
    
    st.markdown("""
    Identify unusual refund patterns that might indicate fraud, training issues, or system problems.
    """)
    
    stats = detector.get_refund_statistics(transactions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Refund Rate", f"{stats['refund_rate']:.2%}")
    with col2:
        st.metric("Total Refunds", stats['refund_transactions'])
    with col3:
        st.metric("Refund Amount", f"${stats['total_refund_amount']:,.2f}")
    with col4:
        if 'anomalous_refunds' in stats:
            st.metric("Anomalies", stats['anomalous_refunds'])
    
    # Staff analysis
    st.subheader("👥 Staff Refund Analysis")
    staff_analysis = detector.get_staff_refund_analysis(transactions)
    
    fig = px.bar(staff_analysis.sort_values('refund_rate'), 
                 x='refund_rate', y='staff_name',
                 orientation='h',
                 title='Refund Rate by Staff Member',
                 labels={'refund_rate': 'Refund Rate', 'staff_name': 'Staff'})
    fig.add_vline(x=stats['refund_rate'], line_dash="dash", 
                  annotation_text=f"Overall: {stats['refund_rate']:.2%}")
    st.plotly_chart(fig, use_container_width=True)
    
    display_staff = staff_analysis.copy()
    display_staff['refund_rate'] = display_staff['refund_rate'].apply(lambda x: f"{x:.2%}")
    display_staff['refund_amount'] = display_staff['refund_amount'].apply(lambda x: f"${x:.2f}")
    
    st.dataframe(display_staff[[
        'staff_name', 'total_transactions', 'refund_transactions', 
        'refund_rate', 'refund_amount'
    ]].rename(columns={
        'staff_name': 'Staff',
        'total_transactions': 'Total Trans.',
        'refund_transactions': 'Refunds',
        'refund_rate': 'Refund Rate',
        'refund_amount': 'Refund Amount'
    }), hide_index=True, use_container_width=True)
    
    # Anomalous refunds
    st.subheader("🔍 Anomalous Refunds")
    anomalous = detector.get_anomalous_refunds(transactions)
    
    if len(anomalous) > 0:
        st.warning(f"Found {len(anomalous)} anomalous refund(s) requiring review:")
        
        display_anomalous = anomalous.copy()
        display_anomalous['total'] = display_anomalous['total'].apply(lambda x: f"${abs(x):.2f}")
        display_anomalous['timestamp'] = display_anomalous['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(display_anomalous[[
            'transaction_id', 'timestamp', 'staff_name', 'product_name', 
            'total', 'anomaly_explanation'
        ]].rename(columns={
            'transaction_id': 'Transaction',
            'timestamp': 'Time',
            'staff_name': 'Staff',
            'product_name': 'Product',
            'total': 'Amount',
            'anomaly_explanation': 'Reason'
        }), hide_index=True, use_container_width=True)
    else:
        st.success("✅ No anomalous refunds detected!")


def show_llm_query(transactions):
    """Show natural language query interface."""
    st.header("💬 Ask Your Sales Data")
    
    st.markdown("""
    Ask questions about your sales data in natural language. The system will convert your question 
    to a data query and return the answer.
    """)
    
    # Initialize query engine
    engine = LLMQueryEngine(use_mock=True)
    
    if engine.use_mock:
        st.info("🤖 Running in **Mock Mode** (no API key required). "
               "Set OPENAI_API_KEY environment variable for LLM-powered queries.")
    else:
        st.success("🌐 Connected to OpenAI API")
    
    # Example questions
    st.subheader("💡 Example Questions")
    example_questions = [
        "What were the top 5 products by revenue?",
        "Show me sales by hour of day",
        "Which staff member had the most transactions?",
        "What was the average transaction value on weekends?",
        "How many refunds did we have?",
        "Show me sales by category",
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question, key=f"example_{i}"):
                st.session_state['selected_question'] = question
    
    # Query input
    st.subheader("🔍 Ask a Question")
    
    default_question = st.session_state.get('selected_question', '')
    user_question = st.text_input(
        "Your question:",
        value=default_question,
        placeholder="e.g., What were the top 3 products last week?"
    )
    
    if st.button("Submit Question", type="primary") or user_question:
        if user_question:
            with st.spinner("Analyzing..."):
                response = engine.query(transactions, user_question)
                
                if response['success']:
                    st.success("✅ Query executed successfully!")
                    
                    st.subheader("Generated Code")
                    st.code(response['code'], language='python')
                    
                    st.subheader("Result")
                    st.text(response['result_str'])
                    
                    # Try to visualize if result is appropriate
                    if isinstance(response['result'], dict) and len(response['result']) > 0:
                        try:
                            result_df = pd.DataFrame(list(response['result'].items()), 
                                                    columns=['Item', 'Value'])
                            if len(result_df) <= 20:  # Only visualize reasonably sized results
                                fig = px.bar(result_df, x='Item', y='Value', 
                                           title='Visualization')
                                st.plotly_chart(fig, use_container_width=True)
                        except:
                            pass
                else:
                    st.error(f"❌ Error: {response['error']}")
        else:
            st.warning("Please enter a question.")


def main():
    """Main dashboard application."""
    
    # Sidebar
    st.sidebar.title("🍁 Navigation")
    
    page = st.sidebar.radio(
        "Select View:",
        ["Overview", "Sales Forecasting", "Basket Analysis", 
         "Anomaly Detection", "Ask Your Data"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "**Maple Corner Café**\n\n"
        "AI-Powered POS Analytics Demo\n\n"
        "📌 Personal learning project\n"
        "🔒 All data is fictional\n"
        "📖 See CASE_STUDY.md for details"
    )
    
    # Load data
    with st.spinner("Loading data..."):
        transactions, inventory = load_data(num_days=90)
        forecaster, analyzer, detector = train_models(transactions)
    
    # Show selected page
    if page == "Overview":
        show_overview(transactions)
    elif page == "Sales Forecasting":
        show_forecasting(transactions, forecaster)
    elif page == "Basket Analysis":
        show_basket_analysis(transactions, analyzer)
    elif page == "Anomaly Detection":
        show_anomaly_detection(transactions, detector)
    elif page == "Ask Your Data":
        show_llm_query(transactions)


if __name__ == "__main__":
    main()
