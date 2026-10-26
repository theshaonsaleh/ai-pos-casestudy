# 🍁 AI-Powered POS Analytics: A Case Study

**Personal learning project, non-commercial, not affiliated with any business. All data is fictional.**

This project demonstrates how artificial intelligence and machine learning can enhance a point-of-sale (POS) system for small retail businesses. Using a fictional café called "Maple Corner Café" as an example, we explore practical AI applications including sales forecasting, market basket analysis, refund anomaly detection, and natural language queries—all using completely synthetic data. All pricing in GBP (£).

## 🎯 What This Project Demonstrates

- **Sales Forecasting**: Predict hourly and daily sales to optimize staffing and inventory
- **Market Basket Analysis**: Discover product combinations for upselling and bundling
- **Anomaly Detection**: Flag unusual refund patterns that might indicate fraud or issues
- **Natural Language Q&A**: Ask questions about sales data in plain English
- **Data Generation**: Create realistic synthetic POS transaction data

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-pos-casestudy.git
cd ai-pos-casestudy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
# Run the command-line demo
python demo.py
```

This will:
- Generate 90 days of synthetic transaction data
- Train all ML models
- Display forecasts, basket analysis, anomaly detection results
- Save visualizations as PNG files

### Run the Interactive Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard.py
```

Then open your browser to `http://localhost:8501` to explore the interactive dashboard.

## 📊 Features

### 1. Sales Forecasting
- Hourly and daily sales predictions
- Peak hour identification
- Day-of-week patterns
- Performance metrics (MAE, RMSE, MAPE)

### 2. Market Basket Analysis
- Association rule mining (Apriori algorithm)
- Frequent product pair discovery
- Upsell recommendations based on current basket
- Product affinity analysis

### 3. Anomaly Detection
- Isolation Forest for refund anomaly detection
- Staff-level refund analysis
- Explanation of detected anomalies
- Statistical refund metrics

### 4. Natural Language Queries
- Convert questions to pandas/SQL code
- OpenAI API integration (optional)
- Offline mock mode for demo without API key
- Example queries included

## 📁 Project Structure

```
ai-pos-casestudy/
├── CASE_STUDY.md           # Detailed case study documentation
├── README.md               # This file
├── LICENSE                 # MIT License
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore rules
│
├── demo.py                # Command-line demo script
├── dashboard.py           # Streamlit interactive dashboard
│
├── src/
│   ├── data/
│   │   └── generator.py   # Synthetic data generator
│   └── models/
│       ├── forecaster.py  # Sales forecasting model
│       ├── basket_analyzer.py  # Market basket analysis
│       ├── anomaly_detector.py # Refund anomaly detection
│       └── llm_query.py   # Natural language query engine
│
└── tests/
    ├── test_data_generator.py
    └── test_models.py
```

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📖 Documentation

See [CASE_STUDY.md](CASE_STUDY.md) for:
- Detailed problem statement and use cases
- Technical approach and architecture
- Implementation details
- Results and evaluation
- Ethics, risks, and limitations
- What I learned
- Future enhancements

## 🔒 Ethics & Privacy

This is a **learning project only**:
- All data is **completely fictional** and generated synthetically
- No real businesses, customers, employees, or transactions
- No real personal data or payment information
- No commercial use or affiliation with any company

**Production considerations** covered in the case study:
- Data privacy and UK GDPR/ICO compliance
- Bias detection and fairness
- Human oversight requirements
- Model monitoring and maintenance

## 🛠️ Technologies Used

- **Python 3.9+**: Core language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning (Isolation Forest)
- **MLxtend**: Market basket analysis (Apriori)
- **Statsmodels**: Time series analysis
- **Matplotlib/Seaborn/Plotly**: Visualization
- **Streamlit**: Interactive dashboard
- **OpenAI API**: Natural language queries (optional)

## 💡 Example Questions (Natural Language Q&A)

- "What were the top 5 products by revenue?"
- "Show me sales by hour of day"
- "Which staff member had the most transactions?"
- "What was the average transaction value on weekends vs weekdays?"
- "How many refunds did we have?"
- "Show me sales by category"

## 🎓 Learning Outcomes

This project demonstrates:
- Synthetic data generation with realistic patterns
- Time series forecasting techniques
- Association rule mining for retail
- Unsupervised anomaly detection
- LLM integration for natural language interfaces
- Building interactive data dashboards
- Writing technical case studies
- Ethical AI considerations

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This is a personal learning project created to explore AI applications in retail operations. It is not affiliated with any company, commercial product, or real business.

**Fictional Business**: Maple Corner Café is entirely fictional and was created for this demonstration.

## 📬 Questions?

This is a portfolio/learning project. For questions about the implementation, see the detailed case study or explore the well-commented source code.

---

**Remember**: This is a learning project with fictional data. Real POS systems require proper security, compliance, and operational considerations beyond this demo scope.
