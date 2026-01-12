# Sentiment & Indicators – Stock Prediction (Streamlit App)

Streamlit web app built during an internship/research project to explore how **news sentiment** and **technical indicators**
relate to **short-term stock price movements** (AAPL, TSLA, NVDA, META, MSFT).  
The app includes data visualization, sentiment analysis, indicator analysis, and model-based prediction pages.

## Features
- 📈 Market data download via `yfinance`
- 📰 News sentiment scoring (VADER) + correlation visualization
- 📊 Technical indicator analysis and impact exploration
- 🔮 Model comparison for prediction (e.g., GRU vs LightGBM) on selected pages
- 🖥️ Streamlit multi-page interface

## Project structure
```
.
├── Home.py
├── pages/
│   ├── 1_Visualization.py
│   ├── 2_StockPrediction.py
│   ├── 3_IndicatorPrediction.py
│   ├── 4_IndicatorEffect.py
│   ├── 5_SentimentVisualization.py
│   └── 6_SentimentPrediction.py
├── requirements.txt
└── .gitignore
```

## Quickstart

### 1) Create a virtual environment and install dependencies
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the app
```bash
streamlit run Home.py
```

## Notes
- If some pages require external API keys / specific data sources, keep them in a local `.env` file (not committed).
- If `prophet` installation fails on your OS, install it separately following the official instructions for your environment.

## Author
Joël Mwemba
