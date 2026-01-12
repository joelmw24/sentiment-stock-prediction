# Sentiment analysis for stock price prediction

This project was developed during a **research-oriented internship** at **BINUS University (Indonesia)**.  
It explores the relationship between **financial indicators, news sentiment and stock price movements**, through an **interactive Streamlit application**.

The application allows users to **visualize market data**, **analyze technical indicators**, **explore sentiment signals**, and **run prediction models** on selected stocks.

> ⚠️ This project is for academic and educational purposes only

---

## Project objectives

The main objectives of this project are to:

- Analyze **historical stock price behavior** using financial time series  
- Study the impact of **technical indicators** on price movements  
- Explore **news sentiment signals** and their relationship with market dynamics  
- Provide an **interactive decision-support tool** for data exploration and model interpretation  

---

## Overview

The project is implemented as a **multi-page streamlit application**.

Each page focuses on a specific analytical aspect:

### 1. Market data visualization
- Visualization of historical stock prices  
- Basic descriptive analysis of time series  
- Identification of trends and volatility patterns  

### 2. Stock price prediction
- Application of predictive models on stock prices  
- Comparison between predicted and actual values  
- Visual assessment of model performance  

### 3. Technical indicator prediction
- Computation of technical indicators (e.g. RSI, MACD, moving averages)  
- Analysis of their predictive power  
- Indicator-based forecasting approaches  

### 4. Indicator effect analysis
- Study of the influence of individual indicators on price movements  
- Visualization of relationships between indicators and returns  
- Interpretation of indicator signals  

### 5. Sentiment visualization
- Visualization of sentiment scores extracted from financial news  
- Temporal analysis of sentiment trends  
- Comparison between sentiment evolution and stock prices  

### 6. Sentiment-based prediction
- Integration of sentiment features into prediction tasks  
- Evaluation of sentiment impact on model outputs  
- Combined analysis of market data and sentiment signals  

---

## Data & methodology

- **Market data**: historical stock prices retrieved from public financial sources  
- **Technical indicators**: computed from price series using standard financial formulas  
- **Sentiment analysis**: based on textual data from financial news, processed using NLP techniques  
- **Modeling approach**:
  - Feature engineering combining price-based indicators and sentiment signals  
  - Predictive modeling with evaluation through visual comparison  

The focus of the project is placed on **interpretation and analysis**, rather than purely on predictive accuracy.

---

## Technologies used

- **Python**
- **Streamlit** (interactive web application)
- **Pandas / NumPy** (data manipulation)
- **Matplotlib** (visualization)
- **Scikit-learn / Statsmodels** (modeling and analysis)
- **NLP tools** for sentiment processing

---

## Repository structure

```
sentiment-stock-prediction/
├── Home.py                       # Main Streamlit entry point
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

---

## How to run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the Streamlit app:
```bash
streamlit run Home.py
```

---

## Author

**Joël Mwemba**  
Engineering student
