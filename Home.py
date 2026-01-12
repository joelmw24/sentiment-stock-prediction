import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Welcome",page_icon="📈")
st.title("Stock analysis web app")

# --- INTRODUCTION ---
st.write("""
This project was created to explore how **financial indicators** (like moving averages, volatility, etc.) can help **understand and anticipate short-term stock price movements**.

### 🎯 Goal of the project
The main objective is to:
- Use **real stock market data** from top global companies
- Analyze this data using **key technical indicators**
- Build and test a **machine and deep learning model** that tries to predict whether a stock’s price will go up or down

This app allows users to **interact with the data**:  
You can visualize price trends, explore indicators, and see how the prediction logic works without needing to be an expert in finance or data science.

---

### 🔍 What you can do 

- Visualize stock price history for selected companies  
- See financial indicators like:
    - `Moving averages (short and long term)`
    - `Rolling volatility`
    - `Daily returns and other signals`  
- Understand how these indicators relate to price changes  
- (Optional) Test a prediction model trained on this data

The analysis is based entirely on **quantitative financial data**, not news or social media sentiment.

---

### 🛠️ How it works 

- Historical stock data is collected (e.g., from Yahoo Finance)
- Indicators like rolling mean, standard deviation, RSI, etc. are calculated
- The data is visualized with simple, clean charts
- (Optional) A machine learning model uses this data to make price movement predictions

---

📌 Select on the side, on the visualization menu, to start exploring the data 📌 
""")


