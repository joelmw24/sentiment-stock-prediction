import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="📈 Sentiment-based Stock Prediction")
st.title("🔮 GRU vs LightGBM: Predicting Stock Price Change Using Sentiment")

# --- SIDEBAR PARAMETERS ---
st.sidebar.markdown("## ⚙️ Parameters")
ticker = st.sidebar.selectbox("Select a stock", ["AAPL", "MSFT", "META", "TSLA", "NVDA"])
days = st.sidebar.slider("Days to analyze", min_value=30, max_value=180, value=60)
timesteps = st.sidebar.slider("Number of past days for GRU input", min_value=3, max_value=30, value=5)

# --- GET SENTIMENT FROM GOOGLE NEWS ---
def get_google_news_sentiment(ticker="Apple", days=30):
    analyzer = SentimentIntensityAnalyzer()
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    today = datetime.datetime.today()
    data = []
    for entry in feed.entries:
        title = entry.title
        pub_date = datetime.datetime(*entry.published_parsed[:6])
        if (today - pub_date).days > days:
            continue
        sentiment = analyzer.polarity_scores(title)["compound"]
        data.append({"date": pub_date.date(), "title": title, "sentiment": sentiment})
    if not data:
        return pd.DataFrame(columns=["date", "sentiment"])
    df = pd.DataFrame(data)
    return df.groupby("date")["sentiment"].mean().reset_index()

# --- GET STOCK DATA ---
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return pd.DataFrame(columns=["date", "Close"])
    df = df[["Close"]].copy()
    df.index = df.index.date
    df.reset_index(inplace=True)
    df.columns = ["date", "Close"]
    return df

# --- MERGE DATA ---
def prepare_data(ticker, days):
    end = datetime.date.today() - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=days)
    sent_df = get_google_news_sentiment(ticker, days)
    price_df = get_stock_data(ticker, start, end)
    if price_df.empty or sent_df.empty:
        return None
    sent_df["date"] = pd.to_datetime(sent_df["date"]) - datetime.timedelta(days=1)
    sent_df["date"] = sent_df["date"].dt.date
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date
    df = pd.merge(price_df, sent_df, on="date", how="left")
    df["sentiment"] = df["sentiment"].ffill()
    df["price_change"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    return df

# --- BUILD SEQUENCES ---
def create_sequences(X_scaled, y, timesteps):
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - timesteps):
        X_seq.append(X_scaled[i:i+timesteps])
        y_seq.append(y[i+timesteps])
    return np.array(X_seq), np.array(y_seq)

# --- TRAIN MODELS ---
def train_models(df, timesteps):
    X = df[["sentiment"]].values
    y = df["price_change"].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences for GRU
    X_gru, y_gru = create_sequences(X_scaled, y, timesteps)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_gru, y_gru, test_size=0.2, shuffle=False)

    # GRU model
    model = Sequential()
    model.add(GRU(50, input_shape=(timesteps, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=30, batch_size=4, verbose=0)
    y_pred_gru = model.predict(X_test).flatten()

    # LightGBM (no sequence)
    X_gbm = X_scaled[timesteps:]
    y_gbm = y[timesteps:]
    X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(X_gbm, y_gbm, test_size=0.2, shuffle=False)
    gbm = GradientBoostingRegressor()
    gbm.fit(X_train_gbm, y_train_gbm)
    y_pred_gbm = gbm.predict(X_test_gbm)

    return y_test, y_pred_gru, y_pred_gbm

# --- MAIN ---
with st.spinner("🔎 Analyzing data and training model..."):
    df = prepare_data(ticker, days)
    if df is not None and len(df) > timesteps + 10:
        y_true, y_gru, y_gbm = train_models(df, timesteps)

        st.markdown("## 📊 Model Comparison")

        def display_metrics(y_true, y_pred, label):
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{label} MAE", f"{mean_absolute_error(y_true, y_pred):.5f}")
            col2.metric(f"{label} MSE", f"{mean_squared_error(y_true, y_pred):.5f}")
            col3.metric(f"{label} R2", f"{r2_score(y_true, y_pred):.2f}")

        display_metrics(y_true, y_gru, "GRU")
        display_metrics(y_true, y_gbm, "LightGBM")

        st.markdown("## 📈 GRU Prediction vs Actual Price Change")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_true, label="Actual")
        ax.plot(y_gru, label="GRU Prediction")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("❌ Not enough data available. Try with a different stock or longer period.")