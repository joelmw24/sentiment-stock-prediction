import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="📈 Sentiment-based Stock Prediction")
st.title("🔮 GRU vs LightGBM: Predicting Stock Price Change Using Sentiment & Technical Indicators")

# Sidebar
st.sidebar.markdown("⚙️ Parameters")
ticker = st.sidebar.selectbox("Select a stock", ["AAPL", "MSFT", "META", "TSLA", "NVDA"])
days = st.sidebar.slider("Days to analyze", min_value=30, max_value=180, value=60)

# --- Sentiment ---
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

# --- Stock data ---
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return pd.DataFrame(columns=["date", "Close"])
    df = df[["Close", "Volume"]].copy()
    df.index = df.index.date
    df.reset_index(inplace=True)
    df.columns = ["date", "Close", "Volume"]
    return df

# --- Merge + Indicators ---
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

    # --- RSI ---
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- Volatility ---
    df["volatility"] = df["Close"].rolling(window=14).std()

    # --- MACD ---
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    # --- Drop NA ---
    df.dropna(inplace=True)
    return df

# --- Modeling ---
def train_models(df):
    X = df[["sentiment", "RSI", "volatility", "MACD"]].values
    y = df["price_change"].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_gru = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    X_train, X_test, y_train, y_test = train_test_split(X_gru, y, test_size=0.2, shuffle=False)

    # GRU Model
    model = Sequential()
    model.add(GRU(50, input_shape=(1, X_scaled.shape[1])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=30, batch_size=4, verbose=0)
    y_pred_gru = model.predict(X_test).flatten()

    # LightGBM
    X_gbm = X_scaled
    X_train_gbm, X_test_gbm, y_train_gbm, y_test_gbm = train_test_split(X_gbm, y, test_size=0.2, shuffle=False)
    gbm = GradientBoostingRegressor()
    gbm.fit(X_train_gbm, y_train_gbm)
    y_pred_gbm = gbm.predict(X_test_gbm)

    return y_test, y_pred_gru, y_pred_gbm

# --- Run ---
with st.spinner("🔎 Analyzing data and training model..."):
    df = prepare_data(ticker, days)
    if df is not None:
        y_true, y_gru, y_gbm = train_models(df)

        st.markdown("📊 Model Comparison")

        def display_metrics(y_true, y_pred, label):
            col1, col2, col3 = st.columns(3)
            col1.metric(f"{label} MAE", f"{mean_absolute_error(y_true, y_pred):.5f}")
            col2.metric(f"{label} MSE", f"{mean_squared_error(y_true, y_pred):.5f}")
            col3.metric(f"{label} R2", f"{r2_score(y_true, y_pred):.2f}")

        display_metrics(y_true, y_gru, "GRU")
        display_metrics(y_true, y_gbm, "LightGBM")

        st.markdown("📈 GRU Prediction vs Actual Price Change")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_true, label="Actual")
        ax.plot(y_gru, label="GRU Prediction")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("❌ Not enough data available. Try with a different stock or longer period.")