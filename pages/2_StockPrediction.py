import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.optimizers import Adam

from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lightgbm import LGBMRegressor

# --- CONFIG ---
st.set_page_config(page_title="Stock prediction", page_icon="🔮")
st.title("🔮 Stock price prediction ")
st.info("""
We’ll use and compare **four different models** to forecast stock prices for the next 30 days.  
Each model takes a different approach to prediction:

- **GRU (Gated Recurrent Unit):** a deep learning model that learns from sequences of past stock prices to forecast future values
- **Prophet:** a forecasting tool developed by Facebook, designed to automatically detect trends, seasonality, and holidays in time series data
- **SARIMA (Seasonal ARIMA):** a classical statistical model that uses patterns in past values and seasonality to make forecasts
- **LightGBM:** a machine learning model based on gradient-boosted decision trees, using past data (lag features) to predict future prices
            
---       

To assess how well each model predicts stock prices, we use three common performance metrics.

#### **MAE (Mean Absolute Error)**

- **What it is:** the average difference between the predicted prices and the actual prices
- **Why it matters:** lower MAE means the predictions are closer to reality

📌 Example: If MAE is 2.5, the model is typically off by about $2.50.

#### **RMSE (Root Mean Squared Error)**

- **What it is:** like MAE, but it penalizes larger mistakes more heavily
- **Why it matters:** gives a better sense of how big the bigger errors are

📌 Example: an RMSE of 3.2 means the typical error is around $3.20, with more weight on larger mistakes.


#### **R² Score**

- **What it is:** a number between 0 and 1 that shows how well the model explains the real stock price movements.
- **Why it matters:**
  - **1.0 =** perfect predictions
  - **0.0 =** no better than guessing the average
  - **Less than 0 =** worse than guessing

📌 Example: an R² of 0.85 means 85% of the variation in stock prices is explained by the model.

            
""")

# --- INPUTS ---
st.sidebar.markdown("## ⚙️ Parameters")
ticker = st.sidebar.selectbox("Choose a stock:", ["AAPL", "TSLA", "NVDA", "META", "MSFT"])
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))
seq_len = st.sidebar.slider("How many past days to use (for GRU & LightGBM)?",  min_value=5, max_value=60, value=20)

if st.button("📥 Run all predictions"):
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        st.warning("No data available.")
        st.stop()

    df = df[["Close"]].dropna()
    df["Date"] = df.index
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]])

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(X), np.array(y)

    def run_gru_model():
        X, y = create_sequences(scaled, seq_len)
        X_train, y_train = X[:-30], y[:-30]
        X_test, y_test = X[-30:], y[-30:]
        X_train = X_train.reshape((-1, seq_len, 1))
        X_test = X_test.reshape((-1, seq_len, 1))
        model = Sequential([
            Input(shape=(seq_len, 1)),
            GRU(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.0005), loss='mse')
        model.fit(X_train, y_train, epochs=100, verbose=0)
        y_pred = model.predict(X_test)
        return scaler.inverse_transform(y_pred), scaler.inverse_transform(y_test.reshape(-1, 1))

    def run_prophet_model():
        # Utiliser l'index comme colonne de dates
        df_prophet = df[["Close"]].copy()
        df_prophet["ds"] = df_prophet.index  # Prophet attend une colonne "ds"
        df_prophet["y"] = df_prophet["Close"]

        df_prophet = df_prophet[["ds", "y"]].dropna()

        # Sécurité : forcer les bons types
        df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
        df_prophet = df_prophet.dropna()

        model = Prophet()
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        y_pred = forecast.tail(30)["yhat"].values
        y_true = df["Close"].values[-30:]

        return y_pred.reshape(-1, 1), y_true.reshape(-1, 1)

    def run_sarima_model():
        model = SARIMAX(df["Close"], order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(30)
        y_true = df["Close"].values[-30:]
        return forecast.values.reshape(-1, 1), y_true.reshape(-1, 1)

    def run_lgbm_model():
        X, y = create_sequences(scaled, seq_len)
        X = X.reshape((X.shape[0], X.shape[1]))
        X_train, y_train = X[:-30], y[:-30]
        X_test, y_test = X[-30:], y[-30:]
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return scaler.inverse_transform(y_pred.reshape(-1, 1)), scaler.inverse_transform(y_test.reshape(-1, 1))

    # --- RUN MODELS ---
    models = {
        "GRU": run_gru_model,
        "Prophet": run_prophet_model,
        "SARIMA": run_sarima_model,
        "LightGBM": run_lgbm_model
    }

    for name, func in models.items():
        st.subheader(f"📊 {name}")
        with st.spinner(f"🔎 Analyzing data and training {name} model..."):
            try:
                y_pred, y_true = func()
                fig, ax = plt.subplots()
                ax.plot(y_true, label="Actual", linewidth=2)
                ax.plot(y_pred, label="Predicted", linestyle="--")
                ax.set_title(f"{ticker} - {name} Prediction")
                ax.legend()
                st.pyplot(fig)

                # Metrics
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)

                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")
                st.write(f"**R² Score:** {r2:.3f}")
            except Exception as e:
                st.error(f"{name} failed: {e}")