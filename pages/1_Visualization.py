import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Selection des paramètres ---
st.set_page_config(page_title="Data Visualization", page_icon="🔍")

st.sidebar.markdown("## ⚙️ Parameters")
ticker = st.sidebar.selectbox("Choose a stock:", ["AAPL", "TSLA", "NVDA", "META", "MSFT"])
start_date = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End date", pd.to_datetime("today"))

# --- Récupération des données ---
df = yf.download(ticker, start=start_date, end=end_date)

if df.empty:
    st.warning("⚠️ No data available for the selected period.")
else:
    # Calcul du rendement journalier
    df["Daily Return"] = df["Close"].pct_change()

    # --- Aperçu des données ---
    st.markdown("## 📄 Recent stock data")
    st.dataframe(df[["Open", "High", "Low", "Close"]].tail(10))

    # --- Courbe des prix de cloture ---
    st.markdown("## 📈 Closing price ")
    st.line_chart(df["Close"])

    # --- Statistiques ---
    st.markdown("## 📊 Key financial statistics")
    st.info("""
                This table shows basic statistics that help describe how the stock has behaved over the selected period:
                - **Mean / Median close price**: The average or middle closing price, gives you an idea of the general price level
                - **Volatility (standard deviation)**: measures how much the price goes up and down. A higher value means the stock is more "unstable" or risky
                - **Min / Max close price**: the lowest and highest prices reached in that period
                - **Average daily Return**: on average, how much the price changed (up or down) each day, in percentage
                - **Return volatility**: how much daily returns fluctuate. This also helps understand risk

                These values are useful to compare companies or to understand how stable or active a stock is.
                """)
    stats = {
        "Mean close price ($)": df["Close"].mean(),
        "Median close price ($)": df["Close"].median(),
        "Price volatility": df["Close"].std(),
        "Min close price ($)": df["Close"].min(),
        "Max close price ($)": df["Close"].max(),
        "Average daily return (%)": df["Daily Return"].mean() * 100,
        "Return volatility (%)": df["Daily Return"].std() * 100
    }

    stats_df = pd.DataFrame({
        "Metric": list(stats.keys()),
        "Value": [f"{float(v):.3f}" for v in stats.values()]
    })
    st.table(stats_df)

    # --- Décomposition saisonnière ---
    st.markdown("## 🧩 Seasonal decomposition (additive)")
    st.info("""
This graph breaks down the stock's price into 3 parts:

- **Trend**: the long-term direction of the stock. Is it going up, down, or sideways?
- **Seasonality**: repeating patterns, like small cycles that happen every few months.
- **Residuals**: random movements that don’t follow a pattern.

Why is this useful?
                
Because financial time series (like stock prices) often contain **hidden structures**. By separating the signal (trend + seasonality) from the noise (residual), we can better understand and even forecast price behavior.
""")
    if len(df) >= 252:
        result = seasonal_decompose(df["Close"], model='additive', period=252)
        fig2, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(result.observed); axs[0].set_title("Observed")
        axs[1].plot(result.trend); axs[1].set_title("Trend")
        axs[2].plot(result.seasonal); axs[2].set_title("Seasonal")
        axs[3].plot(result.resid); axs[3].set_title("Residual")
        for ax in axs: ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.warning("Need at least 252 data points for decomposition.")