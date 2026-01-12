import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# --- CONFIG
st.set_page_config(page_title="Indicator effect", page_icon="📋")
st.title("📋 Impact of macroeconomic indicators")

st.info("""
How certain **macroeconomic indicators** evolve when the **stock price increases or decreases**.
We calculate the **average variation** of each indicator during rising and falling stock weeks.

---

### 📘 Explanation of indicators:
- **VIX (Volatility index):** measures market volatility expectations. A rising VIX typically signals fear or uncertainty
- **SP500 (S&P 500 index):** benchmark for US stock market performance. It reflects overall market sentiment
- **BTC (bitcoin):** considered a speculative asset and alternative investment, often linked with investor risk appetite
- **GOLD:** a traditional safe-haven asset. Investors often turn to gold during market downturns or instability

These indicators are compared weekly to observe their typical behavior when stock prices are either increasing or decreasing.
""")

# --- PARAMETERS
tickers = ["AAPL", "TSLA", "NVDA", "META", "MSFT"]
macro_tickers = {
    "VIX": "^VIX",
    "SP500": "^GSPC",
    "BTC": "BTC-USD",
    "GOLD": "GC=F"
}

@st.cache_data
def load_data():
    data = []

    for ticker in tickers:
        df = yf.download(ticker, start="2018-01-01", interval="1wk")[["Close"]]
        df = df.rename(columns={"Close": "Stock"})

        for name, symbol in macro_tickers.items():
            macro = yf.download(symbol, start="2018-01-01", interval="1wk")[["Close"]]
            macro = macro.rename(columns={"Close": name})
            df = df.join(macro, how="inner")

        df["price_change"] = df["Stock"].pct_change()
        for col in macro_tickers.keys():
            df[f"{col}_pct"] = df[col].pct_change()

        df["ticker"] = ticker
        df.dropna(inplace=True)
        data.append(df)

    return pd.concat(data)

df_all = load_data()

# --- GROUPS
rising = df_all[df_all["price_change"] > 0]
falling = df_all[df_all["price_change"] < 0]

pct_cols = [f"{k}_pct" for k in macro_tickers.keys()]
avg_up = rising[pct_cols].mean()
avg_down = falling[pct_cols].mean()

comparison_df = pd.DataFrame({
    "When price rises": avg_up.values,
    "When price falls": avg_down.values
}, index=list(macro_tickers.keys()))

# --- DISPLAY
st.subheader("📊 Average indicator variations (%)")
st.dataframe(comparison_df.style.format("{:.2%}"))

# --- GRAPH
st.markdown("### 📈 Comparative chart")
fig, ax = plt.subplots()
comparison_df.plot(kind="bar", ax=ax, rot=0)
plt.axhline(0, color="gray", linestyle="--")
plt.ylabel("Average variation")
plt.title("Average behavior of macroeconomic indicators")
plt.tight_layout()
st.pyplot(fig)