import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns


# Config page
st.set_page_config(page_title="Sentiment vs stock price", page_icon="📰")
st.title("📰 Sentiment analysis & stock price correlation")

# Sidebar parameters
st.sidebar.markdown("## ⚙️ Parameters")
ticker = st.sidebar.selectbox("Choose a company:", ["AAPL", "MSFT", "META", "TSLA", "NVDA"])
days = st.sidebar.slider("Days to analyze", min_value=7, max_value=60, value=14)

# --- Sentiment from Google News ---
def get_google_news_sentiment(ticker="Apple", days=14):
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

# --- Stock price data ---
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return pd.DataFrame(columns=["date", "Close"])
    df = df[["Close"]].copy()
    df.index = df.index.date
    df.reset_index(inplace=True)
    df.columns = ["date", "Close"]
    return df

# --- Merge and compute correlation ---
def merge_sentiment_price(ticker, days):
    end = datetime.date.today() - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=days)

    sent_df = get_google_news_sentiment(ticker, days)
    price_df = get_stock_data(ticker, start, end)

    if price_df.empty or sent_df.empty:
        return None, None, None

    sent_df["date"] = pd.to_datetime(sent_df["date"]) - datetime.timedelta(days=1)
    sent_df["date"] = sent_df["date"].dt.date
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.date

    merged = pd.merge(price_df, sent_df, on="date", how="left")
    merged["sentiment"] = merged["sentiment"].ffill()
    merged["price_change"] = merged["Close"].pct_change()

    correlation = merged[["sentiment", "price_change"]].corr()
    return merged, sent_df, correlation

# --- Main execution ---
with st.spinner("🔎 Analyzing data..."):
    merged_df, sentiment_df, corr = merge_sentiment_price(ticker, days)

# --- Display ---
if merged_df is None:
    st.error("❌ Not enough data for sentiment or price.")
else:
    st.subheader("🧠 Daily sentiment score")
    st.info("""
    This chart shows the **average sentiment score per day** based on recent news headlines about the company.

    - A **positive score** (above 0) means the headlines were generally optimistic.
    - A **negative score** (below 0) means the tone was more negative.
    - A score near **0** means neutral or mixed sentiment.

    We use the **VADER sentiment analyzer**, which is great for short text like headlines.
    """)
    st.line_chart(sentiment_df.set_index("date")["sentiment"])

    st.subheader("📈 Closing price")
    st.info("""
This line chart shows the **daily closing price** of the stock over the selected period.
You can compare this trend with sentiment scores to see if there's a visible pattern.
""")
    st.line_chart(merged_df.set_index("date")["Close"])

    st.subheader("🔁 Price change vs. sentiment")
    st.info("""
Here we compare **two lines**:
- Sentiment score (from news)
- Daily price change (%), showing how much the stock moved compared to the previous day

""")
    st.line_chart(merged_df.set_index("date")[["price_change", "sentiment"]])

    st.subheader("📊 Correlation between sentiment & price change")
    st.info("""
This table shows the **correlation coefficient** between:
- News sentiment, and
- Daily stock price change

A value near:
- **+1** means strong positive correlation (sentiment up → price up)
- **-1** means strong negative correlation (sentiment up → price down)
- **0** means no clear relationship
""")
    st.dataframe(corr)
    # Plot heatmap
    st.markdown("### 🔥 Correlation heatmap")
    st.info("""
    This heatmap visually represents the strength of the correlation:

    - **Darker colors** = stronger relationship  
    - The closer the value is to **1 or -1**, the stronger the link
    """)

    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    #st.subheader("📄 Last 10 days: merged data")
    #st.dataframe(merged_df.tail(10))