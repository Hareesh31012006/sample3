import os
import random
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from gnews import GNews
from textblob import TextBlob
from transformers import pipeline
import torch
import torch.nn as nn
from datetime import datetime

# -----------------------------
# Set seeds for reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

st.set_page_config(page_title="Enhanced Stock Predictor", layout="wide")

# -----------------------------
# API Key
# -----------------------------
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    st.error("Alpha Vantage API key not set. Add it to Streamlit secrets or set ALPHA_VANTAGE_API_KEY env var.")
    st.stop()

# -----------------------------
# Sentiment Analysis
# -----------------------------
@st.cache_data
def get_textblob_sentiment(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.5
    tb = TextBlob(text)
    polarity = tb.sentiment.polarity
    return (polarity + 1) / 2

@st.cache_resource
def get_hf_sentiment():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

hf_sentiment_pipe = get_hf_sentiment()

def get_hf_score(text: str) -> float:
    try:
        result = hf_sentiment_pipe(text[:512])[0]  # truncate long text
        score = result['score']
        if result['label'] == 'NEGATIVE':
            score = 1 - score
        return score
    except:
        return 0.5

def combined_sentiment(text: str) -> float:
    return (get_textblob_sentiment(text) + get_hf_score(text)) / 2

# -----------------------------
# Train Model (cached per run)
# -----------------------------
@st.cache_resource
def train_model(X, y):
    model = nn.Linear(X.shape[1],1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(100):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    return model

# -----------------------------
# Stock Analysis + Backtest
# -----------------------------
def analyze_stock(ticker_symbol: str):
    try:
        # Stock Data
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        stock_data, _ = ts.get_daily(symbol=ticker_symbol, outputsize='full')
        stock_data.index = pd.to_datetime(stock_data.index)
        stock_data.index.name = "date"
        stock_df = stock_data.reset_index().sort_values("date")

        # Moving Averages
        stock_df['MA_50'] = stock_df['4. close'].rolling(50, min_periods=1).mean()
        stock_df['MA_200'] = stock_df['4. close'].rolling(200, min_periods=1).mean()

        # Daily Return
        stock_df['Daily_Return'] = stock_df['4. close'].pct_change().fillna(0)

        # -----------------------------
        # News Sentiment (cached hourly)
        # -----------------------------
        @st.cache_data(ttl=3600)  # cache for 1 hour
        def fetch_sentiment(ticker_symbol):
            gn = GNews(language='en', country='US', period='7d')
            articles = gn.get_news(f"{ticker_symbol} stock")
            news_df = pd.DataFrame(articles)
            if not news_df.empty:
                date_col = next((c for c in news_df.columns if "publish" in c.lower() or "date" in c.lower()), None)
                if date_col:
                    news_df[date_col] = pd.to_datetime(news_df[date_col], errors='coerce')
                    news_df = news_df.dropna(subset=[date_col])
                    text_col = "title" if "title" in news_df.columns else news_df.columns[0]
                    news_df['sentiment_score'] = news_df[text_col].astype(str).apply(combined_sentiment)
                    news_df['date_only'] = news_df[date_col].dt.date
                    daily_sentiment = news_df.groupby('date_only')['sentiment_score'].mean().reset_index()
                    daily_sentiment.rename(columns={'date_only':'date','sentiment_score':'average_sentiment_score'}, inplace=True)
                    daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
                else:
                    daily_sentiment = pd.DataFrame(columns=['date','average_sentiment_score'])
            else:
                daily_sentiment = pd.DataFrame(columns=['date','average_sentiment_score'])
            return daily_sentiment

        daily_sentiment = fetch_sentiment(ticker_symbol)

        # Merge Stock + Sentiment
        merged = pd.merge(stock_df, daily_sentiment, on='date', how='left')
        merged['average_sentiment_score'] = merged['average_sentiment_score'].fillna(0.5)

        # -----------------------------
        # Train PyTorch Model (cached)
        # -----------------------------
        features = ['4. close','5. volume','MA_50','MA_200','Daily_Return','average_sentiment_score']
        X = torch.tensor(merged[features].values, dtype=torch.float32)
        y = torch.tensor(merged['4. close'].values, dtype=torch.float32).view(-1,1)

        model = train_model(X, y)
        pred_last = model(X[-1:].float()).item()
        merged['Predicted_Close'] = np.nan
        merged.loc[merged.index[-1], 'Predicted_Close'] = pred_last

        # Suggestion
        last = merged.iloc[-1]
        diff_pct = (last['Predicted_Close'] - last['4. close']) / last['4. close']
        sentiment = last['average_sentiment_score']
        if diff_pct > 0.005 or sentiment > 0.55:
            suggestion = "BUY"
        elif diff_pct < -0.005 or sentiment < 0.45:
            suggestion = "SELL"
        else:
            suggestion = "HOLD"

        # Backtest
        merged['Position'] = np.where(merged['MA_50'] > merged['MA_200'],1,0)
        merged['Strategy_Return'] = merged['Position'].shift(1) * merged['Daily_Return']
        merged['Cumulative_Strategy'] = (1+merged['Strategy_Return']).cumprod()
        merged['Cumulative_Market'] = (1+merged['Daily_Return']).cumprod()

        return merged, suggestion

    except Exception as e:
        st.error(f"Error analyzing {ticker_symbol}: {e}")
        return None, "HOLD"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Enhanced Stock Market Predictor Dashboard")
st.sidebar.header("Input")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()

if st.sidebar.button("Analyze"):
    data, suggestion = analyze_stock(ticker)
    if data is None or data.empty:
        st.warning("No data returned. Check API key or ticker symbol.")
    else:
        st.subheader(f"Investment Suggestion: {suggestion}")

        # Close + Prediction
        fig, ax = plt.subplots(figsize=(12,6))
        sns.lineplot(x='date', y='4. close', data=data, ax=ax, label='Actual Close')
        last = data.iloc[-1]
        ax.scatter([last['date']],[last['Predicted_Close']], color='red', s=80, label='Predicted Next Close')
        ax.set_xlabel("Date"); ax.set_ylabel("Close Price"); ax.legend()
        st.pyplot(fig)

        # Sentiment
        st.subheader("Daily Sentiment")
        fig2, ax2 = plt.subplots(figsize=(12,4))
        sns.lineplot(x='date', y='average_sentiment_score', data=data, ax=ax2, label='Avg Sentiment')
        ax2.axhline(0.5,color='gray', linestyle='--')
        ax2.set_xlabel("Date"); ax2.set_ylabel("Sentiment (0-1)")
        ax2.legend()
        st.pyplot(fig2)

        # Volume
        st.subheader("Volume")
        fig3, ax3 = plt.subplots(figsize=(12,4))
        sns.barplot(x='date', y='5. volume', data=data, ax=ax3)
        ax3.set_xlabel("Date"); ax3.set_ylabel("Volume")
        st.pyplot(fig3)

        # Backtest
        st.subheader("Simple Backtest (MA Crossover)")
        fig4, ax4 = plt.subplots(figsize=(12,4))
        sns.lineplot(x='date', y='Cumulative_Strategy', data=data, ax=ax4, label='Strategy')
        sns.lineplot(x='date', y='Cumulative_Market', data=data, ax=ax4, label='Market')
        ax4.set_xlabel("Date"); ax4.set_ylabel("Cumulative Returns")
        ax4.legend()
        st.pyplot(fig4)

        # Table
        st.subheader("Last 5 rows")
        st.dataframe(data[['date','4. close','Predicted_Close','average_sentiment_score','5. volume']].tail())
