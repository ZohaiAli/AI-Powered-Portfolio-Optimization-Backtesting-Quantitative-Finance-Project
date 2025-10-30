# data_fetch.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(tickers, years=5):
    end = datetime.today()
    start = end - timedelta(days=365*years)
    data = yf.download(tickers, start=start, end=end)['Close']

    return data

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
    prices = fetch_data(tickers)
    print(prices.tail())  # 👈 This line actually shows you data
    prices.to_csv("prices.csv")
    print("\n✅ Data saved to prices.csv")
