# data_fetch.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(tickers, years=5):
    """
    Returns adjusted close prices DataFrame for given tickers over the last `years` years.
    """
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=365*years)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.ffill().dropna(how='all')
    return data

if __name__ == "__main__":
    tickers = ["AAPL","MSFT","GOOGL","AMZN","TSLA","SPY"]
    prices = fetch_data(tickers)
    print(prices.tail())
