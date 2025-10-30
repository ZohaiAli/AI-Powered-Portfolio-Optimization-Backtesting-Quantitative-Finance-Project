# backtest.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from data_fetch import fetch_data
from ai_portfolio_optimizer.portfolio_opt import PortfolioEnv
from gymnasium.wrappers import FlattenObservation

def run_rl_backtest(model_path, test_prices, window=30):
    env = PortfolioEnv(test_prices, window_size=window, transaction_cost=0.001)
    obs, _ = env.reset()

    model = PPO.load(model_path)
    values = [env.portfolio_value]
    weights_history = []

    done = False
    while not done:
        # SB3 models expect flattened obs; policy will accept the shape returned by FlattenObservation
        # However, we didn't wrap env in FlattenObservation here, so model.predict expects flattened array.
        # Flatten obs to 1D
        flattened = obs.flatten()[None, :]
        action, _ = model.predict(flattened, deterministic=True)
        # action may be batch-shaped -> take first
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action[0]
        obs, reward, done, _, info = env.step(action)
        values.append(info["portfolio_value"])
        weights_history.append(info.get("weights"))

    # create pandas Series of portfolio values indexed by test_prices dates (aligned)
    # The env used reset() without preserving original index; reconstruct index:
    # Start index is test_prices.index[window], and we have len(values)-1 steps
    start_idx = test_prices.index[window]
    idx = test_prices.index[window: window + len(values)]
    port_series = pd.Series(values, index=idx)
    weights_df = pd.DataFrame(weights_history, index=idx, columns=test_prices.columns)
    return port_series, weights_df

def compute_metrics(series):
    # series: cumulative portfolio value (starting at 1)
    returns = series.pct_change().dropna()
    # CAGR
    total_periods = len(series)
    cagr = (series.iloc[-1]) ** (252 / total_periods) - 1
    # Sharpe
    if returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    # Max Drawdown
    roll_max = series.cummax()
    max_dd = ((series / roll_max) - 1).min()
    return cagr, sharpe, max_dd, returns

if __name__ == "__main__":
    # tickers: keep same as training (do not include benchmark here); SPY as separate benchmark if desired
    tickers = ["AAPL","MSFT","GOOGL","AMZN","TSLA","JNJ","XOM","NVDA","SPY"]
    prices = fetch_data(tickers, years=5)
    prices = prices.dropna(axis=1, how='all')

    # split
    split = int(len(prices) * 0.8)
    train = prices.iloc[:split]
    test = prices.iloc[split:]

    # separate asset universe and benchmark
    if "SPY" in prices.columns:
        benchmark = prices["SPY"].iloc[split:]
    else:
        benchmark = None
    asset_prices = prices.drop(columns=["SPY"]) if "SPY" in prices.columns else prices

    # Check model
    model_path = "models/ppo_portfolio.zip"
    if not os.path.exists(model_path):
        print("Model not found. Train model first with: python train_agent.py")
        exit(1)

    # Run backtest on test dataset (asset_prices)
    port_series, weights_df = run_rl_backtest(model_path, asset_prices, window=30)

    # Compute metrics (this is your snippet)
    cagr, sharpe, max_dd, returns = compute_metrics(port_series)

    print(f"CAGR: {cagr:.2%}")
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")

    # plot portfolio vs benchmark
    plt.figure(figsize=(10,6))
    plt.plot(port_series / port_series.iloc[0], label="RL Portfolio (normalized)")
    if benchmark is not None:
        benchmark_cum = (1 + benchmark.pct_change().loc[port_series.index].fillna(0)).cumprod()
        plt.plot(benchmark_cum / benchmark_cum.iloc[0], label="SPY (normalized)")
    plt.legend()
    plt.title("Backtest: RL Portfolio vs Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.show()

    # save weights history snapshot
    weights_df.to_csv("weights_history.csv")
    print("Weights history saved to weights_history.csv")
