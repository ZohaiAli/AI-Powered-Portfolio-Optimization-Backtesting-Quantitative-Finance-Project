# train_agent.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import FlattenObservation
from data_fetch import fetch_data
from ai_portfolio_optimizer.portfolio_opt import PortfolioEnv
import numpy as np

def make_env(prices, window=30, tc=0.001):
    def _init():
        env = PortfolioEnv(prices, window_size=window, transaction_cost=tc)
        # Flatten observation so SB3 can handle it easily (obs becomes 1D)
        env = FlattenObservation(env)
        return env
    return _init

if __name__ == "__main__":
    # choose tickers - include SPY if you want benchmark separately
    tickers = ["AAPL","MSFT","GOOGL","AMZN","TSLA","JNJ","XOM","NVDA"]
    prices = fetch_data(tickers, years=5)

    # split train/test (we'll train on first 80%)
    split = int(len(prices) * 0.8)
    train_prices = prices.iloc[:split]

    env = DummyVecEnv([make_env(train_prices, window=30, tc=0.001)])
    model = PPO("MlpPolicy", env, verbose=1, batch_size=64, n_steps=2048)

    # quick test: use small timesteps (20000) while debugging; increase to 200k+ for real training
    TIMESTEPS = int(os.environ.get("RL_TIMESTEPS", 20000))
    model.learn(total_timesteps=TIMESTEPS)

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_portfolio")
    print("Model saved to models/ppo_portfolio.zip")
