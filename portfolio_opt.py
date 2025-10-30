# portfolio_opt.py
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Step 1: Load price data
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
print("✅ Data loaded successfully.\n")

# Step 2: Compute returns and covariance
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

# Step 3: Optimize portfolio for max Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("🔹 Optimal Weights:")
for asset, weight in cleaned_weights.items():
    print(f"{asset}: {weight:.2%}")

# Step 4: Performance
performance = ef.portfolio_performance(verbose=True)

# Step 5: Save results
pd.Series(cleaned_weights).to_csv("optimal_weights.csv")
print("\n✅ Optimal weights saved to optimal_weights.csv")
