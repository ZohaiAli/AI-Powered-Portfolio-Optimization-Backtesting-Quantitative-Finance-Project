
# 🧠 AI-Powered Portfolio Optimization & Backtesting  
### Quantitative Finance Project using Machine Learning and Reinforcement Learning

---

## 📘 Overview  

This project implements an **AI-driven portfolio optimization system** that combines **Modern Portfolio Theory (MPT)** with **Machine Learning** and **Reinforcement Learning (RL)** concepts to build, optimize, and backtest intelligent investment portfolios.

It uses real financial market data (fetched automatically via **Yahoo Finance API**) to:
- Optimize asset allocation
- Maximize risk-adjusted returns (Sharpe Ratio)
- Evaluate portfolio performance vs benchmarks (e.g., S&P 500)
- Prepare for AI-based dynamic rebalancing

---

## ⚙️ Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.11+ |
| **Data** | `yfinance`, `pandas`, `numpy` |
| **Optimization** | `PyPortfolioOpt`, `cvxpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **AI (optional)** | `TensorFlow` / `PyTorch` (for reinforcement learning) |

---

## 🧩 Project Structure  

```

ai_portfolio_optimizer/
│
├── data_fetch.py          # Downloads historical stock price data
├── portfolio_opt.py       # Performs Markowitz portfolio optimization
├── backtest.py            # Evaluates portfolio performance
├── prices.csv             # Saved historical price data
├── optimal_weights.csv    # Saved optimal portfolio weights
├── .gitignore             # Ignore unnecessary files
└── README.md              # Project documentation

````

---

## 🚀 How It Works  

### 1️⃣ Fetch Historical Data  

We use the **Yahoo Finance API** via the `yfinance` library to pull adjusted closing prices for selected tickers over the last 5 years.  

```python
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
    print(prices.tail())
    prices.to_csv("prices.csv")
    print("\n✅ Data saved to prices.csv")
````

---

### 2️⃣ Portfolio Optimization (Markowitz MPT)

The **Efficient Frontier** optimization is performed using `PyPortfolioOpt`.
This step finds the optimal weights for each asset to **maximize the Sharpe ratio** given historical returns and covariance.

```python
# portfolio_opt.py
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Load price data
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
print("✅ Data loaded successfully.\n")

# Calculate expected returns and sample covariance
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

# Optimize for maximum Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("🔹 Optimal Weights:")
for asset, weight in cleaned_weights.items():
    print(f"{asset}: {weight:.2%}")

# Portfolio performance
performance = ef.portfolio_performance(verbose=True)

# Save results
pd.Series(cleaned_weights).to_csv("optimal_weights.csv")
print("\n✅ Optimal weights saved to optimal_weights.csv")
```

**Example Output:**

```
🔹 Optimal Weights:
AAPL: 20.12%
AMZN: 14.45%
GOOGL: 19.88%
MSFT: 25.77%
SPY: 9.30%
TSLA: 10.48%

Expected annual return: 18.2%
Annual volatility: 13.4%
Sharpe Ratio: 1.36
✅ Optimal weights saved to optimal_weights.csv
```

---

### 3️⃣ Backtesting Portfolio Performance

A backtesting script evaluates performance metrics like:

* CAGR (Compound Annual Growth Rate)
* Sharpe Ratio
* Maximum Drawdown

```python
# backtest.py
import pandas as pd
import numpy as np

values = [100, 110, 115, 120, 130, 125]  # Example portfolio values
returns = pd.Series(values).pct_change().dropna()

cagr = (values[-1] / values[0]) ** (252 / len(values)) - 1
sharpe = returns.mean() / returns.std() * np.sqrt(252)
max_dd = (pd.Series(values) / pd.Series(values).cummax() - 1).min()

print(f"CAGR: {cagr:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")
```

---

## 🧠 Future Work: AI Integration

Once the baseline optimization works, the next stage introduces **AI**:

* Build a **Reinforcement Learning agent (RL)** that observes market states and portfolio performance.
* Reward function = Sharpe Ratio improvement or profit growth.
* The RL agent dynamically adjusts portfolio weights instead of static optimization.
* Optional integration of **sentiment data** or **price momentum signals**.

---

## 📈 Performance Metrics Explained

| Metric           | Meaning                                            |
| ---------------- | -------------------------------------------------- |
| **CAGR**         | Compound Annual Growth Rate (true growth per year) |
| **Sharpe Ratio** | Risk-adjusted return measure                       |
| **Max Drawdown** | Maximum observed loss from a peak                  |
| **Volatility**   | Standard deviation of returns (risk)               |

---

## 🧾 Example Workflow

```bash
# Step 1: Fetch data
python data_fetch.py

# Step 2: Optimize portfolio
python portfolio_opt.py

# Step 3: Backtest performance
python backtest.py
```

---

## 📚 References

* [PyPortfolioOpt Documentation](https://pyportfolioopt.readthedocs.io/en/latest/)
* [Modern Portfolio Theory (Markowitz, 1952)](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
* [Yahoo Finance API](https://pypi.org/project/yfinance/)
* [Reinforcement Learning in Finance (OpenAI Gym / FinRL)](https://github.com/AI4Finance-Foundation/FinRL)

---

## 🛡️ .gitignore

Example `.gitignore` for this project:

```gitignore
# Python cache
__pycache__/
*.py[cod]

# Virtual environments
venv/
env/
.venv/

# Data & model files
*.csv
*.xlsx
*.h5
*.pkl
*.pth
*.pt

# Logs and temporary files
*.log
*.tmp
.cache/

# IDE and system files
.vscode/
.idea/
.DS_Store
Thumbs.db
```

---

## 🧠 Keywords

`AI Finance` · `Quantitative Finance` · `Portfolio Optimization` · `Reinforcement Learning` · `Machine Learning` · `Python` · `Backtesting` · `Algorithmic Trading`

---

## 👤 Author

**Zohai [Syed Zohaib Ali]**
📧 zohaibmansoor.ali@gmail.com
💻 Built with Python and curiosity for financial intelligence.


