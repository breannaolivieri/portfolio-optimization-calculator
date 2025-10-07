import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime

# Portfolio stocks - you can customize these!
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
print(f"Building portfolio with: {', '.join(tickers)}")

# Download historical data (past 3 years)
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=3*365)

print("\nDownloading stock data...")
data = yf.download(tickers, start=start_date, end=end_date)['Close']

# Calculate daily returns
returns = data.pct_change().dropna()

print(f"\nAnalyzing {len(returns)} days of market data...")

# Calculate mean returns and covariance
mean_returns = returns.mean() * 252  # Annualized
cov_matrix = returns.cov() * 252  # Annualized

print("\nExpected Annual Returns:")
for ticker, ret in mean_returns.items():
    print(f"  {ticker}: {ret*100:.2f}%")

# Portfolio performance calculation
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

# Negative Sharpe ratio (we'll minimize this)
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std

# Constraints and bounds
num_assets = len(tickers)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
bounds = tuple((0, 1) for _ in range(num_assets))  # Each weight between 0 and 1
initial_guess = num_assets * [1. / num_assets]  # Equal distribution

print("\nOptimizing portfolio allocation...")

# Optimize for maximum Sharpe ratio
opt_results = minimize(neg_sharpe_ratio, initial_guess,
                      args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

# Get optimal weights
optimal_weights = opt_results['x']
opt_returns, opt_std = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
opt_sharpe = (opt_returns - 0.02) / opt_std

print("\n" + "="*50)
print("OPTIMAL PORTFOLIO ALLOCATION")
print("="*50)
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight*100:.2f}%")

print(f"\nExpected Annual Return: {opt_returns*100:.2f}%")
print(f"Expected Volatility: {opt_std*100:.2f}%")
print(f"Sharpe Ratio: {opt_sharpe:.2f}")

# Visualize the allocation
plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights * 100, color='#2E86AB', alpha=0.8)
plt.title('Optimal Portfolio Allocation', fontsize=16, fontweight='bold')
plt.ylabel('Allocation (%)', fontsize=12)
plt.xlabel('Stock', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('optimal_allocation.png', dpi=300)
print("\n✅ Allocation chart saved as 'optimal_allocation.png'!")
plt.show()