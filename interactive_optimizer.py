import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime

print("PORTFOLIO OPTIMIZATION CALCULATOR")
print("="*50)

# Get user input
print("\nEnter stock tickers separated by commas")
print("Example: AAPL,MSFT,GOOGL,AMZN")
user_input = input("Your tickers: ").upper().strip()
tickers = [t.strip() for t in user_input.split(',')]

print(f"\nAnalyzing: {', '.join(tickers)}")

# Download data
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=3*365)

print("\nDownloading stock data...")
data = yf.download(tickers, start=start_date, end=end_date)['Close']

if isinstance(data, pd.Series):
    data = data.to_frame()

# Calculate returns
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

print(f"\nAnalyzed {len(returns)} trading days")
print("\nExpected Annual Returns:")
for ticker in tickers:
    print(f"  {ticker}: {mean_returns[ticker]*100:.2f}%")

# Optimization function
def portfolio_stats(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (portfolio_return - 0.02) / portfolio_std
    return portfolio_return, portfolio_std, sharpe

def neg_sharpe(weights, mean_returns, cov_matrix):
    return -portfolio_stats(weights, mean_returns, cov_matrix)[2]

# Optimize
num_assets = len(tickers)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
initial = num_assets * [1.0 / num_assets]

print("\nOptimizing portfolio...")
result = minimize(neg_sharpe, initial, args=(mean_returns, cov_matrix),
                 method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = result['x']
opt_return, opt_std, opt_sharpe = portfolio_stats(optimal_weights, mean_returns, cov_matrix)

# Display results
print("\n" + "="*50)
print("OPTIMAL PORTFOLIO ALLOCATION")
print("="*50)
for ticker, weight in zip(tickers, optimal_weights):
    if weight > 0.01:
        print(f"{ticker}: {weight*100:.2f}%")

print("\nPortfolio Performance:")
print(f"Expected Annual Return: {opt_return*100:.2f}%")
print(f"Expected Volatility: {opt_std*100:.2f}%")
print(f"Sharpe Ratio: {opt_sharpe:.2f}")
print("="*50)

# Visualize
plt.figure(figsize=(12, 6))
plt.bar(tickers, optimal_weights * 100, color='steelblue', alpha=0.8)
plt.title('Optimal Portfolio Allocation', fontsize=16, fontweight='bold')
plt.ylabel('Allocation (%)', fontsize=12)
plt.xlabel('Stock', fontsize=12)
plt.grid(axis='y', alpha=0.3)

for i, (ticker, weight) in enumerate(zip(tickers, optimal_weights)):
    plt.text(i, weight*100 + 1, f'{weight*100:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('portfolio_allocation.png', dpi=300)
print("\nChart saved as 'portfolio_allocation.png'!")
plt.show()