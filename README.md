# Portfolio Optimization Calculator

A Python-based portfolio optimization tool that helps investors build optimal investment portfolios using Modern Portfolio Theory and the Efficient Frontier.

## Features
- **Interactive Stock Selection**: Users can input any stock tickers they want to analyze
- **Risk-Return Optimization**: Calculates the optimal asset allocation to maximize risk-adjusted returns (Sharpe Ratio)
- **Historical Data Analysis**: Uses 3 years of historical stock data for accurate predictions
- **Visual Portfolio Allocation**: Generates clear bar charts showing recommended investment percentages
- **Performance Metrics**: Displays expected annual return, volatility, and Sharpe ratio

## Technologies Used
- **Python 3.13**
- **yfinance**: Real-time stock market data
- **NumPy**: Mathematical computations
- **pandas**: Data manipulation
- **SciPy**: Portfolio optimization algorithms
- **matplotlib**: Data visualization

## Installation

1. Clone this repository
2. Install required packages:
"```bash"
pip3 install yfinance pandas numpy scipy matplotlib

## How It Works

1. **Data Collection:** Downloads 3 years of historical stock price data
2. **Return Calculation:** Computes annualized returns and covariance matrix
3. **Optimization:** Uses Sequential Least Squares Programming (SLSQP) to maximize the Sharpe ratio
4. **Visualization:** Generates allocation chart showing optimal investment percentages

## Sample Output

- Expected Annual Return: Shows predicted yearly returns
- Expected Volatility: Indicates portfolio risk level
- Sharpe Ratio: Measures risk-adjusted returns (higher is better)
- Portfolio Allocation Chart: Visual breakdown of recommended investments

## Skills Demonstrated

- Quantitative finance and Modern Portfolio Theory
- Optimization algorithms (convex optimization)
- Statistical analysis and risk modeling
- API integration for financial data
- Data visualization