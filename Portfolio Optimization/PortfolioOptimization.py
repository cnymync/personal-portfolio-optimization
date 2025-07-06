import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize

tickers = ['IVV', 'AMZN', 'AAPL', 'VEU', 'GOLD'] 
#tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI'] 

end_date = datetime.today()

start_date = end_date - timedelta(days = 1.5*365)
print(start_date)

close_df = pd.DataFrame()

for ticker in tickers: 
    data = yf.download(ticker, start= start_date, end= end_date)
    close_df[ticker] = data['Close']
    
#Lognormal Returns (additive) 

log_returns = np.log(close_df / close_df.shift(1))

log_returns = log_returns.dropna()
log_returns

# Covariance Matrix, measure total risk of portfolio 

cov_matrix = log_returns.cov()*252 

# Calculate portfolio performance metrics 
# 1. Portfolio Standard Deviation 
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)
    
# 2. Expected Returns (Key assumption: based expected returns 
# on historical returns where avg securities in the past is similar)
def expected_return(weights, log_returns):
    return np.sum(log_returns.mean()*weights)*252

# 3. Sharpe Ratio 
def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return(expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

# Portfolio Optimization 
# Implement risk free rate (assume 2%) via textbook 
# Or use federal reserve API
# Website: https://fredaccount.stlouisfed.org/apikey

from fredapi import Fred

fred = Fred(api_key = 'b006c12cf31afd09359d05a22c8aed02')
ten_year_treasury_rate = fred.get_series_latest_release('GS10')/100
risk_free_rate = ten_year_treasury_rate.iloc[-1]
print(risk_free_rate)

#Implement negative sharpe ratio

def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
    
    # Setting constraints and bounds (ensure weights all sum to 1)
# Reason why we apply "0" bounds is because we are assuming we are only 
# going long or purchasing assets and never go short/selling assets 
# 0.5 implies that we cannot have more than 50% from a single security in the portfolio
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.6) for i in range(len(tickers))]

# Set the initial weights 
initial_weights = np.array([1/len(tickers)]*len(tickers))
print(initial_weights)

# Optimize weights to maximize sharpe ratio 
# SLSQP stands for sequential least squares quadratic programming, 
# which is a numerical optimization technique suitable for solving 
# non-linear optimization problems with constraints

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args = 
                             (log_returns, cov_matrix, risk_free_rate),
                             method = 'SLSQP',
                             constraints = constraints,
                             bounds = bounds)
                             
# Obtain Optimal Weights 
optimal_weights = optimized_results.x
print(optimal_weights)

print("Optimal Weights:")
for ticker, weight in zip(tickers, optimal_weights):
    print(f"{ticker}: {weight:.4f}")
print()

optimal_portfolio_return = expected_return(optimal_weights, log_returns) 
optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)


print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
print (f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

#Visualize 
import matplotlib.pyplot as plt 

plt.figure(figsize=(10,6))
plt.bar(tickers, optimal_weights)

plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')