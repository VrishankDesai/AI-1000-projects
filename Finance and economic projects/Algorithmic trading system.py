import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
 
# 1. Download stock data (e.g., Apple stock)
stock_data = yf.download("AAPL", start="2015-01-01", end="2021-01-01")
 
# 2. Calculate the rolling mean and rolling standard deviation
window = 20  # 20-day window
stock_data['Rolling Mean'] = stock_data['Close'].rolling(window=window).mean()
stock_data['Rolling Std'] = stock_data['Close'].rolling(window=window).std()
 
# 3. Define the trading strategy: Buy when the price is below the mean - 1 std, sell when above the mean + 1 std
stock_data['Buy Signal'] = np.where(stock_data['Close'] < (stock_data['Rolling Mean'] - stock_data['Rolling Std']), 1, 0)
stock_data['Sell Signal'] = np.where(stock_data['Close'] > (stock_data['Rolling Mean'] + stock_data['Rolling Std']), 1, 0)
 
# 4. Simulate a simple trading strategy
initial_cash = 10000
cash = initial_cash
shares = 0
portfolio_value = []
 
for date, row in stock_data.iterrows():
    # Execute Buy Signal
    if row['Buy Signal'] == 1 and cash > row['Close']:
        shares = cash // row['Close']  # Buy as many shares as possible
        cash -= shares * row['Close']  # Deduct cash spent
    # Execute Sell Signal
    elif row['Sell Signal'] == 1 and shares > 0:
        cash += shares * row['Close']  # Sell all shares
        shares = 0
    # Track portfolio value
    portfolio_value.append(cash + shares * row['Close'])
 
stock_data['Portfolio Value'] = portfolio_value
 
# 5. Plot the portfolio value and stock price
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Close'], label='Stock Price', color='blue')
plt.plot(stock_data['Portfolio Value'], label='Portfolio Value', color='green')
plt.title('Stock Price and Portfolio Value')
plt.xlabel('Date')
plt.ylabel('Value (USD)')
plt.legend()
plt.show()