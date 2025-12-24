import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
 
# 1. Download historical stock data (e.g., Apple stock)
stock_data = yf.download("AAPL", start="2018-01-01", end="2021-01-01")['Close']
 
# 2. Calculate the Rate of Change (ROC)
window = 14  # 14-day window
roc = stock_data.pct_change(periods=window) * 100  # Rate of Change in percentage
 
# 3. Generate buy and sell signals based on ROC
buy_signal = roc > 0  # Buy when ROC is positive (indicating upward momentum)
sell_signal = roc < 0  # Sell when ROC is negative (indicating downward momentum)
 
# 4. Plot the stock price and momentum signals
plt.figure(figsize=(14, 7))
 
# Plot the stock price
plt.subplot(2, 1, 1)
plt.plot(stock_data, label="Stock Price", color='blue')
plt.scatter(stock_data.index[buy_signal], stock_data[buy_signal], marker='^', color='green', label="Buy Signal")
plt.scatter(stock_data.index[sell_signal], stock_data[sell_signal], marker='v', color='red', label="Sell Signal")
plt.title("Momentum Trading Strategy - Stock Price")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
 
# Plot the Rate of Change (ROC)
plt.subplot(2, 1, 2)
plt.plot(roc, label="Rate of Change (ROC)", color='purple')
plt.axhline(0, color='black', linestyle='--', label="Zero Line")
plt.title("Rate of Change for Momentum Trading Strategy")
plt.xlabel('Date')
plt.ylabel('ROC (%)')
plt.legend()
 
plt.tight_layout()
plt.show()