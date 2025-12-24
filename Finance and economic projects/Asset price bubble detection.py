import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
 
# 1. Download stock data (e.g., Apple stock) to analyze potential bubbles
stock_data = yf.download("AAPL", start="2015-01-01", end="2021-01-01")['Close']
 
# 2. Calculate moving averages (50-day and 200-day moving averages)
short_window = 50
long_window = 200
stock_data['SMA50'] = stock_data.rolling(window=short_window).mean()
stock_data['SMA200'] = stock_data.rolling(window=long_window).mean()
 
# 3. Detect potential bubbles using price deviation from the long-term average
stock_data['Price_Deviation'] = stock_data['Close'] / stock_data['SMA200']
 
# Define a threshold for identifying a bubble (e.g., price is 20% above the 200-day moving average)
threshold = 1.2  # Price deviating 20% above the SMA200 might signal a bubble
stock_data['Bubble_Detection'] = np.where(stock_data['Price_Deviation'] > threshold, 1, 0)
 
# 4. Plot the price data, moving averages, and bubble detection signals
plt.figure(figsize=(14, 7))
 
# Plot stock price and moving averages
plt.plot(stock_data['Close'], label="Stock Price (AAPL)", color='blue')
plt.plot(stock_data['SMA50'], label="50-Day Moving Average", color='orange', linestyle='--')
plt.plot(stock_data['SMA200'], label="200-Day Moving Average", color='green', linestyle='--')
 
# Highlight bubble detection signals
plt.scatter(stock_data.index[stock_data['Bubble_Detection'] == 1], stock_data['Close'][stock_data['Bubble_Detection'] == 1], 
            marker='o', color='red', label='Potential Bubble')
 
# Title and labels
plt.title("Asset Price Bubble Detection (AAPL Stock)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend(loc="best")
plt.grid(True)
plt.show()
 
# 5. Output the periods with detected bubbles
bubble_dates = stock_data[stock_data['Bubble_Detection'] == 1].index
print("Detected Potential Bubbles in Asset Price:\n")
print(bubble_dates)