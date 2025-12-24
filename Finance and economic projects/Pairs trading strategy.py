import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
 
# 1. Download historical stock data for two stocks (e.g., Apple and Microsoft)
stock1 = yf.download("AAPL", start="2015-01-01", end="2021-01-01")['Close']
stock2 = yf.download("MSFT", start="2015-01-01", end="2021-01-01")['Close']
 
# 2. Perform cointegration test to check if the stocks are cointegrated
def cointegration_test(series1, series2):
    result = sm.tsa.stattools.coint(series1, series2)
    return result[1]  # p-value
 
# Perform the cointegration test between AAPL and MSFT
p_value = cointegration_test(stock1, stock2)
 
print(f"P-value of cointegration test between AAPL and MSFT: {p_value:.4f}")
 
# If p-value < 0.05, the series are cointegrated and we can proceed
if p_value < 0.05:
    print("The stocks are cointegrated and can be used for pairs trading.")
else:
    print("The stocks are not cointegrated. Consider finding a different pair.")
    
# 3. Calculate the spread between the two stocks (spread = stock1 - beta * stock2)
# Calculate beta using linear regression between the two stocks
X = sm.add_constant(stock2)
model = sm.OLS(stock1, X).fit()
beta = model.params[1]  # Coefficient for stock2
spread = stock1 - beta * stock2
 
# 4. Generate trading signals
# Buy when the spread is below the mean - 1 standard deviation
# Sell when the spread is above the mean + 1 standard deviation
mean_spread = spread.mean()
std_spread = spread.std()
 
# Define entry and exit points
buy_signal = spread < mean_spread - std_spread
sell_signal = spread > mean_spread + std_spread
 
# 5. Plot the spread and the trading signals
plt.figure(figsize=(14, 7))
plt.plot(spread, label='Spread (AAPL - Beta * MSFT)', color='blue')
plt.axhline(mean_spread, color='green', linestyle='--', label='Mean')
plt.axhline(mean_spread - std_spread, color='red', linestyle='--', label='Buy Signal Threshold')
plt.axhline(mean_spread + std_spread, color='purple', linestyle='--', label='Sell Signal Threshold')
plt.scatter(spread.index[buy_signal], spread[buy_signal], marker='^', color='g', label="Buy Signal")
plt.scatter(spread.index[sell_signal], spread[sell_signal], marker='v', color='r', label="Sell Signal")
plt.title("Pairs Trading Strategy (AAPL vs MSFT)")
plt.xlabel('Date')
plt.ylabel('Spread')
plt.legend(loc='best')
plt.show()