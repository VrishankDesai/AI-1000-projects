import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
 
# 1. Define the Black-Scholes formula for European call option pricing
def black_scholes_call(S, K, T, r, sigma):
    """
    S: Current stock price
    K: Strike price of the option
    T: Time to expiration in years
    r: Risk-free interest rate
    sigma: Volatility of the underlying stock
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate the call option price
    call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call_price
 
# 2. Example parameters for a European Call Option
S = 100  # Current stock price (USD)
K = 105  # Strike price (USD)
T = 1    # Time to expiration (1 year)
r = 0.05 # Risk-free interest rate (5%)
sigma = 0.2  # Volatility (20%)
 
# 3. Calculate the call option price
call_price = black_scholes_call(S, K, T, r, sigma)
print(f"Call Option Price: ${call_price:.2f}")
 
# 4. Plot the option price as a function of stock price (S)
stock_prices = np.linspace(50, 150, 100)
option_prices = [black_scholes_call(s, K, T, r, sigma) for s in stock_prices]
 
plt.plot(stock_prices, option_prices, label="Call Option Price")
plt.title("European Call Option Pricing (Black-Scholes Model)")
plt.xlabel("Stock Price (S)")
plt.ylabel("Call Option Price")
plt.grid(True)
plt.legend()
plt.show()