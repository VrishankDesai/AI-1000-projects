import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
# 1. Simulate bond yield data for different maturities (short-term to long-term)
maturities = np.array([1, 2, 3, 5, 7, 10, 20, 30])  # Bond maturities in years
# Simulating yields (in %) for each maturity. Normally, long-term bonds have higher yields than short-term ones.
yields = np.array([1.5, 1.7, 1.8, 2.0, 2.3, 2.5, 3.0, 3.5]) + np.random.normal(0, 0.1, len(maturities))  # Adding some randomness
 
# 2. Plot the bond yield curve
plt.figure(figsize=(10, 6))
plt.plot(maturities, yields, marker='o', color='b', label='Bond Yield Curve')
plt.title('Bond Yield Curve Analysis')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.xticks(maturities)  # Ensures that all maturity years are shown on the x-axis
plt.yticks(np.arange(1, 4, 0.25))  # Adjust y-axis ticks for better visualization
plt.legend(loc="best")
plt.show()
 
# 3. Analyze the shape of the yield curve
if np.all(yields[1:] > yields[:-1]):
    print("The yield curve is upward sloping, indicating normal market conditions (long-term rates > short-term rates).")
elif np.all(yields[1:] < yields[:-1]):
    print("The yield curve is downward sloping (inverted), indicating potential recessionary fears.")
else:
    print("The yield curve is flat or has mixed slopes, suggesting uncertain or transitioning market conditions.")