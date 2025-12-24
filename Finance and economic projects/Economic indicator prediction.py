import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
 
# 1. Simulate historical economic data (GDP growth and related indicators)
np.random.seed(42)
 
years = np.arange(2000, 2021)
gdp_growth = np.random.normal(2, 0.5, len(years))  # Simulate GDP growth as a normal distribution around 2% with some variability
interest_rate = np.random.normal(3, 1, len(years))  # Simulate interest rates
inflation = np.random.normal(2, 0.8, len(years))  # Simulate inflation rates
 
# Create a DataFrame with simulated data
df = pd.DataFrame({
    'Year': years,
    'GDP_Growth': gdp_growth,
    'Interest_Rate': interest_rate,
    'Inflation': inflation
})
 
# 2. Prepare the data for regression
X = df[['Interest_Rate', 'Inflation']]  # Independent variables: Interest Rate and Inflation
y = df['GDP_Growth']  # Dependent variable: GDP Growth
 
# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 4. Train the regression model
model = LinearRegression()
model.fit(X_train, y_train)
 
# 5. Make predictions
y_pred = model.predict(X_test)
 
# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
 
# 7. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted GDP Growth")
plt.xlabel("Actual GDP Growth")
plt.ylabel("Predicted GDP Growth")
plt.show()
 
# 8. Display the coefficients
print(f"Model Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")