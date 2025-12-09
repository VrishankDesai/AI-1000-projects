import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

#Sample historical stock price data
data = {
    'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'),
    'Close': [150, 152, 153, 151, 155, 157, 158, 160, 162, 165]
}   

#Convert to DataFrame
df = pd.DataFrame(data)

df['Date'] = df['Date'].map(pd.Timestamp.toordinal)

print("Stock Price Data:")
print(df)

X = df[['Date']]
y = df['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R2 Score: {r2}")

plt.figure(figsize=(10, 6))
plt.plot(X_test, y_pred, label='Actual Stock Prices', marker='o')
plt.plot(X_test, y_pred, label='Predicted Stock Prices', marker='x')
plt.xlabel('Date(Ordinal)')
plt.ylabel('Stock Price($)')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.show()

future_date = pd.Timestamp(start='2024-01-01').toordinal()
future_date = pd.DataFrame({'Date': [future_date]})

predicted_price = model.predict(future_date)
print(f"\nPredicted stock price: ${predicted_price[0]:.2f}")