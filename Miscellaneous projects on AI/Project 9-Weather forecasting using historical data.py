import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,],
    'Temperature': [30, 32, 31, 29, 28, 27, 26, 25, 24, 23],
    'Humidity': [70, 65, 68, 72, 75, 80, 78, 77, 74, 73],
    'WindSpeed': [10, 12, 11, 9, 8, 7, 6, 5, 4, 3],
    'Precipitation': [0, 0, 1, 2, 1, 0, 3, 2, 1, 0],
    'NextDayTemperature': [32, 34, 30, 28, 27, 26, 25, 24, 23, 22]
}

df = pd.DataFrame(data)

X = df[['Temperature', 'Humidity', 'WindSpeed', 'Precipitation']]
y = df['NextDayTemperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Temperature', marker='o')
plt.plot(y_pred, label='Predicted Temperature', marker='x')
plt.xlabel('Test Sample Index')
plt.ylabel('Next Day Temperature')
plt.title('Actual vs Predicted Next Day Temperature')
plt.legend()
plt.show()

new_data = pd.DataFrame({
    'Temperature': [29, 27],
    'Humidity': [68, 75],
    'WindSpeed': [11, 8],
    'Precipitation': [1, 0]
})

predicted_temperature = model.predict(new_data)
print(f"Predicted temperautres: {predicted_temperature[0]:.2f}C")