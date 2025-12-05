import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Load California housing dataset
housing = fetch_california_housing(as_frame=True)

#Create dataframe from the dataset
df = housing.frame

print("California Housing Dataset:")
print(df.head())

#Fetures(independent variables) and Target(dependent variable)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model using Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-2 score: {r2}")

print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nCoefficients for each features:")
print(coef_df)

#Test model with new data
new_data = pd.DataFrame({
    'MedInc': [8.3252],
    'HouseAge': [41],
    'AveRooms': [6.9841],
    'AveBedrms': [1.0238],
    'Population': [322],
    'AveOccup': [2.5556],
    'Latitude': [37.88],
    'Longitude': [-122.23]
})

predicted_price = model.predict(new_data)
print(f"\nPredicted House price: {predicted_price[0]:,.2f}")