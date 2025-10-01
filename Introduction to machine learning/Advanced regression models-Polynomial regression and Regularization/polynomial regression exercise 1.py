import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Loading california housing dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

#Select feature(median income) and target(median house value)
X=df[['MedInc']]
y=df[['MedHouseVal']]

#Transform feaature into polynomial feature
poly = PolynomialFeatures(degree=2,include_bias=False)
X_poly = poly.fit_transform(X)

#Fit polynomial regression model
model = LinearRegression()
model.fit(X_poly,y)

#Make predictions
y_pred = model.predict(X_poly)

#Plot actual vs predicted values
plt.figure(figsize=(10,6))
plt.scatter(X,y,color="blue",label="Actual data",alpha=0.5)
plt.scatter(X,y_pred,color="red",label="Predicted curve",alpha=0.5)
plt.title("Polynomial Regression")
plt.xlabel("Median income in California")
plt.ylabel("Median house value in California")
plt.legend()
plt.show()

#Evaluate model preference
mse=mean_squared_error(y,y_pred)
print("Mean Squared Error is: ",mse)