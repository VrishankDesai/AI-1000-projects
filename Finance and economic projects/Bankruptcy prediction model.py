import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# 1. Simulate financial data (e.g., debt-to-equity, liquidity ratio, profitability)
np.random.seed(42)
data = {
    'debt_to_equity': np.random.normal(1.5, 0.5, 1000),  # Debt-to-equity ratio
    'current_ratio': np.random.normal(1.2, 0.3, 1000),  # Current ratio (liquidity ratio)
    'return_on_assets': np.random.normal(0.05, 0.02, 1000),  # Return on assets
    'interest_coverage': np.random.normal(3, 1.5, 1000),  # Interest coverage ratio
    'bankruptcy': np.random.choice([0, 1], 1000)  # 0 = Non-bankrupt, 1 = Bankrupt
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df.drop('bankruptcy', axis=1)
y = df['bankruptcy']
 
# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Logistic Regression model to predict bankruptcy
model = LogisticRegression()
model.fit(X_train, y_train)
 
# 6. Make predictions
y_pred = model.predict(X_test)
 
# 7. Evaluate the model
print("Bankruptcy Prediction Report:\n")
print(classification_report(y_test, y_pred))
 
# 8. Plot the feature importance (coefficients of the logistic regression)
coefficients = model.coef_[0]
features = X.columns
 
plt.figure(figsize=(10, 6))
plt.bar(features, coefficients)
plt.title("Feature Importance in Bankruptcy Prediction")
plt.xlabel("Financial Ratios")
plt.ylabel("Coefficient Value")
plt.show()
 
# 9. Predict bankruptcy for a new company based on financial ratios
new_data = np.array([[2.0, 1.1, 0.04, 2]])  # Example: debt_to_equity=2.0, current_ratio=1.1, return_on_assets=4%, interest_coverage=2
new_data_scaled = scaler.transform(new_data)
predicted_bankruptcy = model.predict(new_data_scaled)
print(f"\nPredicted Bankruptcy: {'Yes' if predicted_bankruptcy[0] == 1 else 'No'}")