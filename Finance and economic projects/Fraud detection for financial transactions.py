import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
 
# 1. Simulate financial transaction data (features: transaction amount, time, user behavior)
np.random.seed(42)
n_samples = 1000
 
# Simulated features: transaction amount, user ID, transaction time (random for simulation)
data = {
    'transaction_amount': np.random.normal(150, 50, n_samples),
    'user_id': np.random.randint(1, 100, n_samples),
    'transaction_time': np.random.randint(1, 24, n_samples),  # Hour of the day
    'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud rate
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
 
# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# 6. Evaluate the model
y_pred = model.predict(X_test)
print("Fraud Detection Model Report:\n")
print(classification_report(y_test, y_pred))
 
# 7. Predict fraud for a new transaction
new_transaction = np.array([[200, 12, 10]])  # Example: $200, transaction time 12 PM
new_transaction_scaled = scaler.transform(new_transaction)
predicted_fraud = model.predict(new_transaction_scaled)
print(f"\nPredicted Fraud: {'Fraud' if predicted_fraud[0] == 1 else 'Legitimate'}")
 
# 8. Visualize feature importance (optional)
feature_importances = model.feature_importances_
plt.bar(X.columns, feature_importances)
plt.title("Feature Importance in Fraud Detection")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()