import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
# 1. Simulate customer data
np.random.seed(42)
data = {
    'account_balance': np.random.normal(5000, 1500, 1000),  # USD
    'age': np.random.randint(18, 70, 1000),
    'num_transactions': np.random.randint(1, 20, 1000),
    'years_with_bank': np.random.randint(1, 20, 1000),
    'churn': np.random.choice([0, 1], 1000)  # 0 = stay, 1 = churn
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df.drop('churn', axis=1)
y = df['churn']
 
# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# 6. Evaluate the model
y_pred = model.predict(X_test)
print("Customer Churn Prediction Report:\n")
print(classification_report(y_test, y_pred))
 
# 7. Predict churn for a new customer
new_customer = np.array([[4500, 35, 10, 5]])  # Example customer: balance = 4500, age = 35, 10 transactions, 5 years with the bank
new_customer_scaled = scaler.transform(new_customer)
predicted_churn = model.predict(new_customer_scaled)
print(f"\nPredicted Churn: {'Churn' if predicted_churn[0] == 1 else 'No Churn'}")