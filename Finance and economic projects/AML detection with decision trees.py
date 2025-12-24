import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
 
# 1. Simulate financial transaction data
np.random.seed(42)
data = {
    'transaction_amount': np.random.normal(5000, 2000, 1000),  # USD
    'transaction_frequency': np.random.randint(1, 10, 1000),  # number of transactions per day
    'customer_age': np.random.randint(18, 70, 1000),
    'transaction_location': np.random.choice(['Domestic', 'International'], 1000),
    'previous_suspicious_activity': np.random.choice([0, 1], 1000),  # 0 = no, 1 = yes
    'is_suspicious': np.random.choice([0, 1], 1000)  # 0 = no, 1 = suspicious
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
df['transaction_location'] = df['transaction_location'].map({'Domestic': 0, 'International': 1})
X = df.drop('is_suspicious', axis=1)
y = df['is_suspicious']
 
# 3. Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['transaction_amount', 'transaction_frequency', 'customer_age']])
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Decision Tree model for detection
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
 
# 6. Evaluate the model
y_pred = model.predict(X_test)
print("AML Detection Model Report:\n")
print(classification_report(y_test, y_pred))
 
# 7. Predict suspicious activity for a new transaction
new_transaction = np.array([[7000, 8, 45, 1, 1]])  # Example: high transaction amount, frequent international transactions, previous suspicious activity
new_transaction_scaled = scaler.transform(new_transaction)
predicted_suspicious = model.predict(new_transaction_scaled)
print(f"\nPredicted Suspicious Activity: {'Suspicious' if predicted_suspicious[0] == 1 else 'Not Suspicious'}")