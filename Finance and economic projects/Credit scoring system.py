import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
 
# 1. Simulated borrower data
np.random.seed(42)
data = {
    'credit_score': np.random.randint(300, 850, 1000),
    'annual_income': np.random.normal(60000, 15000, 1000),  # In USD
    'loan_amount': np.random.normal(25000, 7000, 1000),  # In USD
    'debt_to_income_ratio': np.random.uniform(0.05, 0.45, 1000),
    'previous_default': np.random.choice([0, 1], 1000),  # 0 = no, 1 = yes
    'employment_status': np.random.choice(['Employed', 'Unemployed'], 1000),
    'credit_approved': np.random.choice([0, 1], 1000)  # 0 = not approved, 1 = approved
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
df['employment_status'] = df['employment_status'].map({'Employed': 1, 'Unemployed': 0})
X = df.drop('credit_approved', axis=1)
y = df['credit_approved']
 
# 3. Scaling numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio']])
 
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# 5. Decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
 
# 6. Evaluate the model
y_pred = model.predict(X_test)
print("Credit Scoring System Report:\n")
print(classification_report(y_test, y_pred))
 
# 7. Predict credit approval for a new borrower
new_borrower = np.array([[700, 65000, 20000, 0.2]])  # Example borrower data
new_borrower_scaled = scaler.transform(new_borrower)
predicted_approval = model.predict(new_borrower_scaled)
print(f"\nPredicted Credit Approval: {'Approved' if predicted_approval[0] == 1 else 'Not Approved'}")