import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
 
# Simulated dataset: sensor readings and failure status (1 = failure, 0 = normal)
data = {
    'Temperature': [70, 75, 80, 85, 90, 95, 100, 105],
    'Vibration': [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
    'RPM': [1500, 1600, 1650, 1700, 1800, 1850, 1900, 1950],
    'Failed': [0, 0, 0, 0, 1, 1, 1, 1]
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df[['Temperature', 'Vibration', 'RPM']]
y = df['Failed']
 
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
 
# Predict failure
y_pred = model.predict(X_test)
 
# Output model performance
print("Predictive Maintenance Classification Report:")
print(classification_report(y_test, y_pred))
 
# Example: Predict failure for new sensor reading
new_reading = pd.DataFrame({'Temperature': [92], 'Vibration': [0.9], 'RPM': [1820]})
failure_risk = model.predict_proba(new_reading)[0][1]
print(f"\nPredicted Failure Risk: {failure_risk:.2%}")