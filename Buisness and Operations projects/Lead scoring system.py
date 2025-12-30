import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
 
# Sample dataset: each row is a lead with features and a label for conversion (1 = converted, 0 = not converted)
data = {
    'PageViews': [5, 10, 2, 15, 3, 8, 1, 12],          # number of website pages viewed
    'TimeOnSite': [2, 8, 1, 10, 1.5, 5, 0.5, 7],       # time spent on site (minutes)
    'EmailOpened': [1, 1, 0, 1, 0, 1, 0, 1],           # did they open marketing emails (1=yes)
    'IndustryScore': [3, 5, 1, 5, 2, 4, 1, 5],         # internal score assigned by industry fit
    'Converted': [0, 1, 0, 1, 0, 1, 0, 1]              # lead conversion status
}
 
df = pd.DataFrame(data)
 
# Features and target
X = df[['PageViews', 'TimeOnSite', 'EmailOpened', 'IndustryScore']]
y = df['Converted']
 
# Split into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
 
# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
 
# Predict lead conversion on test set
y_pred = model.predict(X_test)
 
# Evaluate the model
print("Lead Scoring Classification Report:")
print(classification_report(y_test, y_pred))