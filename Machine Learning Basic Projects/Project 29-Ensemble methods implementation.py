# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
 
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# ---- 1. Bagging: Random Forest ----
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.2f}")
 
# ---- 2. Boosting: Gradient Boosting ----
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_preds):.2f}")
 
# ---- 3. Voting Classifier (Hard Voting) ----
log_clf = LogisticRegression(max_iter=200)
svc_clf = SVC(probability=True)  # Enable probability for soft voting
voting_clf = VotingClassifier(estimators=[
    ('lr', log_clf), ('svc', svc_clf), ('rf', rf)
], voting='hard')
 
voting_clf.fit(X_train, y_train)
voting_preds = voting_clf.predict(X_test)
print(f"Voting Classifier Accuracy: {accuracy_score(y_test, voting_preds):.2f}")