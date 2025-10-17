from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
data = load_iris()
X,y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training logistic regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

#Predicting using logistic regression
y_pred_log_reg = log_reg.predict(X_test)

# Evaluate logistic regression model
print("Logistic Regression Model Evaluation:")

#Evaluate knn
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Model Accuracy with k={best_k}: ", accuracy_knn)

#Detailed comparison
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))



# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Experimenting with different values of k
for k in range(1, 11):
    # Initialize the KNN classifier with current k
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions on the test data
    y_pred = knn.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy for k={k}: {accuracy:.2f}')
 