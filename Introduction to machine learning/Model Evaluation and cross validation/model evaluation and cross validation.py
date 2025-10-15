from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,KFold
from sklearn.ensemble import RandomForestClassifier

#Load datasets
data = load_iris()
X,y = data.data, data.target

#Initialize classifier
model = RandomForestClassifier(random_state=42)

#Peform K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

#Output results
print("Cross validation scores: ",cv_scores)
print("Mean accuracy: ",cv_scores.mean())