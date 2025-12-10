# Import required libraries
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
 
# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
 
# Convert to DataFrame for clarity
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
 
# ---- 1. Univariate Selection (ANOVA F-test) ----
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)
selected_features_univariate = [feature_names[i] for i in selector.get_support(indices=True)]
 
print("Top 2 Features (Univariate Selection - ANOVA F-test):")
print(selected_features_univariate)
 
# ---- 2. Recursive Feature Elimination (RFE) ----
rfe_model = RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=2)
rfe_model.fit(X, y)
selected_features_rfe = [feature_names[i] for i in range(len(feature_names)) if rfe_model.support_[i]]
 
print("\nTop 2 Features (Recursive Feature Elimination):")
print(selected_features_rfe)
 
# ---- 3. Feature Importance from Random Forest ----
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)
feature_importances = rf_model.feature_importances_
 
# Rank and select top 2
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)
 
print("\nFeature Importances (Random Forest):")
print(importance_df)
 
# Display top 2
top_features_rf = importance_df["Feature"].values[:2]
print("\nTop 2 Features (Random Forest Importance):")
print(top_features_rf)