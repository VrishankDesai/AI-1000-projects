# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, Binarizer, KBinsDiscretizer, LabelEncoder
 
# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46],
    'Salary': [25000, 30000, 52000, 70000, 62000],
    'Purchased': ['No', 'Yes', 'Yes', 'No', 'Yes']  # Categorical label
}
df = pd.DataFrame(data)
print("Original Data:\n", df)
 
# ---- 1. Polynomial Features ----
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['Age', 'Salary']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['Age', 'Salary']))
print("\nPolynomial Features (Degree 2):\n", poly_df)
 
# ---- 2. Binarization ----
binarizer = Binarizer(threshold=50000)
binarized_salary = binarizer.fit_transform(df[['Salary']])
df['High_Salary'] = binarized_salary
print("\nBinarized 'Salary' (1 if > 50,000):\n", df[['Salary', 'High_Salary']])
 
# ---- 3. Discretization (Bucketing Age) ----
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df['Age_Bin'] = discretizer.fit_transform(df[['Age']])
print("\nDiscretized 'Age' into 3 bins:\n", df[['Age', 'Age_Bin']])
 
# ---- 4. Label Encoding ----
label_encoder = LabelEncoder()
df['Purchased_Encoded'] = label_encoder.fit_transform(df['Purchased'])
print("\nLabel Encoded 'Purchased':\n", df[['Purchased', 'Purchased_Encoded']])