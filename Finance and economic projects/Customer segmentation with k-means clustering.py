import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
 
# 1. Simulate customer data
np.random.seed(42)
data = {
    'account_balance': np.random.normal(5000, 1500, 1000),  # USD
    'num_transactions': np.random.randint(1, 20, 1000),
    'loan_amount': np.random.normal(20000, 5000, 1000),  # USD
    'age': np.random.randint(18, 70, 1000)
}
 
df = pd.DataFrame(data)
 
# 2. Preprocessing
X = df[['account_balance', 'num_transactions', 'loan_amount', 'age']]
 
# 3. Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 
# 4. Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Assume 4 segments
df['Segment'] = kmeans.fit_predict(X_scaled)
 
# 5. Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['account_balance'], df['loan_amount'], c=df['Segment'], cmap='viridis', alpha=0.6)
plt.title('Customer Segmentation')
plt.xlabel('Account Balance (USD)')
plt.ylabel('Loan Amount (USD)')
plt.colorbar(label='Segment')
plt.show()
 
# 6. Show summary of each segment
segment_summary = df.groupby('Segment').mean()
print("Segment Summary (Average Features per Segment):\n")
print(segment_summary)