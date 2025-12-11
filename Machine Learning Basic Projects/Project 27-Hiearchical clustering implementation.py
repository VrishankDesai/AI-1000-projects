# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
 
# Generate synthetic 2D data
X, _ = make_blobs(n_samples=100, centers=3, cluster_std=1.2, random_state=42)
 
# Plot the raw data
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], s=50, color='gray')
plt.title("Raw Data Points")
plt.grid(True)
plt.tight_layout()
plt.show()
 
# ---- 1. Create Dendrogram ----
linked = linkage(X, method='ward')  # 'ward' minimizes variance within clusters
 
plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=20, leaf_rotation=45., leaf_font_size=12.)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index or Cluster Size")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
 
# ---- 2. Apply Agglomerative Clustering ----
model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
labels = model.fit_predict(X)
 
# Plot clustered data
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=50)
plt.title("Clusters from Agglomerative Clustering")
plt.grid(True)
plt.tight_layout()
plt.show()