from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Generate synthetic data for clustering
X, _ = make_moons(n_samples=200, noise=0.1, random_state=42)

# Create a DBSCAN clustering model
dbscan = DBSCAN(eps=0.2, min_samples=5)

# Fit the model to the data
dbscan.fit(X)

# Extract cluster labels (-1 represents noise points)
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise (-1)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'DBSCAN Clustering with {n_clusters} clusters')
plt.show()