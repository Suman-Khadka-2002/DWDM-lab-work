from sklearn.datasets import make_blobs
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data for clustering
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Create k-medoids algorithm instance
amount_clusters = 4
initial_medoids = [0, 1, 2, 3]  # Initial medoids (you can choose any data points as initial medoids)
kmedoids_instance = kmedoids(X, initial_medoids)

# Run cluster analysis and get results
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

# Convert cluster results to numpy array
clusters = np.array(clusters)

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster in clusters:
    plt.scatter(X[cluster][:, 0], X[cluster][:, 1], s=50)
plt.scatter(X[medoids][:, 0], X[medoids][:, 1], c='red', marker='*', s=200, label='Medoids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Medoids Clustering')
plt.legend()
plt.show()