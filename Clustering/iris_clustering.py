import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()

# Create DataFrame (features only)
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Get number of clusters from user
k = int(input("Enter number of clusters (k): "))

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Store cluster labels
X['cluster'] = clusters

# Plot (petal features)
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X['petal length (cm)'],
    X['petal width (cm)'],
    c=X['cluster'],
    cmap='viridis'
)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title(f'K-Means Clusters (k={k}) on Iris Dataset')

plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()