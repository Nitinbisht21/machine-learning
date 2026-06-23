import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read dataset
data = pd.read_csv("means.csv")

print("Dataset:")
print(data)

# Elbow Method
wcss = []

for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 6), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means with K=2
model = KMeans(n_clusters=2, random_state=0)
model.fit(data)

# Cluster labels
print("\nCluster Labels:")
print(model.labels_)

# Plot Clusters
plt.scatter(data["Age"], data["Income"], c=model.labels_)
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering")
plt.show()