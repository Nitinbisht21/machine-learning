import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Read dataset
data = pd.read_csv("means.csv")

print("Dataset:\n")
print(data)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(data)

print("\nK-Means Labels:")
print(kmeans_labels)

# EM Clustering using Gaussian Mixture Model
em = GaussianMixture(n_components=2, random_state=0)
em.fit(data)
em_labels = em.predict(data)

print("\nEM Labels:")
print(em_labels)

# Plot graphs
plt.figure(figsize=(10,5))

# K-Means plot
plt.subplot(1,2,1)
plt.scatter(data["Age"], data["Income"], c=kmeans_labels)
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering")

# EM plot
plt.subplot(1,2,2)
plt.scatter(data["Age"], data["Income"], c=em_labels)
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("EM Clustering")

plt.tight_layout()
plt.show()