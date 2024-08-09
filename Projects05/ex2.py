# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.mixture import GaussianMixture
# from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# # 1. Data Preparation: Generate synthetic social network data
# np.random.seed(42)
# n__samples = 300
# n__features = 3

# # Features: [number of friend, number of ]

# data = np.random.rand(n__samples, n__features) * [1000, 500, 100]


# print(" First few rows of the dataset:")
# print(data[:5])

# kmeans = KMeans(n_clusters=3, random_state=42)
# y_kmeans = kmeans.fit_predict(data)

# inertia = []
# for n in range(1, 11):
#     kmeans = KMeans(n_clusters=n, random_state=42)
#     kmeans.fit(data)
#     inertia.append(kmeans.inertia_)
    
# plt.plot(range(1, 11), inertia, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.title(" Elbow Method")
# plt.show()

# plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, cmap="viridis")
# plt.xlabel("Number of Friends")
# plt.ylabel("Number of Likes")
# plt.title("K_Means Clustering")
# plt.show()



# pca = PCA(n_components=2)
# data__pca = pca.fit_transform(data)


# plt.scatter(data__pca[:, 0], data__pca[:, 1], c=y_kmeans, cmap="Viridis")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA on Social Network Data")
# plt.show()

# linked = linkage(data, "ward")

# dendrogram(linkage)
# plt.title("Dendrogram")
# plt.xlabel("Sample")
# plt.ylabel("Euclidean Distance")
# plt.show()

# y__hc = fcluster(linked, t= 3, criterion="maxclust")
# plt.scatter(data[:, 0], data[:, 1], c=y__hc, cmap="Viridis")
# plt.xlabel("Number of Friends")
# plt.ylabel("Number of Likes")
# plt.title("hierarchical Clustering")
# plt.show()

# gmm = GaussianMixture(n_components=3, random_state=42)
# gmm.fit(data)
# y_gmm = gmm.predict(data)
# probs = gmm.predict_proba(data)
# anomalies = probs.max(axis=1) < 0.1

# plt.scatter(data[:, 0], data[:, 1], c=anomalies, cmap="Viridis")
# plt.xlabel("Number of Friends")
# plt.ylabel("Number of Likes")
# plt.title("Anomaly Detection with Gaussian Mixture")
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 1. Data Preparation: Generate synthetic social network data
np.random.seed(42)
n_samples = 300
n_features = 3

# Features: [number of friends, number of likes, number of posts]
data = np.random.rand(n_samples, n_features) * [1000, 500, 100]

# Print the first few rows to understand its structure
print("First few rows of the dataset:")
print(data[:5])

# 2. K-Means Clustering: a. Perform K-Means clustering on the dataset
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(data)

# b. Use the Elbow Method to determine the optimal number of clusters
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# c. Visualize clusters with a scatter plot, coloring data points by cluster labels
plt.scatter(data[:, 0], data[:, 1], c=y_kmeans, cmap='viridis')
plt.xlabel('Number of Friends')
plt.ylabel('Number of Likes')
plt.title('K-Means Clustering')
plt.show()

# 3. Dimensionality Reduction with PCA
# a. Apply Principal Component Analysis (PCA) to reduce the dataset to 2 components
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# b. Visualize the dataset in 2D, coloring data points by K-Means cluster labels
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Social Network Data')
plt.show()

# 4. Hierarchical Clustering: a. Perform Hierarchical clustering on the dataset
linked = linkage(data, 'ward')

# b. Visualize the dendrogram and identify an appropriate number of clusters
dendrogram(linked)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distance')
plt.show()

# c. Compare clusters from Hierarchical clustering with those from K-Means
hc = fcluster(linked, t=3, criterion='maxclust')

# Visualize Hierarchical Clusters
plt.scatter(data[:, 0], data[:, 1], c=hc, cmap='viridis')
plt.xlabel('Number of Friends')
plt.ylabel('Number of Likes')
plt.title('Hierarchical Clustering')
plt.show()

# 5. Anomaly Detection: a. Use the GaussianMixture model to detect anomalies in the data
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(data)
probs = gmm.predict_proba(data)
anomalies = probs.max(axis=1) < 0.1

# b. Identify and visualize anomalies using a scatter plot
plt.scatter(data[:, 0], data[:, 1], c=anomalies, cmap='viridis')
plt.xlabel('Number of Friends')
plt.ylabel('Number of Likes')
plt.title('Anomaly Detection with Gaussian Mixture')
plt.show()