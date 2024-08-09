import numpy as np
import matplotlib.pyplot as plt
# Python -m pip install scikit-learn for Windows
from sklearn.semi_supervised import LabelPropagation
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import networkx as nx # pip install networkx

# 1. Data Preparation
# Generate synthetic email data
np.random.seed(42)
n_samples = 300
n_features = 4

# Features: [number of words, number of attachments, presence of keyword (0 or 1), sender's domain (0 to 9)]
data = np.random.rand(n_samples, n_features) * [1000, 10, 1, 10]
labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]) # 0: not spam, 1: spam
labels[np.random.choice(n_samples, size=200, replace=False)] = -1 # Marking 200 samples as unlabeled

# Print the first few rows to understand its structure
print("First few rows of the dataset:")
print(data[:5])

# 2. Label Propagation
label_prop_model = LabelPropagation()
label_prop_model.fit(data, labels)
predicted_labels = label_prop_model.transduction_

# Evaluate model's performance on the initially labeled data
known_labels = labels != -1
accuracy = accuracy_score(labels[known_labels], predicted_labels[known_labels])
print(f"Label Propagation accuracy: {accuracy}")
# 3. Graph-Based Semi-Supervised Learning
# Construct a similarity graph
G = nx.Graph()
for i in range(n_samples):
    for j in range(i+1, n_samples):
        similarity = np.exp(-np.sum((data[i] - data[j])**2))
        if similarity > 0.5:  # Threshold for creating an edge
            G.add_edge(i, j, weight=similarity)

# Apply a graph-based semi-supervised learning algorithm to classify emails
# Using the same Label Propagation model since it works on graph structures
predicted_labels_graph = label_prop_model.transduction_

# Visualize the graph with nodes colored by their predicted labels
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=predicted_labels_graph, cmap=plt.get_cmap('viridis'), with_labels=False, node_size=50)
plt.title("Graph-Based Semi-Supervised Learning")
plt.show()

# 4. Dimensionality Reduction with t-SNE
tsne = TSNE(n_components=2)
data_tsne = tsne.fit_transform(data)

# Visualize t-SNE results
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=predicted_labels, cmap='viridis')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('t-SNE on Email Dataset')
plt.show()

# 5. Anomaly Detection with Isolation Forest
isolation_forest = IsolationForest(contamination=0.1)
anomaly_labels = isolation_forest.fit_predict(data)

# Visualize Anomalies
plt.scatter(data[:, 0], data[:, 1], c=anomaly_labels, cmap='viridis')
plt.xlabel('Number of Words')
plt.ylabel('Number of Attachments')
plt.title('Anomaly Detection with Isolation Forest')
plt.show()
