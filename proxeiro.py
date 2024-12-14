import matplotlib.pyplot as plt
import numpy as np

# Δεδομένα για Max Leaf Nodes
leaf_nodes = [3, 5, 10, 15, 50, 100, 300, 450, 750, 1000]
accuracy_leaf = [0.8982, 0.8982, 0.9163, 0.9303, 0.9421, 0.9502, 0.9575, 0.9634, 0.9629, 0.9629]
precision_leaf = [0.88, 0.88, 0.88, 0.93, 0.94, 0.95, 0.97, 0.96, 0.96, 0.96]
recall_leaf = [0.90, 0.90, 0.94, 0.92, 0.93, 0.93, 0.94, 0.96, 0.96, 0.96]
f1_leaf = [0.89, 0.89, 0.91, 0.92, 0.94, 0.94, 0.95, 0.96, 0.96, 0.96]

# Δεδομένα για Neighbors
neighbors = [1, 3, 5, 10, 15, 20, 50]
accuracy_neighbors = [0.89, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
precision_neighbors = [0.87, 0.89, 0.90, 0.92, 0.93, 0.94, 0.95]
recall_neighbors = [0.88, 0.90, 0.91, 0.93, 0.93, 0.94, 0.96]
f1_neighbors = [0.87, 0.89, 0.91, 0.92, 0.93, 0.94, 0.95]

# Accuracy Comparison
plt.figure()
plt.plot(leaf_nodes, accuracy_leaf, label="Max Leaf Nodes", marker="o")
plt.plot(neighbors, accuracy_neighbors, label="Neighbors", marker="s")
plt.title("Accuracy Comparison")
plt.xlabel("Parameter Value")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

# Precision Comparison
plt.figure()
plt.plot(leaf_nodes, precision_leaf, label="Max Leaf Nodes", marker="o")
plt.plot(neighbors, precision_neighbors, label="Neighbors", marker="s")
plt.title("Precision Comparison")
plt.xlabel("Parameter Value")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.show()

# Recall Comparison
plt.figure()
plt.plot(leaf_nodes, recall_leaf, label="Max Leaf Nodes", marker="o")
plt.plot(neighbors, recall_neighbors, label="Neighbors", marker="s")
plt.title("Recall Comparison")
plt.xlabel("Parameter Value")
plt.ylabel("Recall")
plt.legend()
plt.grid()
plt.show()

# F1-Score Comparison
plt.figure()
plt.plot(leaf_nodes, f1_leaf, label="Max Leaf Nodes", marker="o")
plt.plot(neighbors, f1_neighbors, label="Neighbors", marker="s")
plt.title("F1-Score Comparison")
plt.xlabel("Parameter Value")
plt.ylabel("F1-Score")
plt.legend()
plt.grid()
plt.show()
