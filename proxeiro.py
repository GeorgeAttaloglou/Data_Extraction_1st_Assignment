import pandas as pd
import matplotlib.pyplot as plt

# Δεδομένα
leaf_nodes = [3, 5, 10, 15, 50, 100, 300, 450, 750, 10000]
accuracies = [
    0.898236092265943, #3
    0.898236092265943, #5
    0.9163274536408865, #10
    0.9303482587064676, #15
    0.942107643600181, #50
    0.9502487562189055, #100
    0.9574853007688828, #300
    0.9633649932157394, #450
    0.9629127091813658, #750
    0.9629127091813658 #10000
]

# Δημιουργία πίνακα
data = {
    'Max Leafs': leaf_nodes,
    'Model Accuracy': accuracies
}
df = pd.DataFrame(data)

# Εμφάνιση πίνακα
print(df)

# Δημιουργία γραφήματος
plt.figure(figsize=(10, 6))
plt.plot(leaf_nodes, accuracies, marker='o', linestyle='-', color='blue', label='Model Accuracy')

# Διαμόρφωση γραφήματος
plt.title('Model Accuracy vs. Number of Max Leafs', fontsize=16)
plt.xlabel('Number of Max Leafs', fontsize=14)
plt.ylabel('Model Accuracy', fontsize=14)
plt.xscale('log')  # Λογαριθμική κλίμακα στον άξονα x για καλύτερη απεικόνιση
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(leaf_nodes, labels=leaf_nodes, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)

# Εμφάνιση γραφήματος
plt.show()
