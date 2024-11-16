from ucimlrepo import fetch_ucirepo  # Εισαγωγή της βιβλιοθήκης για λήψη δεδομένων από το UCI ML Repository
import numpy as np  # Εισαγωγή της βιβλιοθήκης numpy για αριθμητικές πράξεις
from sklearn.model_selection import train_test_split, cross_val_score  # Εισαγωγή συναρτήσεων για διαχωρισμό δεδομένων και διασταυρούμενη επικύρωση
from sklearn.tree import DecisionTreeClassifier  # Εισαγωγή του ταξινομητή δέντρου απόφασης
import matplotlib.pyplot as plt  # Εισαγωγή της βιβλιοθήκης matplotlib για γραφήματα
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # Εισαγωγή συναρτήσεων για υπολογισμό ακρίβειας και εμφάνιση πίνακα σύγχυσης

# Παίρνουμε το dataset
phishing_websites = fetch_ucirepo(id=327) 

# Αποθηκεύουμε τα χαρακτηριστικά και τις ετικέτες του dataset σε pandas dataframe
#Με την dropna() αφαιρούμε τις ελλιπείς τιμές
X = phishing_websites.data.features.dropna()
y = phishing_websites.data.targets.dropna()

# Οπτικοποίηση της κατανομής των κλάσεων
plt.figure(figsize=(10, 6))
y.value_counts().plot(kind='bar', color='skyblue')
plt.title('Κατανομή κλάσεων')
plt.xlabel('Ομάδα')
plt.ylabel('Συχνότητα')
plt.show()

# Διαχωρίζουμε τα δεδομένα σε training και testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Δημιουργούμε έναν ταξινομητή δέντρου απόφασης
DecisionTree = DecisionTreeClassifier(random_state=1, max_leaf_nodes=15)

# Υπολογίζουμε τα σκορ διασταυρούμενης επικύρωσης για το DecisionTree
Cross_Validation_Score = cross_val_score(DecisionTree, X_train, y_train, cv=10) 
print("Cross Validation Score: ", np.mean(Cross_Validation_Score))

# Εκπαιδεύουμε τον ταξινομητή DecisionTree
DecisionTree.fit(X_train, y_train)

# Κάνουμε προβλέψεις στο testing dataset
y_pred = DecisionTree.predict(X_test)

# Υπολογίζουμε την ακρίβεια του ταξινομητή DecisionTree
Accuracy = accuracy_score(y_test, y_pred)
Confusion_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy: ", Accuracy)

disp = ConfusionMatrixDisplay(confusion_matrix = Confusion_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.show()
