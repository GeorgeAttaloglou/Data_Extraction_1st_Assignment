from ucimlrepo import fetch_ucirepo  # Εισαγωγή της βιβλιοθήκης για λήψη δεδομένων από το UCI ML Repository
import numpy as np  # Εισαγωγή της βιβλιοθήκης numpy για αριθμητικές πράξεις
from sklearn.model_selection import KFold, cross_val_score, train_test_split  # Εισαγωγή συναρτήσεων για διαχωρισμό δεδομένων και διασταυρούμενη επικύρωση
from sklearn.neighbors import KNeighborsClassifier # Εισαγωγή του ταξινομητή k-NN
from sklearn.preprocessing import StandardScaler # Εισαγωγή της συνάρτησης StandardScaler για κανονικοποίηση δεδομένων
import matplotlib.pyplot as plt  # Εισαγωγή της βιβλιοθήκης matplotlib για γραφήματα
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # Εισαγωγή συναρτήσεων για υπολογισμό ακρίβειας και εμφάνιση πίνακα σύγχυσης

# Παίρνουμε το dataset
phishing_websites = fetch_ucirepo(id=327) 

# Αποθηκεύουμε τα χαρακτηριστικά και τις ετικέτες του dataset σε pandas dataframe
# Με την dropna() αφαιρούμε τις ελλιπείς τιμές
X = phishing_websites.data.features.dropna()
y = phishing_websites.data.targets.dropna()

# Οπτικοποίηση της κατανομής των κλάσεων
plt.figure(figsize=(10, 6))
y.value_counts().plot(kind='bar', color='skyblue')
plt.title('Κατανομή κλάσεων')
plt.xlabel('Ομάδα')
plt.ylabel('Συχνότητα')
plt.show()

# Κανονικοποίηση των δεδομένων
Scaler = StandardScaler()
X_scaled = Scaler.fit_transform(X)

# Διαχωρισμός των δεδομένων σε σύνολο εκπαίδευσης και σύνολο ελέγχου
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Εκπαίδευση του ταξινομητή k-NN
kNN = KNeighborsClassifier(n_neighbors=3)

# Εκπαίδευση του ταξινομητή k-NN
kNN.fit(X_train, y_train)

# Πρόβλεψη των ετικετών του συνόλου ελέγχου
y_pred = kNN.predict(X_test)

# Υπολογισμός της ακρίβειας του ταξινομητή
Accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", Accuracy)

# Υπολογισμός της διασταυρούμενης επικύρωσης
Cross_val_score = cross_val_score(kNN, X_train, y_train, cv=10)
print("Cross-Validation Score: ", np.mean(Cross_val_score))

# Οπτικοποίηση του πίνακα σύγχυσης
Confusion_matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = Confusion_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.show()
