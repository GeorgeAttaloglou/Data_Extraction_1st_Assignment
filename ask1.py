from ucimlrepo import fetch_ucirepo  # Βιβλιοθήκη για εύκολη φόρτωση datasets από το UCI Machine Learning Repository.
import pandas as pd  # Βιβλιοθήκη για διαχείριση δεδομένων σε μορφή DataFrame, ιδανική για ανάλυση και επεξεργασία.
import numpy as np  # Βιβλιοθήκη για αριθμητικούς υπολογισμούς και υποστήριξη εργασιών με πίνακες δεδομένων.
import matplotlib.pyplot as plt  # Βιβλιοθήκη για δημιουργία γραφημάτων και οπτικοποίηση δεδομένων.
from sklearn.model_selection import train_test_split  # Χρησιμοποιείται για διαχωρισμό δεδομένων σε σύνολα εκπαίδευσης και δοκιμής.
from sklearn.model_selection import cross_val_score  # Για εφαρμογή τεχνικών διασταυρούμενης επικύρωσης.
from sklearn.utils import resample  # Χρησιμοποιείται για επαναδειγματοληψία δεδομένων, αν απαιτείται (π.χ. oversampling).
from sklearn.tree import DecisionTreeClassifier  # Ταξινομητής που χρησιμοποιεί δέντρα απόφασης για κατηγοριοποίηση δεδομένων.
from sklearn.neighbors import KNeighborsClassifier  # Ταξινομητής που βασίζεται στον αλγόριθμο k-πλησιέστερων γειτόνων (k-NN).
from sklearn.preprocessing import StandardScaler  # Βοηθά στην κανονικοποίηση (scaling) των χαρακτηριστικών για καλύτερη απόδοση των αλγορίθμων.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay  # Βιβλιοθήκες για αξιολόγηση της απόδοσης των μοντέλων μηχανικής μάθησης.

pd.set_option('display.width', 500)  # Ρυθμίζει το πλάτος εμφάνισης των DataFrames για καλύτερη ορατότητα.
pd.set_option('display.max_columns', 100)  # Εμφανίζει έως και 100 στήλες ενός DataFrame κατά την εκτύπωση.
pd.set_option('display.notebook_repr_html', True)  # Επιτρέπει την HTML αναπαράσταση των DataFrames στο Jupyter Notebook.

def main():
    # Φόρτωση του dataset από το UCI Machine Learning Repository
    phishing_websites = fetch_ucirepo(id=327)  # Το ID 327 αντιστοιχεί στο dataset για phishing websites.

    # Εισαγωγή των δεδομένων σε pandas DataFrames
    x = phishing_websites.data.features  # Περιέχει τα χαρακτηριστικά (features) του dataset.
    y = phishing_websites.data.targets  # Περιέχει τις ετικέτες (targets) του dataset.

    # Παρουσίαση του πλήθους των δεδομένων για κάθε κατηγορία
    category_counts = y.value_counts()
    print("Number of samples for each category:")
    print(category_counts)

    # Δημιουργία γραφήματος για την κατανομή των δειγμάτων ανά κατηγορία
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Number of Samples for Each Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Samples')
    plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Phishing'], rotation=0)
    plt.show()

    # Έλεγχος για απουσιάζουσες τιμές και διαγραφή τους
    print("Checking for missing values and removing them")
    print(x.isnull().sum())  # Έλεγχος για κενά στις στήλες του X.
    print(y.isnull().sum())  # Έλεγχος για κενά στις ετικέτες (δεν αναμένονται).
    x = x.dropna()  # Αφαίρεση των γραμμών με απουσιάζουσες τιμές.

    # Μετατροπή κατηγορικών μεταβλητών σε δυαδικές (one-hot encoding)
    x = pd.get_dummies(x)

    while True:
        print()
        choice = input("Which algorithm should be used? Decision Trees, K-Nearest Neighbors or exit? (1/2/0):")

        if choice == '0':
            print("Exiting program...")
            break

        elif choice == '1':
            print("You chose Decision Trees")

            maxleafs = int(input("Enter the maximum number of leaf nodes: "))

            # Διαχωρισμός των δεδομένων σε training και testing datasets (80% training, 20% testing)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

            # Εκπαίδευση του ταξινομητή DecisionTreeClassifier
            dt = DecisionTreeClassifier(random_state=1, max_leaf_nodes=maxleafs)
            dt.fit(x_train, y_train)

            # Δημιουργία πρόβλεψης για το test set
            y_pred = dt.predict(x_test)

            # Αξιολόγηση του μοντέλου
            accuracy = accuracy_score(y_test, y_pred)
            print("\nModel Accuracy:", accuracy)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Δημιουργία ενός confusion matrix για οπτικοποίηση των αποτελεσμάτων
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix of Phishing Website Classifier")
            plt.show()

        elif choice == '2':
            print("You chose K-Nearest Neighbors")

            neighbors = int(input("Enter the number of neighbors: "))

            # Κανονικοποίηση των δεδομένων
            Scaler = StandardScaler()
            x_scaled = Scaler.fit_transform(x)

            # Διαχωρισμός των δεδομένων σε training και testing datasets (80% training, 20% testing)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

            # Εκπαίδευση του ταξινομητή KNN
            kNN = KNeighborsClassifier(n_neighbors=neighbors)
            kNN.fit(x_train, y_train)

            # Δημιουργία πρόβλεψης για το test set
            y_pred = kNN.predict(x_test)

            # Αξιολόγηση του μοντέλου
            accuracy = accuracy_score(y_test, y_pred)
            print ("\nModel Accuracy:", accuracy)
            print(classification_report(y_test, y_pred)) 
            Cross_val_score = cross_val_score(kNN, x_train, y_train, cv=10)
            print("Cross-Validation Score: ", np.mean(Cross_val_score))

            # Δημιουργία ενός confusion matrix για οπτικοποίηση των αποτελεσμάτων
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=kNN.classes_)
            disp.plot(cmap=plt.cm.Blues)
            plt.title("Confusion Matrix of Phishing Website Classifier")
            plt.show()

        else:
            print("Invalid choice. Please enter 1, 2 or 0.")

if __name__ == "__main__":
    main()
