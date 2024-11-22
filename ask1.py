from ucimlrepo import fetch_ucirepo  # Για φόρτωση datasets από το UCI Machine Learning Repository
import pandas as pd  # Για επεξεργασία και ανάλυση δεδομένων
import numpy as np  # Για αριθμητικούς υπολογισμούς και πίνακες
import matplotlib.pyplot as plt  # Για γραφική απεικόνιση δεδομένων
from sklearn.model_selection import train_test_split  # Για διαχωρισμό δεδομένων σε σύνολα εκπαίδευσης και δοκιμής
from sklearn.model_selection import cross_val_score  # Για διασταυρούμενη επικύρωση
from sklearn.utils import resample  # Για επαναδειγματοληψία δεδομένων
from sklearn.tree import DecisionTreeClassifier  # Για ταξινόμηση με δέντρα απόφασης
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay  # Για αξιολόγηση μοντέλων μηχανικής μάθησης

pd.set_option('display.width', 500) # Ρυθμίζει το πλάτος εμφάνισης των DataFrames στα 500 pixels, επιτρέποντας την καλύτερη ορατότητα των δεδομένων.
pd.set_option('display.max_columns', 100) # Ορίζει τον μέγιστο αριθμό στηλών που θα εμφανίζονται σε ένα DataFrame κατά την εκτύπωση, σε 100 στήλες.
pd.set_option('display.notebook_repr_html', True)  # Ρύθμιση εμφάνισης δεδομένων σε HTML μορφή

def main ():
    # Φόρτωση του dataset
    phishing_websites = fetch_ucirepo(id=327)  # UCI repo dataset ID for phishing websites

    #Εισαγωγη των δεδομενων σε panda dataframe
    x = phishing_websites.data.features
    y = phishing_websites.data.targets

    # Παρουσίαση του πλήθους των δεδομένων για κάθε κατηγορία
    category_counts = y.value_counts()
    print("Number of samples for each category:")
    print(category_counts)

    # Δημιουργία γραφήματος για το πλήθος των δεδομένων για κάθε κατηγορία
    plt.figure(figsize=(8, 6))
    category_counts.plot(kind='bar', color=['green', 'red'])
    plt.title('Number of Samples for Each Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Samples')
    plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Phishing'], rotation=0)
    plt.show()

    #Ελεγχος για απουσιαζουσες τιμες και διαγραφη τους
    print ("Checking for missing values and removing them")
    print(x.isnull().sum()) #Ελέγχει για απουσιάζουσες τιμές στο DataFrame X
    print(y.isnull().sum()) #Ελέγχει για απουσιάζουσες τιμές στο DataFrame y (αν και δεν θα υπαρχουν)
    x = x.dropna() #Αφαιρεί τις γραμμές με απουσιάζουσες τιμές από το DataFrame X

    #Μετατροπή των κατηγορικών μεταβλητών σε δυαδικές
    x = pd.get_dummies(x)

    while True:
            print ()
            choice = input("Which algorithm should be used? Decision Trees, K-Nearest Neighbors or exit? (1/2/0):")

            if choice == '0':
                print("Exiting program...")
                break

            elif choice == '1':
                print("You chose Decision Trees")

                #Διαχωρισμος των δεδομενων σε training και testing datasets (80% training, 20% testing)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

                #Εκπαιδευση του ταξινομητη DecisionTreeClassifier
                dt = DecisionTreeClassifier(random_state=1, max_leaf_nodes=10)
                dt.fit(x_train, y_train)

                #Δημιουργία πρόβλεψης για το test set
                y_pred = dt.predict(x_test)

                #Αξιολόγιση του μοντέλου
                accuracy = accuracy_score(y_test, y_pred)
                print("\nModel Accuracy:", accuracy)
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred))

                #Δημιουργία ενός confusion matrix για οπτικοποίηση των αποτελεσμάτων
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
                disp.plot(cmap=plt.cm.Blues)
                plt.title("Confusion Matrix of Phishing Website Classifier")
                plt.show()

            elif choice == '2':
                print("You chose K-Nearest Neighbors")
        
                # Κανονικοποίηση των δεδομένων
                Scaler = StandardScaler()
                x_scaled = Scaler.fit_transform(x)

                # Μετατροπή του y σε μονοδιάστατο array (Δημιουργουσε προβλημα το οτι το y ηταν πολυδιαστατος πινακας και οχι μονοδιαστατος)
                y = y.values.ravel()

                #Διαχωρισμος των δεδομενων σε training και testing datasets (80% training, 20% testing)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

                #Εκπαιδευση του ταξινομιτη KNN
                kNN = KNeighborsClassifier(n_neighbors=3)
                kNN.fit(x_train, y_train)

                #Δημιουργία πρόβλεψης για το test set
                y_pred = kNN.predict(x_test)

                #Αξιολόγιση του μοντέλου
                accuracy = accuracy_score(y_test, y_pred)
                print("\nModel Accuracy:", accuracy)
                Cross_val_score = cross_val_score(kNN, x_train, y_train, cv=10)
                print("Cross-Validation Score: ", np.mean(Cross_val_score))

                #Δημιουργία ενός confusion matrix για οπτικοποίηση των αποτελεσμάτων
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=kNN.classes_)
                disp.plot(cmap=plt.cm.Blues)
                plt.title("Confusion Matrix of Phishing Website Classifier")
                plt.show()

            else:
                print("Invalid choice. Please enter 1, 2 or 0.")

if __name__ == "__main__":
    main()