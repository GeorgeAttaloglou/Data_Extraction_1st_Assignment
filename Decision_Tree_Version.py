from ucimlrepo import fetch_ucirepo
import numpy as np # Εισάγει τη βιβλιοθήκη NumPy, η οποία παρέχει υποστήριξη για πίνακες (arrays) και αριθμητικούς υπολογισμούς.
import seaborn as sns
import scipy as sp # Εισάγει τη βιβλιοθήκη SciPy, η οποία χρησιμοποιείται για επιστημονικούς υπολογισμούς.
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree # Εισάγει το υποσύνολο tree από τη βιβλιοθήκη scikit-learn, το οποίο περιλαμβάνει υλοποιήσεις αλγορίθμων δέντρου απόφασης.
from sklearn.utils import resample # Εισάγει τη συνάρτηση resample, η οποία χρησιμοποιείται για την επαναδειγματοληψία (resampling) δεδομένων.
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier # Εισάγει τον ταξινομητή τυχαίου δάσους RandomForestClassifier, ο οποίος είναι ένας αλγόριθμος που χρησιμοποιεί πολλά δέντρα απόφασης για να βελτιώσει την ακρίβεια των προβλέψεων.
import matplotlib as mpl # Εισάγει τη βιβλιοθήκη matplotlib, που χρησιμοποιείται για τη δημιουργία γραφημάτων και οπτικοποιήσεων.
import matplotlib.cm as cm # Εισάγει το υποσύνολο cm της βιβλιοθήκης matplotlib, που παρέχει έτοιμες παλέτες χρωμάτων (colormaps) για τη δημιουργία γραφημάτων.
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Παίρνουμε το dataset
phishing_websites = fetch_ucirepo(id=327) 

# Αποθηκεύουμε τα χαρακτηριστικά και τις ετικέτες του dataset σε μεταβλητές
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

#TODO: Play around with max_leaf_nodes and keep track of the results
# Δημιουργούμε έναν ταξινομητή δέντρου απόφασης
DecisionTree = DecisionTreeClassifier(random_state=1)

# Υπολογίζουμε τα σκορ διασταυρούμενης επικύρωσης για τον ταξινομητή DecisionTree
Cross_Validation_Score = cross_val_score(DecisionTree, X_train, y_train, cv=10) 
print("Cross Validation Score: ", np.mean(Cross_Validation_Score))

# Εκπαιδεύουμε τον ταξινομητή DecisionTree στο training dataset
DecisionTree.fit(X_train, y_train)

# Κάνουμε προβλέψεις στο testing dataset
y_pred = DecisionTree.predict(X_test)

# Υπολογίζουμε την ακρίβεια του ταξινομητή DecisionTree
Accuracy = accuracy_score(y_test, y_pred)
Confusion_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy: ", Accuracy)
#print("Confusion Matrix:\n", Confusion_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix = Confusion_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.show()


#plt.figure(figsize=(13, 8)) # Ορίζουμε μέγεθος γραφήματος
#tree_vis = plot_tree(DecisionTree, filled=True) # Δημιουργούμε γράφημα δέντρου με τα χρώματα να υποδεικνύουν την κατανομή των κλάσεων