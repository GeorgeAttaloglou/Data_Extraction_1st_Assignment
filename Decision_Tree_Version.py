from ucimlrepo import fetch_ucirepo #Κατεβάζουμε το fetch_ucirepo από το ucimlrepo
import numpy as np # Εισάγει τη βιβλιοθήκη NumPy, η οποία παρέχει υποστήριξη για πίνακες (arrays) και αριθμητικούς υπολογισμούς.
import seaborn as sns
import scipy as sp # Εισάγει τη βιβλιοθήκη SciPy, η οποία χρησιμοποιείται για επιστημονικούς υπολογισμούς.
from sklearn.model_selection import train_test_split # Εισάγει τη συνάρτηση train_test_split από τη βιβλιοθήκη scikit-learn, η οποία χρησιμοποιείται για να χωρίσει τα δεδομένα σε training και testing datasets.
from sklearn import tree # Εισάγει το υποσύνολο tree από τη βιβλιοθήκη scikit-learn, το οποίο περιλαμβάνει υλοποιήσεις αλγορίθμων δέντρου απόφασης.
from sklearn.model_selection import cross_val_score # Εισάγει τη συνάρτηση cross_val_score, η οποία υπολογίζει τα σκορ διασταυρούμενης επικύρωσης για μοντέλα μηχανικής μάθησης.
from sklearn.utils import resample # Εισάγει τη συνάρτηση resample, η οποία χρησιμοποιείται για την επαναδειγματοληψία (resampling) δεδομένων.
from sklearn.tree import DecisionTreeClassifier, plot_tree # Εισάγει τον ταξινομητή δέντρου απόφασης DecisionTreeClassifier και τη συνάρτηση plot_tree από τη βιβλιοθήκη scikit-learn.
from sklearn.ensemble import RandomForestClassifier # Εισάγει τον ταξινομητή τυχαίου δάσους RandomForestClassifier, ο οποίος είναι ένας αλγόριθμος που χρησιμοποιεί πολλά δέντρα απόφασης για να βελτιώσει την ακρίβεια των προβλέψεων.
import matplotlib as mpl # Εισάγει τη βιβλιοθήκη matplotlib, που χρησιμοποιείται για τη δημιουργία γραφημάτων και οπτικοποιήσεων.
import matplotlib.cm as cm # Εισάγει το υποσύνολο cm της βιβλιοθήκης matplotlib, που παρέχει έτοιμες παλέτες χρωμάτων (colormaps) για τη δημιουργία γραφημάτων.
import matplotlib.pyplot as plt # Εισάγει το υποσύνολο pyplot από τη βιβλιοθήκη matplotlib, το οποίο παρέχει μια συλλογή συναρτήσεων για τη δημιουργία γραφημάτων.
import pandas as pd # Εισάγει τη βιβλιοθήκη pandas, η οποία χρησιμοποιείται για την επεξεργασία και ανάλυση δεδομένων, κυρίως μέσω των DataFrames.
pd.set_option('display.width', 500) # Ρυθμίζει το πλάτος εμφάνισης των DataFrames στα 500 pixels, επιτρέποντας την καλύτερη ορατότητα των δεδομένων.
pd.set_option('display.max_columns', 100) # Ορίζει τον μέγιστο αριθμό στηλών που θα εμφανίζονται σε ένα DataFrame κατά την εκτύπωση, σε 100 στήλες.
pd.set_option('display.notebook_repr_html', True) # Ενεργοποιεί την εμφάνιση των DataFrames σε μορφή HTML στο Jupyter Notebook, κάνοντάς τα πιο ευανάγνωστα.
from sklearn.metrics import accuracy_score, confusion_matrix

# Παίρνουμε το dataset
phishing_websites = fetch_ucirepo(id=327) 

data = phishing_websites.data

data = phishing_websites.data.dropna() # Αφαιρούμε τις γραμμές με απουσιάζουσες τιμές

# Αποθηκεύουμε τα χαρακτηριστικά και τις ετικέτες του dataset σε μεταβλητές
X = phishing_websites.data.features
y = phishing_websites.data.targets

# Διαχωρίζουμε τα δεδομένα σε training και testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Δημιουργούμε έναν ταξινομητή δέντρου απόφασης
DecisionTree = DecisionTreeClassifier(random_state=1)

#plt.figure(figsize=(13, 8)) # Ορίζουμε μέγεθος γραφήματος
#tree_vis = plot_tree(DecisionTree, filled=True) # Δημιουργούμε γράφημα δέντρου με τα χρώματα να υποδεικνύουν την κατανομή των κλάσεων