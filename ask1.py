import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 

def main():
    # Fetch dataset 
    phishing_websites = fetch_ucirepo(id=327)  # UCI repo dataset ID for phishing websites
    
    # Data (as pandas dataframes) 
    X = phishing_websites.data.features
    y = phishing_websites.data.targets

    # Step 1: Check and Remove Missing Values
    print("Checking for missing values...")
    print(X.isnull().sum())  # Displays the missing values per column

    # Remove rows with missing values
    X = X.dropna()
    y = y.loc[X.index]  # Keep corresponding target values
    
    # Step 2: Encode Categorical Variables
    # If there are categorical variables, convert them to numeric
    X = pd.get_dummies(X)  # Converts all categorical variables to numeric (one-hot encoding)

    # Step 3: Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Initialize and train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Step 5: Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Step 7: Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix of Phishing Website Classifier")
    plt.show()

    
    print("testing branches")

    
 
# Call main function
if __name__ == "__main__":
    main()
