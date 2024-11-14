import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

    print("Checking for missing values again...")
    print(X.isnull().sum())  # Displays the missing values per column 


    

# Call main function
if __name__ == "__main__":
    main()

