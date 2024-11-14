import pandas as pd
from ucimlrepo import fetch_ucirepo 

def main():
    # Fetch the dataset from the UCI Machine Learning Repository
    data = fetch_ucirepo(id=327)  # Example ID, replace with the actual dataset ID you need
    
    # Print the keys of the dataset
    print(data.keys())
    
    # Print the features of the phishing websites
    features = data['features']
    print("Features of the phishing websites:")
    for feature in features:
        print(feature)

if __name__ == "__main__":
    main()