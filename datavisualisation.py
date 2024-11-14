import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the CSV file
    df = pd.read_csv('Training Dataset.csv')
    
    # Print the structure of the dataset
    print("Dataset structure:")
    print(df.info())
    
    # Print the features of the phishing websites
    features = df.columns
    print("Features of the phishing websites:")
    for feature in features:
        print(feature)
    
    # Visualize the features
    num_features = len(features)
    num_cols = 3  # Number of columns in the grid
    num_rows = (num_features + num_cols - 1) // num_cols  # Calculate the number of rows needed
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
    
    for i, feature in enumerate(features):
        axes[i].hist(df[feature].dropna(), bins=30, alpha=0.75)
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
