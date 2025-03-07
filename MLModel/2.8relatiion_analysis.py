import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    return data

def calculate_correlation(data):
    # Create new columns for pairwise multiplication of specific columns by their names
    columns = ['Oxygen Diss','Temp Water','pH','ln Turd','ln Coliform','ln Cond']  # Update with actual column names
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            data[f'{columns[i]}_x_{columns[j]}'] = data[columns[i]] * data[columns[j]]
    
    # Calculate correlation for the specified columns and the new columns
    correlation_matrix = data[columns + [f'{columns[i]}_x_{columns[j]}' for i in range(len(columns)) for j in range(i+1, len(columns))]].corr()
    return correlation_matrix

def plot_correlation_matrix(correlation_matrix):
    # Mask for correlations with absolute value less than 0.3 or equal to 1
    mask = (np.abs(correlation_matrix) < 0.3) | (correlation_matrix == 1)
    
    # Plot the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix for Specified Columns')
    plt.show()

if __name__ == "__main__":
    file_path = '2.7sin_cos_transformed.csv'  # Update with your actual file path
    data = load_data(file_path)
    correlation_matrix = calculate_correlation(data)
    plot_correlation_matrix(correlation_matrix)