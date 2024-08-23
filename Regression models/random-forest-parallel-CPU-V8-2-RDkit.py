import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

def calculate_descriptors(smiles_list):
    print("Calculating molecular descriptors using RDKit...")
    
    # Define the list of descriptors to be calculated
    descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
    calc = MolecularDescriptorCalculator(descriptor_names)
    
    # Generate RDKit molecules
    mols = [Chem.MolFromSmiles(smiles) for smiles in tqdm(smiles_list, desc="Processing SMILES", unit="molecule") if Chem.MolFromSmiles(smiles) is not None]
    
    # Calculate descriptors and store them in a DataFrame
    desc_values = []
    for mol in mols:
        desc_values.append(calc.CalcDescriptors(mol))
    
    desc_df = pd.DataFrame(desc_values, columns=descriptor_names)
    return desc_df

def process_smiles(input_data_file):
    print("Loading SMILES data from file...")
    df = pd.read_csv(input_data_file, sep='\t', header=None)
    smiles_list = df.iloc[:, 0].dropna().tolist()
    return smiles_list

def describe_data(df):
    """ Print a summary of the DataFrame. """
    print("Data Summary:")
    print(df.describe(include='all'))
    print(f"Number of zero columns: {(df == 0).all().sum()}")
    print(f"Number of NaN values: {df.isna().sum().sum()}")
    if np.issubdtype(df.values.dtype, np.number):
        print(f"Number of infinite values: {np.isinf(df.values).sum()}")
    else:
        print("Data contains non-numeric values; unable to check for infinite values.")

def handle_missing_values(desc_df, target_values):
    print("Handling missing values...")
    
    # Convert all columns in desc_df to numeric types, forcing errors to NaN
    numeric_desc_df = desc_df.apply(pd.to_numeric, errors='coerce')
    
    # Print summary before cleaning
    describe_data(numeric_desc_df)
    
    # Remove columns that are all zero
    zero_columns = (numeric_desc_df == 0).all()
    numeric_desc_df = numeric_desc_df.loc[:, ~zero_columns]
    
    # Impute missing values with the median value of each column
    imputer = SimpleImputer(strategy='median')
    numeric_desc_df_imputed = pd.DataFrame(imputer.fit_transform(numeric_desc_df), columns=numeric_desc_df.columns, index=numeric_desc_df.index)
    
    # Ensure data is numeric and handle infinite values
    if np.issubdtype(numeric_desc_df_imputed.values.dtype, np.number):
        valid_rows_array = numeric_desc_df_imputed.to_numpy(dtype=np.float64)
    else:
        raise ValueError("Data contains non-numeric values after conversion.")
    
    valid_target_values_array = target_values[numeric_desc_df.index.isin(numeric_desc_df_imputed.index)].to_numpy(dtype=np.float64)
    
    # Check for infinite values
    if np.isfinite(valid_rows_array).all():
        print("No infinite values detected in the cleaned descriptors.")
    else:
        print("Warning: Infinite values detected in the cleaned descriptors.")
        valid_rows_array = valid_rows_array[np.isfinite(valid_rows_array).all(axis=1)]
        valid_target_values_array = valid_target_values_array[np.isfinite(valid_rows_array).all(axis=1)]
    
    # Convert back to DataFrame
    valid_rows = pd.DataFrame(valid_rows_array, index=numeric_desc_df_imputed.index, columns=numeric_desc_df_imputed.columns)
    valid_target_values = pd.Series(valid_target_values_array, index=numeric_desc_df_imputed.index)
    
    # Final check for NaN and infinite values in cleaned data
    if valid_rows.isnull().values.any():
        print("Warning: Cleaned descriptors still contain NaN values.")
    
    if np.isnan(valid_rows.values).any():
        print("Warning: Cleaned descriptors still contain NaN values.")
    
    if np.isinf(valid_rows.values).any():
        print("Warning: Cleaned descriptors still contain infinite values.")
    
    if np.isnan(valid_target_values).any():
        print("Warning: Cleaned target values still contain NaN values.")
    
    if np.isinf(valid_target_values).any():
        print("Warning: Cleaned target values still contain infinite values.")
    
    # Check if there's data left to train the model
    if valid_rows.shape[0] == 0:
        raise ValueError("No valid data available for training after cleaning.")

    return valid_rows, valid_target_values

def compute_median_descriptors(desc_df):
    print("Computing median descriptors...")
    # Remove descriptors with all zero values
    non_zero_desc_df = desc_df.loc[:, (desc_df != 0).any(axis=0)]
    print(f"Descriptors after removing all-zero columns: {non_zero_desc_df.shape[1]}")
    median_desc = non_zero_desc_df.median(axis=1)  # Median per molecule
    return median_desc


def train_random_forest(X, y):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

    # Convert DataFrames to NumPy arrays for NaN and infinite value checks
    X_train_array = X_train.to_numpy()
    X_test_array = X_test.to_numpy()
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    # Check for NaN or infinite values before fitting
    if np.isnan(X_train_array).any() or np.isinf(X_train_array).any():
        print("Error: Training features contain NaN or infinite values.")

    if np.isnan(y_train_array).any() or np.isinf(y_train_array).any():
        print("Error: Training target values contain NaN or infinite values.")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # Plot the results
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, color='blue', edgecolor='k')
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--', linewidth=2)
    plt.title('Training Set')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, color='green', edgecolor='k')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', linewidth=2)
    plt.title('Test Set')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.show()

    # Return model and metrics
    return model, mse_train, mse_test, r2_train, r2_test



def plot_feature_importances(model, feature_names, top_n=10):
    """
    Plots the most important descriptors based on the feature importances from the Random Forest model.

    Parameters:
    - model: Trained Random Forest model.
    - feature_names: List of feature names corresponding to the descriptors.
    - top_n: Number of top features to display.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Get the top N features
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_features, top_importances, color='blue', edgecolor='black')
    
    # Add labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center', ha='left', color='black')
    
    plt.xlabel('Feature Importance')
    plt.ylabel('Descriptors')
    plt.title(f'Top {top_n} Most Important Descriptors')
    plt.gca().invert_yaxis()  # To display the highest importance at the top
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.show()


def print_important_descriptors(model, feature_names, top_n=10):
    """
    Prints the most important descriptors based on the feature importances from the Random Forest model.

    Parameters:
    - model: Trained Random Forest model.
    - feature_names: List of feature names corresponding to the descriptors.
    - top_n: Number of top features to display.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"Top {top_n} most important descriptors:")
    for i in range(top_n):
        print(f"{i + 1}: {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")


def main(input_data_file):
    # Load the SMILES data
    smiles_list = process_smiles(input_data_file)

    # Calculate the molecular descriptors
    desc_df = calculate_descriptors(smiles_list)

    # Optionally, compute the median descriptors for further analysis
    median_desc = compute_median_descriptors(desc_df)

    # Handle missing values in the descriptors
    desc_df_cleaned, median_desc_cleaned = handle_missing_values(desc_df, median_desc)

    # Train a Random Forest model using the cleaned data
    model, mse_train, mse_test, r2_train, r2_test = train_random_forest(desc_df_cleaned, median_desc_cleaned)

    # Print out model performance metrics
    print(f"Training MSE: {mse_train}")
    print(f"Test MSE: {mse_test}")
    print(f"Training R²: {r2_train}")
    print(f"Test R²: {r2_test}")

    # Plot the top 10 feature importances
    plot_feature_importances(model, desc_df_cleaned.columns, top_n=10)

    # Optionally, print the top 10 most important descriptors
    print_important_descriptors(model, desc_df_cleaned.columns, top_n=10)

    # Save the trained model for future use
    joblib.dump(model, 'trained_random_forest_model.pkl')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <input_data_file>")
        sys.exit(1)

    input_data_file = sys.argv[1]
    main(input_data_file)

