import sys
import pandas as pd
import numpy as np
import joblib
import time
import concurrent.futures
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from mordred import Calculator, descriptors
from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_rdkit_descriptors(smiles, calc, descriptor_names):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            descriptors = calc.CalcDescriptors(mol)
            return descriptors
        else:
            print(f"Invalid SMILES: {smiles}")
            return [None] * len(descriptor_names)
    except Exception as e:
        print(f"Error computing RDKit descriptors for SMILES {smiles}: {e}")
        return [None] * len(descriptor_names)

def compute_mordred_descriptors(mol, calc):
    try:
        if mol is not None:
            desc = calc(mol)
            return desc
        else:
            print(f"Invalid molecule")
            return [None] * len(calc.descriptors)
    except Exception as e:
        print(f"Error computing Mordred descriptors: {e}")
        return [None] * len(calc.descriptors)

def calculate_rdkit_descriptors(smiles_list, output_file):
    print("Calculating molecular descriptors using RDKit...")
    descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
    calc = MolecularDescriptorCalculator(descriptor_names)

    start_time = time.time()
    desc_values = []

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8)) as executor:
            print("Starting parallel processing...")
            partial_compute = partial(compute_rdkit_descriptors, calc=calc, descriptor_names=descriptor_names)
            future_to_smiles = {executor.submit(partial_compute, smiles): smiles for smiles in smiles_list}
            for future in concurrent.futures.as_completed(future_to_smiles):
                try:
                    desc_values.append(future.result())
                except Exception as e:
                    print(f"Error retrieving result: {e}")
            print("Parallel processing completed.")
    except Exception as e:
        print(f"Error during parallel processing: {e}")

    end_time = time.time()
    print(f"Descriptor calculation took {end_time - start_time:.2f} seconds.")

    valid_desc_values = [value for value in desc_values if value is not None]
    if valid_desc_values:
        desc_df = pd.DataFrame(valid_desc_values, columns=descriptor_names)
        desc_df.to_csv(output_file, index=False)
        print(f"Descriptors saved to {output_file}.")
    else:
        print("No valid descriptors calculated.")
        desc_df = pd.DataFrame(columns=descriptor_names)

    return desc_df

def calculate_mordred_descriptors(smiles_list, output_file):
    print("Calculating molecular descriptors using Mordred...")
    calc = Calculator(descriptors, ignore_3D=True)
    start_time = time.time()
    
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    desc_values = []
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 8)) as executor:
            print("Starting parallel processing...")
            future_to_mol = {executor.submit(compute_mordred_descriptors, mol, calc): mol for mol in mols}
            for future in concurrent.futures.as_completed(future_to_mol):
                try:
                    desc = future.result()
                    if desc:
                        desc_values.append(desc)
                except Exception as e:
                    print(f"Error retrieving result: {e}")
            print("Parallel processing completed.")
    except Exception as e:
        print(f"Error during parallel processing: {e}")
    
    end_time = time.time()
    print(f"Descriptor calculation took {end_time - start_time:.2f} seconds.")
    
    desc_df = pd.DataFrame(desc_values, columns=[str(d) for d in calc.descriptors])
    desc_df.to_csv(output_file, index=False)
    print(f"Descriptors saved to {output_file}.")
    
    return desc_df

def validate_smiles(smiles_list):
    """Validate SMILES strings and filter out invalid ones."""
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
        else:
            print(f"Invalid SMILES: {smiles}")
    return valid_smiles

def process_smiles(input_data_file):
    print("Loading SMILES data from file...")
    df = pd.read_csv(input_data_file, sep='\t', header=None)
    smiles_list = df.iloc[:, 0].dropna().tolist()
    valid_smiles_list = validate_smiles(smiles_list)
    print(f"Loaded {len(valid_smiles_list)} valid SMILES strings.")
    return valid_smiles_list

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

def handle_infinite_values(desc_df):
    print("Handling infinite values...")
    desc_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    desc_df.dropna(inplace=True)  # Remove rows with NaN values after replacing inf
    return desc_df

def handle_missing_values(desc_df, median_desc):
    print("Handling missing values...")

    # Filter numeric columns and drop columns that are all zero or NaN
    numeric_desc_df = desc_df.select_dtypes(include=[np.number])
    numeric_desc_df = numeric_desc_df.loc[:, (numeric_desc_df != 0).any(axis=0)]  # Remove all-zero columns
    numeric_desc_df = numeric_desc_df.dropna(axis=1, how='all')  # Drop columns with all NaN values

    # Handle infinite values
    numeric_desc_df = handle_infinite_values(numeric_desc_df)

    # Remove rows with NaN values after replacing inf
    numeric_desc_df = numeric_desc_df.dropna()

    # Align median_desc with numeric_desc_df by keeping only corresponding rows
    median_desc = median_desc.loc[numeric_desc_df.index]

    print(f"Numeric descriptors after cleaning: {numeric_desc_df.shape[1]}")
    
    return numeric_desc_df, median_desc

def compute_median_descriptors(desc_df):
    print("Computing median descriptors...")
    non_zero_desc_df = desc_df.loc[:, (desc_df != 0).any(axis=0)]
    print(f"Descriptors after removing all-zero columns: {non_zero_desc_df.shape[1]}")
    median_desc = non_zero_desc_df.median(axis=1)  # Median per molecule
    return median_desc

def train_random_forest(X, y):
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

    X_train_array = X_train.to_numpy()
    X_test_array = X_test.to_numpy()
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()

    if np.isfinite(X_train_array).all() and np.isfinite(y_train_array).all():
        model.fit(X_train_array, y_train_array)
    else:
        print("Training data contains infinite values; check data cleaning steps.")
        return None, None, None, None, None

    print("Evaluating model...")
    y_train_pred = model.predict(X_train_array)
    y_test_pred = model.predict(X_test_array)

    mse_train = mean_squared_error(y_train_array, y_train_pred)
    mse_test = mean_squared_error(y_test_array, y_test_pred)
    r2_train = r2_score(y_train_array, y_train_pred)
    r2_test = r2_score(y_test_array, y_test_pred)

    print(f"Training MSE: {mse_train:.4f}, R2: {r2_train:.4f}")
    print(f"Test MSE: {mse_test:.4f}, R2: {r2_test:.4f}")

    return model, X_train, X_test, y_train, y_test

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', linewidth=2)
    plt.title('Predicted vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')

    plt.subplot(1, 2, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(0, min(y_pred), max(y_pred), colors='r', linestyles='dashed')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')

    plt.tight_layout()
    plt.savefig('random_forest_predictions.png')
    plt.show()

def plot_feature_importances(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('random_forest_feature_importances.png')
    plt.show()

def main(input_data_file, use_rdkit=True):
    # Process SMILES input
    smiles_list = process_smiles(input_data_file)

    # Calculate descriptors (choose either RDKit or Mordred)
    if use_rdkit:
        output_file = 'rdkit_descriptors.csv'
        desc_df = calculate_rdkit_descriptors(smiles_list, output_file)
    else:
        output_file = 'mordred_descriptors.csv'
        desc_df = calculate_mordred_descriptors(smiles_list, output_file)

    # Describe the data
    describe_data(desc_df)

    # Compute median descriptors and handle missing values
    median_desc = compute_median_descriptors(desc_df)
    desc_df, median_desc = handle_missing_values(desc_df, median_desc)

    # Train Random Forest model
    model, X_train, X_test, y_train, y_test = train_random_forest(desc_df, median_desc)

    # Plot predictions and feature importances if model training was successful
    if model is not None:
        y_test_pred = model.predict(X_test)
        plot_predictions(y_test, y_test_pred)
        plot_feature_importances(model, desc_df.columns)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_data_file> <use_rdkit>")
        print("<use_rdkit> should be 'True' or 'False'")
        sys.exit(1)
    
    input_data_file = sys.argv[1]
    use_rdkit = sys.argv[2].lower() == 'true'
    main(input_data_file, use_rdkit)

