from rdkit import Chem
from rdkit.Chem import PandasTools
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def calculate_mordred_descriptors(mols):
    calc = Calculator(descriptors, ignore_3D=True)
    return calc.pandas(mols)

def load_molecules(input_file):
    if input_file.endswith('.sdf'):
        mols = PandasTools.LoadSDF(input_file)
    elif input_file.endswith('.smiles'):
        mols = PandasTools.LoadSMILES(input_file, smilesColumn='SMILES', nameColumn='ID', includeFingerprints=False)
    else:
        raise ValueError("Unsupported file format. Use .sdf or .smiles.")
    return mols

def train_and_evaluate_models(X, y):
    results = {}
    for target_column in y.columns:
        y_target = y[target_column]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y_target, test_size=0.2, random_state=42)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_valid)
        rf_mse = mean_squared_error(y_valid, rf_pred)
        rf_r2 = r2_score(y_valid, rf_pred)

        # Gradient Boosting Machine
        gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbm.fit(X_train, y_train)
        gbm_pred = gbm.predict(X_valid)
        gbm_mse = mean_squared_error(y_valid, gbm_pred)
        gbm_r2 = r2_score(y_valid, gbm_pred)

        results[target_column] = {
            "rf_model": rf,
            "rf_mse": rf_mse,
            "rf_r2": rf_r2,
            "rf_pred": rf_pred,
            "gbm_model": gbm,
            "gbm_mse": gbm_mse,
            "gbm_r2": gbm_r2,
            "gbm_pred": gbm_pred,
            "y_valid": y_valid,
        }
        
        print(f"{target_column} - Random Forest MSE: {rf_mse:.4f}, R2: {rf_r2:.4f}")
        print(f"{target_column} - Gradient Boosting MSE: {gbm_mse:.4f}, R2: {gbm_r2:.4f}")
    return results

def plot_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for target, res in results.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(res["y_valid"], res["rf_pred"], alpha=0.7, label="Random Forest")
        plt.scatter(res["y_valid"], res["gbm_pred"], alpha=0.7, label="Gradient Boosting", color='orange')
        plt.plot([min(res["y_valid"]), max(res["y_valid"])], [min(res["y_valid"]), max(res["y_valid"])], color='red', linestyle='--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{target} - Actual vs Predicted")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{target}_prediction_plot.png"))
        plt.close()

def main(train_file, test_file, output_dir):
    print("Loading training molecules...")
    train_mols = load_molecules(train_file)
    print("Calculating Mordred descriptors for training set...")
    X_train = calculate_mordred_descriptors(train_mols['ROMol'])
    
    y_train = X_train.copy()  # Assuming using all descriptors as target properties for this example
    
    print("Training models...")
    results = train_and_evaluate_models(X_train, y_train)
    
    print("Loading test molecules...")
    test_mols = load_molecules(test_file)
    print("Calculating Mordred descriptors for test set...")
    X_test = calculate_mordred_descriptors(test_mols['ROMol'])
    
    print("Testing models...")
    # Assuming the test set has the same descriptor columns as the training set.
    plot_results(results, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test models on molecular descriptors using Random Forest and Gradient Boosting.")
    parser.add_argument("train_file", help="Input file (SDF/SMILES) for training.")
    parser.add_argument("test_file", help="Input SMILES file for testing.")
    parser.add_argument("output_dir", help="Directory to save output plots.")
    
    args = parser.parse_args()
    main(args.train_file, args.test_file, args.output_dir)

