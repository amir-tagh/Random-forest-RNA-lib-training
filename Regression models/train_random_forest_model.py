import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(input_file, input_format):
    if input_format == "sdf":
        sdf_supplier = Chem.SDMolSupplier(input_file)
        mols = [mol for mol in sdf_supplier if mol is not None]
    elif input_format == "smiles":
        df = pd.read_csv(input_file, delim_whitespace=True, header=None, names=['smiles'])
        mols = [Chem.MolFromSmiles(smiles) for smiles in df['smiles'] if smiles]
    else:
        raise ValueError("Unsupported input format. Use 'sdf' or 'smiles'.")
    return mols

def calculate_descriptors(mols):
    calc = Calculator(descriptors, ignore_3D=True)
    descriptors_df = calc.pandas(mols)
    return descriptors_df

def train_and_evaluate_models(X, y):
    results = {}
    for target_column in y.columns:
        y_target = y[target_column]
        X_train, X_valid, y_train, y_valid = train_test_split(X, y_target, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)
        
        results[target_column] = {
            "model": rf,
            "mse": mse,
            "r2": r2,
            "y_valid": y_valid,
            "y_pred": y_pred
        }
        print(f"{target_column} - MSE: {mse:.4f}, R2: {r2:.4f}")
    return results

def predict_test_set(models, X_test):
    predictions = {}
    for target_column, model_info in models.items():
        model = model_info['model']
        y_pred = model.predict(X_test)
        predictions[target_column] = y_pred
    return pd.DataFrame(predictions)

def plot_results(y_valid, y_pred, output_dir, descriptor_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_valid, y_pred, alpha=0.3)
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], '--r', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs Predicted for {descriptor_name}")
    plt.savefig(os.path.join(output_dir, f"{descriptor_name}_actual_vs_predicted.png"))
    plt.close()

def main(train_file, train_format, test_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training data
    train_mols = load_data(train_file, train_format)
    
    # Calculate descriptors for training data
    print("Calculating descriptors for training data...")
    descriptors_df = calculate_descriptors(train_mols)
    
    # Prepare features (X) and targets (y)
    X = descriptors_df
    y = descriptors_df  # Using all calculated descriptors as targets

    # Train and evaluate models for each target descriptor
    print("Training and evaluating models...")
    models = train_and_evaluate_models(X, y)
    
    # Plot actual vs predicted for validation set
    for descriptor_name, model_info in models.items():
        y_valid = model_info['y_valid']
        y_pred = model_info['y_pred']
        plot_results(y_valid, y_pred, output_dir, descriptor_name)
    
    # Load test data
    test_mols = load_data(test_file, "smiles")
    
    # Calculate descriptors for test data
    print("Calculating descriptors for test data...")
    X_test = calculate_descriptors(test_mols)
    
    # Predict on test data
    print("Predicting on test data...")
    test_predictions = predict_test_set(models, X_test)
    
    # Save predictions to a CSV file
    test_predictions.to_csv(os.path.join(output_dir, "test_predictions_multi_target.csv"), index=False)
    print(f"Test predictions saved to '{output_dir}/test_predictions_multi_target.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest models using MORDRED descriptors as both features and targets, and plot the results.")
    parser.add_argument("train_file", help="Input training file (SDF or SMILES format).")
    parser.add_argument("train_format", choices=["sdf", "smiles"], help="Format of the training file: 'sdf' or 'smiles'.")
    parser.add_argument("test_file", help="Input test file (SMILES format).")
    parser.add_argument("output_dir", help="Directory to save output plots and predictions.")
    
    args = parser.parse_args()
    main(args.train_file, args.train_format, args.test_file, args.output_dir)

