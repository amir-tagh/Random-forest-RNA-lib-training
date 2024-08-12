import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import argparse
import os

def calculate_descriptors(molecules):
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas(molecules)
    return df

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gbm(X_train, y_train):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_elastic_net(X_train, y_train):
    model = ElasticNet(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lasso(X_train, y_train):
    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, output_dir, model_name):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{model_name} Mean Squared Error: {mse}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.savefig(os.path.join(output_dir, f'{model_name}_actual_vs_predicted.png'))
    plt.close()

def main(input_file, test_file, output_dir, selected_models):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and calculate descriptors
    if input_file.endswith('.sdf'):
        mol_supplier = Chem.SDMolSupplier(input_file)
        molecules = [mol for mol in mol_supplier if mol is not None]
    elif input_file.endswith('.smiles'):
        smiles_df = pd.read_csv(input_file, header=None)
        molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_df[0]]

    train_descriptors = calculate_descriptors(molecules)
    X_train = train_descriptors.dropna(axis=1)

    # Assume the target properties are all columns
    y_train = X_train.copy()

    # Load test data
    test_smiles_df = pd.read_csv(test_file, header=None)
    test_molecules = [Chem.MolFromSmiles(smiles) for smiles in test_smiles_df[0]]
    test_descriptors = calculate_descriptors(test_molecules)
    X_test = test_descriptors.dropna(axis=1)
    y_test = X_test.copy()

    models = {}

    if 'rf' in selected_models:
        print("Training Random Forest...")
        rf_model = train_random_forest(X_train, y_train)
        models['rf'] = rf_model

    if 'gbm' in selected_models:
        print("Training Gradient Boosting Machines...")
        gbm_model = train_gbm(X_train, y_train)
        models['gbm'] = gbm_model

    if 'enet' in selected_models:
        print("Training Elastic Net...")
        enet_model = train_elastic_net(X_train, y_train)
        models['enet'] = enet_model

    if 'lasso' in selected_models:
        print("Training Lasso...")
        lasso_model = train_lasso(X_train, y_train)
        models['lasso'] = lasso_model

    for model_name, model in models.items():
        evaluate_model(model, X_test, y_test, output_dir, model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on molecular descriptors and evaluate.")
    parser.add_argument("input_file", help="Input file containing molecules (.sdf or .smiles).")
    parser.add_argument("test_file", help="Test file containing SMILES.")
    parser.add_argument("output_dir", help="Directory to save output plots.")
    parser.add_argument("--models", nargs='+', default=['rf', 'gbm', 'enet', 'lasso'], 
                        choices=['rf', 'gbm', 'enet', 'lasso'], help="Models to train: rf, gbm, enet, lasso.")

    args = parser.parse_args()
    main(args.input_file, args.test_file, args.output_dir, args.models)

