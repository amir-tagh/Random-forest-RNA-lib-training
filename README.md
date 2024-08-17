# Explanations
Takes both SDF and SMILES files.\
Calculates molecular descriptors using Mordred.\
Trains a Random Forest model on all calculated properties.\
Tests the model on a provided test set of SMILES.\
Plots the predicted vs. actual values for each target property.\
Descriptor Calculation: Mordred descriptors are computed for the molecules.\
Training: A Random Forest model is trained using the computed descriptors.\
Evaluation: The model is evaluated on a validation set.\
Prediction: The model predicts properties for the test set, and results are saved to test_predictions.csv.\

# Proposed work-flow (under development)
Parse Input SMILES and Calculate Descriptors:\
1. Use RDKit and Mordred to calculate molecular descriptors from SMILES data.\
2. Split Data:\
Split the dataset into training and testing sets.\

3. Model Training:\
Train multiple regression models such as Random Forest, Gradient Boosting, Elastic Net, and Lasso.\

4. Model Evaluation:\
Evaluate models based on their performance metrics like R-squared, Mean Squared Error (MSE), etc.
