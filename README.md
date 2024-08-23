# Explanations
Takes both SDF and SMILES files.

Calculates molecular descriptors using Mordred.

Trains a Random Forest model on all calculated properties.

Tests the model on a provided test set of SMILES.

Plots the predicted vs. actual values for each target property.

Descriptor Calculation: Mordred descriptors are computed for the molecules.

Training: A Random Forest model is trained using the computed descriptors.
Evaluation: The model is evaluated on a validation set.

Prediction: The model predicts properties for the test set, and results are saved to test_predictions.csv.

# Proposed work-flow (under development)
Parse Input SMILES and Calculate Descriptors:
1. Use RDKit and Mordred to calculate molecular descriptors from SMILES data.
2. Split Data:
Split the dataset into training and testing sets.

3. Model Training:
Train multiple regression models such as Random Forest, Gradient Boosting, Elastic Net, and Lasso.

4. Model Evaluation:
Evaluate models based on their performance metrics like R-squared, Mean Squared Error (MSE), etc.

# How the Script Works:
Input File: The input file should be space-separated, with the first column containing SMILES strings and one column with target values.

Model Choice: Use the --model argument to choose between 'random_forest', 'gradient_boosting', 'elastic_net', and 'lasso'.

Descriptor Calculation: Descriptors are calculated using Mordred, and missing values in the descriptor data are handled by dropping columns with NaN.

Model Training: The script splits the data into training and testing sets and trains the specified regression model.

Evaluation: It evaluates the model using R-squared and Mean Squared Error and prints the results.

# Install the required packages:
## Create a Virtual Environment:

python -m venv myenv
source myenv/bin/activate
## Install RDKit:
conda install -c conda-forge rdkit
## Install Mordred:
pip install mordred
## Install Scikit-Learn:
pip install scikit-learn
## Install Pandas:
pip install pandas
## Install Numpy:
pip install numpy

# Software used
During the development several freely available packages were used. Here we acknowledge and thanks:\
numpy (https://numpy.org/)

pandas (https://pandas.pydata.org/)

rdkit (https://www.rdkit.org/)

mordred (https://github.com/mordred-descriptor/mordred)

sklearn (https://scikit-learn.org/stable/)

tqdm (https://github.com/tqdm/tqdm)

seaborn (https://seaborn.pydata.org/)

# Preliminary results on RNA binders
![model_performance](https://github.com/user-attachments/assets/58018f87-8e27-4687-8ef7-0909339c6348)


![feature_importances](https://github.com/user-attachments/assets/8733e384-825c-4c6f-ad81-3b4f7039fac4)

# Calculation of molecular descriptors
Now the model can be trained on both RDkit and Mordred extracted descriptors
Script is parallelized to pick up the available CPU cores and handle the training for big libraries







