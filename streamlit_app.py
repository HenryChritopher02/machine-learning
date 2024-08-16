import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import joblib
import requests
import os
from io import BytesIO
from sklearn.exceptions import InconsistentVersionWarning
import warnings

# Suppress the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# RDLogger.DisableLog('rdApp.*')

st.title('🎈 Machine learning App')

st.write('Hello world!')

# File using for prediction
model_urls = ['https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model1.pkl',
              'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model2.pkl',
              'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model3.pkl',
              'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model4.pkl',
              'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/model5.pkl']
train_files = [
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M1_Train_6components.csv',
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M2_Train_5components.csv',
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M3_Train_3components.csv',
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M4_Train_5components.csv',
    'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/QSAR-HB_M5_Train_5components.csv'
]

# Magic function
def predict_with_models(model_urls, data, train_files):

    # Initialize an empty DataFrame to store predictions
    result_df = pd.DataFrame({'SrNo': data['SrNo']})

    # Loop through each model URL and corresponding training file
    for i, (model_url, train_file_path) in enumerate(zip(model_urls, train_files)):
        # Download the model file from the URL
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an error if the download fails

        # Load the model from the downloaded content
        model = joblib.load(BytesIO(response.content))

        # Read the training data
        train_data = pd.read_csv(train_file_path)
        
        # Extract feature columns (excluding 'SrNo' and 'pIC50_HB')
        cols_to_use = train_data.drop(columns=['SrNo', 'pIC50_HB']).columns.tolist()

        # Extract the corresponding features from the input data
        X = data[cols_to_use]

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_data[cols_to_use])
        X_scaled = scaler.transform(X)

        # Perform predictions
        y_pred = model.predict(X_scaled).ravel()

        # Add the predictions to the result DataFrame
        result_df[f'pIC50_Model_{i+1}'] = y_pred

    # Calculate the average prediction across all models
    result_df['Average_pIC50'] = result_df.iloc[:, 1:].mean(axis=1)

    return result_df
used_columns = ['VE3sign_D/Dt', 'P_VSA_s_4', 'SM14_AEA(dm)', 'CATS2D_09_AA', 'F07[O-O]', 'ATSC7dv', 'ATSC7i',	'VE1_B(p)', 'SsssNH+', 'P_VSA_LogP_2', 'TSRW10', 'P_VSA_ppp_P',	'VR3_Dzs',	'SssCH2	B06[N-N]',	'AATS8i']
with st.expander('Input'):
    option = st.radio("Choose an option", ("Upload CSV file", "Upload XLSX file"), index=None)
    if option == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            result = predict_with_models(model_urls, data, train_files)
            st.write('Descriptors for prediction')
            st.write(used_columns)
        else:
            st.error('Please upload a CSV file')

    elif option == "Upload XLSX file":
        uploaded_file = st.file_uploader("Choose a XLSX file", type='xlsx')
        if uploaded_file is not None:
            data = pd.read_excel(uploaded_file)
            result = predict_with_models(model_urls, data, train_files)
            st.write('Descriptors for prediction')
            st.write(used_columns)
        else:
            st.error('Please upload a XLSX file')
    else:
        st.error('Please choose your option')

with st.expander('Prediction results'):
    if 'result' in locals():
        st.write(result)
    else:
        st.error('No input data provided')

# def standardize(smiles, invalid_smiles_list):
#             try:
#                 mol = Chem.MolFromSmiles(smiles)  # Sanitize = True
#                 if mol is None:
#                     raise ValueError(f"Invalid SMILES string: {smiles}")
#                 Chem.SanitizeMol(mol)
#                 Chem.Kekulize(mol)
#                 mol = Chem.RemoveHs(mol)
#                 mol = rdMolStandardize.Uncharger().uncharge(mol)
#                 mol = rdMolStandardize.Reionize(mol)
#                 mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
#                 mol = rdMolStandardize.FragmentParent(mol)
#                 Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
#                 standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, kekuleSmiles=True)
#                 return standardized_smiles
#             except Exception as e:
#                 print(f"Error standardizing SMILES {smiles}: {e}")
#                 invalid_smiles_list.append(smiles)
#                 return None
                        
# def standardize_smiles(smiles_series):
#             invalid_smiles_list = []
#             standardized_smiles = []

#             for smiles in smiles_series:
#                 standardized_smile = standardize(smiles, invalid_smiles_list)
#                 if standardized_smile:
#                     standardized_smiles.append(standardized_smile)

#             return pd.Series(standardized_smiles), invalid_smiles_list
# def rdkit_descriptors(smiles):
#     # Check if smiles is a string or a list/series
#     if isinstance(smiles, str):
#         smiles = [smiles]  # Convert string to a list
#     elif isinstance(smiles, pd.Series):
#         smiles = smiles.tolist()  # Convert pandas Series to a list
    
#     mols = [Chem.MolFromSmiles(i) for i in smiles]
#     calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0]
#                                                               for x in Descriptors._descList])
#     desc_names = calc.GetDescriptorNames()

#     mol_descriptors = []
#     for mol in mols:
#         # Add hydrogens to molecules
#         mol = Chem.AddHs(mol)
#         # Calculate all 200 descriptors for each molecule
#         descriptors = calc.CalcDescriptors(mol)
#         mol_descriptors.append(descriptors)
#     return mol_descriptors, desc_names

# # with st.expander('Data'):
# #   st.write('**Standardized data**')
# data = pd.read_csv('https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/bace1_standardized.csv')
# data = data.drop(data.columns[0], axis=1)
# # data

# # st.write('**Calculated descriptors data**')
# mol_descriptors, desc_names = rdkit_descriptors(data['standardized_smiles'])
# data_des = pd.DataFrame(mol_descriptors,columns=desc_names)
# data_des = data_des.apply(pd.to_numeric, errors='coerce')
# data_des.dropna(axis=1, inplace=True)
# columns = data_des.columns
# data_des = data_des.astype('float64')
# total = pd.concat([data['pIC50'], data_des], axis=1)
# # total

# # st.write('**X**')
# X = total.drop('pIC50', axis=1)
# # X
  
# # st.write('**y**')
# y = total['pIC50']
# # y
  
# with st.expander('Input'):
#     option = st.radio("Choose an option", ("Upload CSV file", "Input SMILES string"), index=None)
#     if option == "Upload CSV file":
#         uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
#         if uploaded_file is not None:
#             data_uploaded = pd.read_csv(uploaded_file)
#             if 'SMILES' in data_uploaded.columns:
#                 st.write('Input Data:')
#                 st.write(data_uploaded)

#                 standardized_smiles, invalid_smiles = standardize_smiles(data_uploaded['SMILES'])
#                 data_standardized = pd.DataFrame(standardized_smiles, columns=['Standardized SMILES'])
#                 st.write('Standardized SMILES:')
#                 st.write(data_standardized)

#                 if invalid_smiles:
#                     st.write('Invalid SMILES string(s):')
#                     st.error(invalid_smiles)

#                 mol_descriptors, desc_names = rdkit_descriptors(standardized_smiles)
#                 data_new = pd.DataFrame(mol_descriptors, columns=desc_names)
#                 data_new = data_new[columns]
#                 data_new = data_new.apply(pd.to_numeric, errors='coerce')
#                 st.write('Calculated Descriptors:')
#                 st.write(data_new)
#                 X_new = data_new.values
#                 st.write('Descriptors used for prediction:')
#                 st.write(desc_names)
#             else:
#                 st.write('The CSV file does not contain a "SMILES" column.')
#         else:
#             st.write('Please upload a CSV file with a "SMILES" column.')

#     elif option == "Input SMILES string":
#         smiles_input = st.text_input("Enter a SMILES string")
#         if smiles_input:
#             data_uploaded = pd.DataFrame({'SMILES': [smiles_input]})
#             standardized_smiles, invalid_smiles = standardize_smiles(data_uploaded['SMILES'])
#             data_standardized = pd.DataFrame(standardized_smiles, columns=['Standardized SMILES'])
#             st.write('Standardized SMILES:')
#             st.write(data_standardized)
    
#             if invalid_smiles:
#                 st.write('Invalid SMILES string:')
#                 st.error(invalid_smiles)
#             mol_descriptors, desc_names = rdkit_descriptors(standardized_smiles)
#             data_new = pd.DataFrame(mol_descriptors, columns=desc_names, index=[0])
#             data_new = data_new[columns]
#             data_new = data_new.apply(pd.to_numeric, errors='coerce')
#             st.write('Calculated Descriptors:')
#             st.write(data_new)
#             X_new = data_new.values
#             st.write('Descriptors used for prediction:')
#             st.write(desc_names)
#         else:
#             st.write('Please enter a SMILES string.')
    
#     else:
#         st.write('Please choose your input option.')

# with st.expander('Properties domain of molecules'):
#     if 'X_new' in locals() and 'X' in locals():
#     # Initialize t-SNE with 2 components for 2D visualization
#             # Initialize PCA with 2 components for 2D visualization
#             pca = PCA(n_components=2)
        
#             # Fit PCA on the training data (X) and transform it
#             X_pca = pca.fit_transform(X.values)
        
#             # Transform the new data (X_new) using the fitted PCA model
#             X_new_pca = pca.transform(X_new)
            
#             # Plot t-SNE visualization
#             plt.figure(figsize=(12, 6))
#             plt.scatter(X_pca[:, 0], X_pca[:, 1], label='Our molecules')
#             plt.scatter(X_new_pca[:, 0], X_new_pca[:, 1], label="User's molecule(s)")
#             # Set labels for the axes
#             plt.xlabel('PCA 1')
#             plt.ylabel('PCA 2')
            
#             # Set a title for the plot
#             plt.title('2D PCA Plot')
            
#             # Add a legend to distinguish between the two datasets
#             plt.legend()
            
#             # Save the plot to a BytesIO buffer
#             buf = BytesIO()
#             plt.savefig(buf, format='png')
#             buf.seek(0)
#             plt.close()
        
#             # Display the plot in Streamlit
#             st.image(buf, caption='t-SNE Visualization', use_column_width=True)
#     else:
#         st.error('No input data provided')     

# with st.expander('Prediction'):
#     # Download the model file from GitHub
#     model_url = 'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/rf_model.pkl'
#     model_file = 'rf_model.pkl'

#     # Check if the model file exists locally
#     if os.path.exists(model_file):
#         loaded_model = joblib.load(model_file)
#     else:
#         # Download the model file if it doesn't exist
#         response = requests.get(model_url)

#         # Save the model file locally
#         with open(model_file, 'wb') as f:
#             f.write(response.content)

#         # Load the model
#         loaded_model = joblib.load(model_file)

#     if 'X_new' in locals() and 'standardized_smiles' in locals():
#         y_pred = loaded_model.predict(X_new)

#         try:
#             # Create a dataframe with standardized SMILES and predicted pIC50
#             prediction_df = pd.DataFrame({
#                 'Standardized SMILES': standardized_smiles,
#                 'Predicted pIC50': y_pred
#             })

#             st.write('Predictions:')
#             st.write(prediction_df)
#         except ValueError as e:
#             st.error('An error in input data')
#     else:
#         st.error('No input data provided')
