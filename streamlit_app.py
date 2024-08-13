import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import os

# RDLogger.DisableLog('rdApp.*')

st.title('🎈 Machine learning App')

st.write('Hello world!')

with st.expander('Data'):
  st.write('**Standardized data**')
  data = pd.read_csv('https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/bace1_standardized.csv')
  data = data.drop(data.columns[0], axis=1)
  data

  st.write('**Calculated descriptors data**')
  def standardize(smiles, invalid_smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)  # Sanitize = True
                if mol is None:
                    raise ValueError(f"Invalid SMILES string: {smiles}")
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol)
                mol = Chem.RemoveHs(mol)
                mol = rdMolStandardize.Uncharger().uncharge(mol)
                mol = rdMolStandardize.Reionize(mol)
                mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)
                mol = rdMolStandardize.FragmentParent(mol)
                Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
                standardized_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, kekuleSmiles=True)
                return standardized_smiles
            except Exception as e:
                print(f"Error standardizing SMILES {smiles}: {e}")
                invalid_smiles_list.append(smiles)
                return None
  
  def standardize_smiles(smiles_series):
            invalid_smiles_list = []
            standardized_smiles = []

            for smiles in smiles_series:
                standardized_smile = standardize(smiles, invalid_smiles_list)
                if standardized_smile:
                    standardized_smiles.append(standardized_smile)

            return pd.Series(standardized_smiles), invalid_smiles_list
  
  def rdkit_descriptors(smiles):
    # Check if smiles is a string or a list/series
    if isinstance(smiles, str):
        smiles = [smiles]  # Convert string to a list
    elif isinstance(smiles, pd.Series):
        smiles = smiles.tolist()  # Convert pandas Series to a list
    
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0]
                                                              for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    mol_descriptors = []
    for mol in mols:
        # Add hydrogens to molecules
        mol = Chem.AddHs(mol)
        # Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)
    return mol_descriptors, desc_names

  mol_descriptors, desc_names = rdkit_descriptors(data['standardized_smiles'])
  data_des = pd.DataFrame(mol_descriptors,columns=desc_names)
  data_des = data_des.apply(pd.to_numeric, errors='coerce')
  data_des.dropna(axis=1, inplace=True)
  columns = data_des.columns
  data_des = data_des.astype('float64')
  total = pd.concat([data['pIC50'], data_des], axis=1)
  total

  st.write('**X**')
  X = total.drop('pIC50', axis=1)
  X
  
  st.write('**y**')
  y = total['pIC50']
  y
  
with st.expander('Input'):
    option = st.radio("Choose an option", ("Upload CSV file", "Input SMILES string"), index=None,)

    if option == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            @st.cache_data
            def process_csv(uploaded_file):
                data = pd.read_csv(uploaded_file)
                if 'SMILES' in data.columns:
                    st.write('Input Data:')
                    st.write(data)

                    standardized_smiles, invalid_smiles = standardize_smiles(data['SMILES'])
                    st.write('Standardized SMILES:')
                    st.write(standardized_smiles)

                    if invalid_smiles:
                        st.write('Invalid SMILES string(s):')
                        st.error(invalid_smiles)

                    mol_descriptors, desc_names = rdkit_descriptors(standardized_smiles)
                    data_new = pd.DataFrame(mol_descriptors, columns=desc_names)
                    data_new = data_new[columns]
                    data_new = data_new.apply(pd.to_numeric, errors='coerce')
                    st.write('Calculated Descriptors:')
                    st.write(data_new)
                    return data_new.values
                else:
                    st.write('The CSV file does not contain a "SMILES" column.')
                    return None

            X_new = process_csv(uploaded_file)

        else:
            st.write('Please upload a CSV file with a "SMILES" column.')

    elif option == "Input SMILES string":
        smiles_input = st.text_input("Enter a SMILES string")
        data = pd.DataFrame({'SMILES': [smiles_input]})
        standardized_smiles, invalid_smiles = standardize_smiles(data['SMILES'])
        st.write('Standardized SMILES:')
        st.write(standardized_smiles)

        if invalid_smiles:
            st.write('Invalid SMILES string:')
            st.error(invalid_smiles)
        mol_descriptors, desc_names = rdkit_descriptors(standardized_smiles)
        data_new = pd.DataFrame(mol_descriptors, columns=desc_names, index=[0])
        data_new = data_new[columns]
        data_new = data_new.apply(pd.to_numeric, errors='coerce')
        st.write('Calculated Descriptors:')
        st.write(data_new)
        X_new = data_new.values
    else:
        st.write('Please enter a SMILES string.')
    st.stop()  # Stop here and wait for user input

with st.expander('Prediction'):
    # Download the model file from GitHub
    model_url = 'https://raw.githubusercontent.com/HenryChritopher02/bace1/main/saved_model/rf_model.pkl'
    model_file = 'rf_model.pkl'

    # Check if the model file exists locally
    if os.path.exists(model_file):
        loaded_model = joblib.load(model_file)
    else:
        # Download the model file if it doesn't exist
        response = requests.get(model_url)

        # Save the model file locally
        with open(model_file, 'wb') as f:
            f.write(response.content)

        # Load the model
        loaded_model = joblib.load(model_file)

    if 'X_new' in locals():
        y_pred = loaded_model.predict(X_new)

        try:
            # Create a dataframe with standardized SMILES and predicted pIC50
            prediction_df = pd.DataFrame({
                'Standardized SMILES': standardized_smiles,
                'Predicted pIC50': y_pred
            })

            st.write('Predictions:')
            st.write(prediction_df)
        except ValueError as e:
            st.error('An error in input data')
    else:
        st.error('No input data provided')
