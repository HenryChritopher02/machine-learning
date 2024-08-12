import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors, AllChem
from sklearn.preprocessing import StandardScaler
from astartes import train_test_split

# RDLogger.DisableLog('rdApp.*')

st.title('🎈 Machine learning App')

st.write('Hello world!')

with st.expander('Data'):
  st.write('**Standardized data**')
  data = pd.read_csv('https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/bace1_standardized.csv')
  data = data.drop(data.columns[0], axis=1)
  data

  st.write('**Calculated descriptors data**')

  def rdkit_descriptors(smiles):
      mols = [Chem.MolFromSmiles(i) for i in smiles]
      calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0]
                                       for x in Descriptors._descList])
      desc_names = calc.GetDescriptorNames()
    
      mol_descriptors =[]
      for mol in mols:
        #Add hydrogens to molecules
        mol=Chem.AddHs(mol)
        #Calculate all 200 descriptors for each molecule
        descriptors = calc.CalcDescriptors(mol)
        mol_descriptors.append(descriptors)
      return mol_descriptors, desc_names

  mol_descriptors, desc_names = rdkit_descriptors(data['standardized_smiles'])
  data_des = pd.DataFrame(mol_descriptors,columns=desc_names)
  data_des = data_des.apply(pd.to_numeric, errors='coerce')
  data_des.dropna(axis=1, inplace=True)
  scaler = StandardScaler()
  data_des_scaled = scaler.fit_transform(data_des)
  data_des = pd.DataFrame(data_des_scaled, columns=data_des.columns)
  data_des = data_des.astype('float64')
  total = pd.concat([data['pIC50'], data_des], axis=1)
  total

  st.write('**X**')
  X = total.drop('pIC50', axis=1).values
  X
  
  st.write('**y**')
  y = total['pIC50'].values
  y
  
  X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size = 0.8,
    random_state = 42,
    sampler = 'kennard_stone', #random, kennard_stone
  )
  
