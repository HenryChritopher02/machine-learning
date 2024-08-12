pip install mordred

import streamlit as st
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem, RDLogger
from mordred import Calculator, descriptors
from sklearn.preprocessing import StandardScaler

# RDLogger.DisableLog('rdApp.*')

st.title('🎈 Machine learning App')

st.write('Hello world!')

with st.expander('Data'):
  st.write('**Standardized data**')
  data = pd.read_csv('https://raw.githubusercontent.com/HenryChritopher02/bace1/main/data/bace1_standardized.csv')
  data = data.drop(data.columns[0], axis=1)
  data

  # st.write('**Calculated descriptors data**')

  # def All_Mordred_descriptors(smiles_data):
  #     calc = Calculator(descriptors, ignore_3D=True)
  #     mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in tqdm(smiles_data, desc='Adding Hydrogens')]
      
  #     # pandas df
  #     df = calc.pandas(mols)
  #     return df
  
  # data_des = All_Mordred_descriptors(data['standardized_smiles'])
  # data_des = data_des.apply(pd.to_numeric, errors='coerce')
  # data_des.dropna(axis=1, inplace=True)
  # scaler = StandardScaler()
  # data_des_scaled = scaler.fit_transform(data_des)
  # data_des = pd.DataFrame(data_des_scaled, columns=data_des.columns)
  # data_des = data_des.astype('float64')
  # total = pd.concat([data['pIC50'], data_des], axis=1)
  # total
