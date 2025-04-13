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
from vina import Vina

# Suppress the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# RDLogger.DisableLog('rdApp.*')

st.title("AutoDock Vina Docking App"

receptor = st.file_uploader("Upload Receptor (PDBQT)")
ligand = st.file_uploader("Upload Ligand (PDBQT)")

if receptor and ligand:
    # Save uploaded files
    with open("receptor.pdbqt", "wb") as f:
        f.write(receptor.getbuffer())
    with open("ligand.pdbqt", "wb") as f:
        f.write(ligand.getbuffer())

  if st.button("Run Docking"):
          cmd = [
              "vina",
              "--receptor", "receptor.pdbqt",
              "--ligand", "ligand.pdbqt",
              "--center_x", "0",  # Replace with your coordinates
              "--center_y", "0",
              "--center_z", "0",
              "--size_x", "20",
              "--size_y", "20",
              "--size_z", "20",
          ]
          result = subprocess.run(cmd, capture_output=True, text=True)
          
          st.code(result.stdout)  # Display results
