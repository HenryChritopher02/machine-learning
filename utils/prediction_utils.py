import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors as mordred_descriptors_collection # Renamed to avoid conflict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import requests

# Import paths from paths.py
from paths import BACE_TRAIN_DATA_URL

def calculate_mordred_descriptors(mols):
    """Calculates Mordred descriptors for a list of RDKit Mol objects."""
    if not mols:
        return pd.DataFrame()
    calc = Calculator(mordred_descriptors_collection, ignore_3D=True)
    df_mordred = calc.pandas(mols)

    # Process error objects from Mordred to NaN
    for col in df_mordred.columns:
        # Check if column is object type and contains non-numeric, non-NaN values
        if df_mordred[col].dtype == 'object':
            df_mordred[col] = pd.to_numeric(df_mordred[col], errors='coerce')
        # Ensure all resulting NaNs are actual np.nan for SimpleImputer
    df_mordred = df_mordred.fillna(np.nan)
    return df_mordred

def calculate_ecfp4_fingerprints(mols):
    """Calculates ECFP4 fingerprints (2048 bits) for a list of RDKit Mol objects."""
    if not mols:
        return pd.DataFrame()
    fp_list = []
    for mol in mols:
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_list.append(list(fp.ToBitString())) # Store as list of '0'/'1' strings
        else:
            fp_list.append([np.nan] * 2048) # Or handle None mol differently
    
    df_ecfp4 = pd.DataFrame(fp_list, columns=[f"Bit_{i}" for i in range(2048)]).astype(float)
    return df_ecfp4


@st.cache_data # Cache the download and processing of training data
def load_and_prepare_train_data_desc():
    """Loads training data descriptors from URL."""
    try:
        df_train_raw = pd.read_csv(BACE_TRAIN_DATA_URL)
        # Assuming descriptors start from the 7th column (index 6)
        train_descriptors_df = df_train_raw.iloc[:, 6:].copy()
        
        # Convert any non-numeric to NaN, then fill NaN (e.g. with mean)
        for col in train_descriptors_df.columns:
            if train_descriptors_df[col].dtype == 'object':
                 train_descriptors_df[col] = pd.to_numeric(train_descriptors_df[col], errors='coerce')
        train_descriptors_df = train_descriptors_df.fillna(np.nan) # Ensure all NaNs are consistent

        train_descriptor_names = train_descriptors_df.columns.tolist()
        return train_descriptors_df, train_descriptor_names
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading training data: {e}")
        return None, None
    except pd.errors.EmptyDataError:
        st.error(f"Training data CSV file from {BACE_TRAIN_DATA_URL} is empty.")
        return None, None
    except Exception as e:
        st.error(f"Error processing training data: {e}")
        return None, None

def align_input_descriptors(df_input_mordred, train_descriptor_names):
    """Aligns input descriptors with training descriptor names."""
    if df_input_mordred.empty or not train_descriptor_names:
        return pd.DataFrame(columns=train_descriptor_names)

    # Select only columns that are in train_descriptor_names
    # and ensure they are in the same order
    df_aligned = pd.DataFrame()
    for name in train_descriptor_names:
        if name in df_input_mordred.columns:
            df_aligned[name] = df_input_mordred[name]
        else:
            df_aligned[name] = np.nan # Add missing columns with NaN
    return df_aligned


def generate_pca_plot(df_train_desc, df_input_desc_aligned):
    """Generates and displays PCA scatter plot."""
    if df_train_desc is None or df_train_desc.empty:
        st.warning("Training descriptors are not available for PCA.")
        return
    if df_input_desc_aligned.empty:
        st.warning("Input descriptors are not available for PCA.")
        return

    # 1. Impute NaNs
    imputer = SimpleImputer(strategy='mean')
    
    # Fit imputer on training data and transform both
    train_desc_imputed = imputer.fit_transform(df_train_desc)
    # If input_desc might have all NaNs in a column not in train or different NaN patterns:
    # It's generally safer to impute them based on the training data's statistics.
    input_desc_imputed = imputer.transform(df_input_desc_aligned)

    # Convert back to DataFrame for consistent column names if needed, though not strictly for PCA
    df_train_desc_imputed = pd.DataFrame(train_desc_imputed, columns=df_train_desc.columns)
    df_input_desc_imputed = pd.DataFrame(input_desc_imputed, columns=df_input_desc_aligned.columns)

    # 2. Scale data
    scaler = StandardScaler()
    # Fit scaler on (imputed) training data and transform both
    train_desc_scaled = scaler.fit_transform(df_train_desc_imputed)
    input_desc_scaled = scaler.transform(df_input_desc_imputed)

    # 3. Apply PCA
    pca = PCA(n_components=2)
    # Fit PCA on (scaled, imputed) training data
    pca.fit(train_desc_scaled)
    
    train_pca = pca.transform(train_desc_scaled)
    input_pca = pca.transform(input_desc_scaled)

    # 4. Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(train_pca[:, 0], train_pca[:, 1], label='Train Data (BACE)', alpha=0.5, c='blue', edgecolors='k', s=50)
    if input_pca.shape[0] > 0: # Check if there's any input data to plot
        ax.scatter(input_pca[:, 0], input_pca[:, 1], label='Input SMILES', alpha=0.8, c='red', marker='X', s=100, edgecolors='k')
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    ax.set_title("PCA of Mordred Descriptors (Input vs BACE Train Set)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)