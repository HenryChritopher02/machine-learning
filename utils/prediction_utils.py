import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors as mordred_descriptors_collection
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import requests
import os
# Removed: from huggingface_hub import hf_hub_download, HfFolder (if only used for PCA data)

# PyTorch and PyG for GNN
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from deepchem.feat import MolGraphConvFeaturizer

# Import from your GNN modules
from .gnn.gnn_architecture import GIN
from .gnn.gnn_train import load_model, predict_pic50_gnn

# Import paths from .paths (assuming paths.py is in utils/ as per your app structure)
from .paths import (
    BACE_TRAIN_DATA_URL, # For PCA training data
    WORKSPACE_PARENT_DIR # For saving downloaded GNN model
)


# --- Existing functions for Mordred, ECFP4, PCA ---
def calculate_mordred_descriptors(mols):
    if not mols:
        return pd.DataFrame()
    calc = Calculator(mordred_descriptors_collection, ignore_3D=True)
    df_mordred = calc.pandas(mols)
    for col in df_mordred.columns:
        if df_mordred[col].dtype == 'object':
            df_mordred[col] = pd.to_numeric(df_mordred[col], errors='coerce')
    df_mordred = df_mordred.fillna(np.nan)
    return df_mordred

def calculate_ecfp4_fingerprints(mols):
    if not mols:
        return pd.DataFrame()
    fp_list = []
    for mol in mols:
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_list.append(list(fp.ToBitString()))
        else:
            fp_list.append([np.nan] * 2048)
    df_ecfp4 = pd.DataFrame(fp_list, columns=[f"Bit_{i}" for i in range(2048)]).astype(float)
    return df_ecfp4

@st.cache_data
def load_and_prepare_train_data_desc(): # Removed hf_token parameter
    """Loads training data descriptors directly from BACE_TRAIN_DATA_URL."""
    st.info(f"Attempting to load PCA training data directly from: {BACE_TRAIN_DATA_URL}")
    try:
        # Directly use pd.read_csv with the URL
        df_train_raw = pd.read_csv(BACE_TRAIN_DATA_URL)
        
        st.success("Successfully loaded PCA training data.")
        
        # Assuming descriptors start from the 7th column (index 6)
        train_descriptors_df = df_train_raw.drop(['mol', 'CID', 'Class', 'Model', 'standardized_smiles', 'pIC50'], axis=1).copy()
        
        for col in train_descriptors_df.columns:
            if train_descriptors_df[col].dtype == 'object':
                 train_descriptors_df[col] = pd.to_numeric(train_descriptors_df[col], errors='coerce')
        train_descriptors_df = train_descriptors_df.fillna(np.nan)

        train_descriptor_names = train_descriptors_df.columns.tolist()
        return train_descriptors_df, train_descriptor_names

    except requests.exceptions.HTTPError as http_err: # More specific error for HTTP issues
        st.error(f"HTTP error occurred while fetching PCA training data: {http_err}")
        st.error(f"URL: {BACE_TRAIN_DATA_URL}. This might be due to the URL being private, gated, or temporarily unavailable.")
        return None, None
    except Exception as e:
        st.error(f"Error processing PCA training data from URL: {e}")
        st.error(f"URL: {BACE_TRAIN_DATA_URL}")
        return None, None

def align_input_descriptors(df_input_mordred, train_descriptor_names):
    if df_input_mordred.empty or not train_descriptor_names:
        return pd.DataFrame(columns=train_descriptor_names)
    df_aligned = pd.DataFrame()
    for name in train_descriptor_names:
        if name in df_input_mordred.columns:
            df_aligned[name] = df_input_mordred[name]
        else:
            df_aligned[name] = np.nan
    return df_aligned

def generate_pca_plot(df_train_desc, df_input_desc_aligned):
    if df_train_desc is None or df_train_desc.empty:
        st.warning("Training descriptors for PCA are not available.")
        return
    if df_input_desc_aligned.empty and not df_train_desc.empty :
        st.warning("Input descriptors for PCA are not available or could not be aligned.")
    elif df_input_desc_aligned.empty:
        st.warning("No data available for PCA plot.")
        return

    imputer = SimpleImputer(strategy='mean')
    train_desc_imputed = imputer.fit_transform(df_train_desc)
    
    scaler = StandardScaler()
    train_desc_scaled = scaler.fit_transform(train_desc_imputed)
    
    pca = PCA(n_components=2)
    pca.fit(train_desc_scaled)
    train_pca = pca.transform(train_desc_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(train_pca[:, 0], train_pca[:, 1], label='Train Data (BACE)', alpha=0.5, c='blue', edgecolors='k', s=50)

    if not df_input_desc_aligned.empty:
        input_desc_imputed = imputer.transform(df_input_desc_aligned)
        input_desc_scaled = scaler.transform(input_desc_imputed)
        input_pca = pca.transform(input_desc_scaled)
        if input_pca.shape[0] > 0:
            ax.scatter(input_pca[:, 0], input_pca[:, 1], label='Input SMILES', alpha=0.8, c='red', marker='X', s=100, edgecolors='k')
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    ax.set_title("PCA of Mordred Descriptors (Input vs BACE Train Set)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# --- New GNN Prediction Functions (remain unchanged from the previous version) ---

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

GNN_MODEL_URL = "https://raw.githubusercontent.com/HenryChritopher02/machine-learning/master/model/gnn/regression/gnn.pth"
GNN_MODEL_DIR = WORKSPACE_PARENT_DIR / "gnn_models"
GNN_MODEL_FILENAME = "gnn_bace_pic50_downloaded.pth"
DEFAULT_NODE_FEATURE_SIZE = 30

def mol_to_graph_data_for_prediction(smiles_str, featurizer):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        st.warning(f"Could not parse SMILES: {smiles_str}")
        return None
    
    try:
        features = featurizer._featurize(mol)
    except Exception as e:
        st.warning(f"Could not featurize SMILES '{smiles_str}': {e}")
        return None

    if features.node_features is None or features.node_features.size == 0 :
        st.warning(f"No node features generated for SMILES: {smiles_str}")
        return None

    node_features = torch.tensor(features.node_features, dtype=torch.float)
    edge_index = torch.tensor(features.edge_index, dtype=torch.long)
    edge_features = torch.tensor(features.edge_features, dtype=torch.float) if features.edge_features is not None else torch.empty(0, dtype=torch.float)

    data = Data(x=node_features,
                edge_index=edge_index,
                edge_attr=edge_features,
                smiles=smiles_str,
                y=torch.tensor([float('nan')], dtype=torch.float))
    return data

def create_graph_data_list_from_smiles(standardized_smiles_list):
    data_list = []
    featurizer = MolGraphConvFeaturizer(use_edges=True) 
    
    for smiles_str in standardized_smiles_list:
        graph_data = mol_to_graph_data_for_prediction(smiles_str, featurizer)
        if graph_data is not None:
            data_list.append(graph_data)
    return data_list

@st.cache_resource
def download_gnn_model_file(model_url, model_save_dir, model_filename):
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_local_path = model_save_dir / model_filename

    if not model_local_path.exists():
        st.info(f"Downloading GNN model from {model_url} to {model_local_path}...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            with open(model_local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("GNN Model file downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download GNN model: {e}")
            return None
    else:
        st.info(f"GNN Model file already exists at {model_local_path}.")
    return model_local_path

def run_gnn_prediction_workflow(standardized_smiles_list):
    if not standardized_smiles_list:
        st.warning("No standardized SMILES received for GNN prediction.")
        return pd.DataFrame()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"Using device for GNN operations: {device}")

    with st.spinner("Generating graph representations for input SMILES..."):
        graph_data_list = create_graph_data_list_from_smiles(standardized_smiles_list)
    
    if not graph_data_list:
        st.error("Graph data could not be created from the provided SMILES.")
        return pd.DataFrame()
    st.success(f"Generated graph data for {len(graph_data_list)} SMILES.")

    graph_loader = DataLoader(graph_data_list, batch_size=32, shuffle=False, num_workers=0)

    model_local_path = download_gnn_model_file(GNN_MODEL_URL, GNN_MODEL_DIR, GNN_MODEL_FILENAME)
    if not model_local_path:
        return pd.DataFrame()

    try:
        params = {'embedding_size': 64,
         'dense_neuron': 1024,
         'num_gin_layers': 2,
         'num_mlp_layers': 2,
         'dropout_value': 0.11225846425316804,
         'dropout_mlp': 0.3937283675371147,
         'learning_rate': 0.004222524354917754}

        model_architecture = GIN(
            feature_size=DEFAULT_NODE_FEATURE_SIZE,
            model_params = params).to(device)
        
        trained_gnn_model = load_model(model_architecture, str(model_local_path), device)
        if trained_gnn_model is None:
            st.error("Failed to load the GNN model after download.")
            return pd.DataFrame()
        st.success("GNN model loaded successfully.")
    except Exception as e:
        st.error(f"Error initializing or loading GNN model: {e}")
        return pd.DataFrame()

    with st.spinner("Predicting pIC50 values using the GNN model..."):
        try:
            df_predictions = predict_pic50_gnn(trained_gnn_model, graph_loader, device)
        except Exception as e:
            st.error(f"An error occurred during GNN pIC50 prediction: {e}")
            return pd.DataFrame()
            
    if df_predictions.empty:
        st.warning("GNN prediction resulted in an empty DataFrame.")
    else:
        st.success("GNN pIC50 predictions are ready.")
        
    return df_predictions
