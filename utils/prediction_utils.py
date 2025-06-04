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

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset # For numerical features
from deepchem.feat import MolGraphConvFeaturizer

from .gnn.gnn_architecture import GIN
from .gnn.gnn_train import load_model, predict_pic50_gnn
from .gnn.gnn_mlp_architecture import GIN_hybrid, MLP1, CombinedMLP
from .gnn.gnn_mlp_train import predict_pic50_hybrid

from .paths import (
    BACE_TRAIN_DATA_URL,
    WORKSPACE_PARENT_DIR,
    HYBRID_MODEL_BASE_URL # Get this from paths.py
)

# --- Existing functions for Mordred, ECFP4, PCA (largely unchanged) ---
def calculate_mordred_descriptors(mols):
    if not mols: return pd.DataFrame()
    calc = Calculator(mordred_descriptors_collection, ignore_3D=True) # Use alias
    df_mordred = calc.pandas(mols)
    for col in df_mordred.columns:
        if df_mordred[col].dtype == 'object':
            df_mordred[col] = pd.to_numeric(df_mordred[col], errors='coerce')
    df_mordred = df_mordred.replace([np.inf, -np.inf], np.nan) # Handle infinities
    df_mordred = df_mordred.fillna(np.nan)
    return df_mordred

def calculate_ecfp4_fingerprints(mols):
    # ... (keep your existing implementation) ...
    if not mols: return pd.DataFrame()
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
def load_and_prepare_train_data_desc():
    st.info(f"Attempting to load PCA training data directly from: {BACE_TRAIN_DATA_URL}")
    try:
        df_train_raw = pd.read_csv(BACE_TRAIN_DATA_URL)
        st.success("Successfully loaded PCA training data.")
        # User-specified columns to drop for PCA training data preparation
        cols_to_drop = ['mol', 'CID', 'Class', 'Model', 'standardized_smiles', 'pIC50']
        # Filter out columns that might not exist to prevent errors
        cols_to_drop_existing = [col for col in cols_to_drop if col in df_train_raw.columns]
        train_descriptors_df = df_train_raw.drop(columns=cols_to_drop_existing).copy()
        
        for col in train_descriptors_df.columns:
            if train_descriptors_df[col].dtype == 'object':
                train_descriptors_df[col] = pd.to_numeric(train_descriptors_df[col], errors='coerce')
        train_descriptors_df = train_descriptors_df.replace([np.inf, -np.inf], np.nan) # Handle infinities
        train_descriptors_df = train_descriptors_df.fillna(np.nan)
        train_descriptor_names = train_descriptors_df.columns.tolist()
        return train_descriptors_df, train_descriptor_names
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred while fetching PCA training data: {http_err}")
        return None, None
    except Exception as e:
        st.error(f"Error processing PCA training data from URL: {e}")
        return None, None

def align_input_descriptors(df_input_mordred, train_descriptor_names):
    # ... (keep your existing implementation) ...
    if df_input_mordred.empty or not train_descriptor_names:
        return pd.DataFrame(columns=train_descriptor_names)
    df_aligned = pd.DataFrame()
    for name in train_descriptor_names:
        if name in df_input_mordred.columns:
            df_aligned[name] = df_input_mordred[name]
        else:
            df_aligned[name] = np.nan # Ensure column exists even if all NaN
    return df_aligned


def generate_pca_plot(df_train_desc, df_input_desc_aligned):
    # ... (Using your version from the prompt which fits PCA on df_train_desc directly)
    if df_train_desc is None or df_train_desc.empty:
        st.warning("Training descriptors for PCA are not available.")
        return
    
    # Handle potential non-finite values before PCA
    df_train_desc_finite = df_train_desc.replace([np.inf, -np.inf], np.nan)
    imputer_train = SimpleImputer(strategy='mean')
    train_desc_imputed = imputer_train.fit_transform(df_train_desc_finite)
    
    # Optional: Scaling (user had it commented out, respecting that)
    # scaler = StandardScaler()
    # train_desc_scaled = scaler.fit_transform(train_desc_imputed)
    # pca_input_train = train_desc_scaled 
    pca_input_train = train_desc_imputed # Using imputed, non-scaled as per user's code structure

    pca = PCA(n_components=2)
    pca.fit(pca_input_train) # Fit on processed training data
    train_pca = pca.transform(pca_input_train)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(train_pca[:, 0], train_pca[:, 1], label='Train Data (BACE)', alpha=0.5, c='blue', edgecolors='k', s=50)

    if not df_input_desc_aligned.empty:
        df_input_desc_aligned_finite = df_input_desc_aligned.replace([np.inf, -np.inf], np.nan)
        # Use the imputer fitted on training data
        input_desc_imputed = imputer_train.transform(df_input_desc_aligned_finite)
        
        # input_desc_scaled = scaler.transform(input_desc_imputed) # if scaler was used
        # pca_input_test = input_desc_scaled
        pca_input_test = input_desc_imputed # Using imputed, non-scaled

        if pd.DataFrame(pca_input_test).isnull().any().any():
             st.error("NaN values detected in data prepared for PCA transformation of input. This will fail.")
             # return # Or handle as appropriate
        else:
            try:
                input_pca = pca.transform(pca_input_test)
                if input_pca.shape[0] > 0:
                    ax.scatter(input_pca[:, 0], input_pca[:, 1], label='Input SMILES', alpha=0.8, c='red', marker='X', s=100, edgecolors='k')
            except ValueError as ve:
                st.error(f"ValueError during PCA transform for input data: {ve}. Check for NaNs or Infs.")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    ax.set_title("PCA of Mordred Descriptors (Input vs BACE Train Set)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

# --- GNN Prediction Functions ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Original GNN Model
ORIGINAL_GNN_MODEL_URL = "https://raw.githubusercontent.com/HenryChritopher02/machine-learning/master/model/gnn/regression/gnn.pth"
ORIGINAL_GNN_MODEL_FILENAME = "gnn_bace_pic50_original.pth"
DEFAULT_NODE_FEATURE_SIZE = 30 # From MolGraphConvFeaturizer

# User-provided params for the original GIN model
ORIGINAL_GIN_PARAMS = {
    'embedding_size': 64,
    'dense_neuron': 1024,
    'num_gin_layers': 2,
    'num_mlp_layers': 2,
    'dropout_value': 0.11225846425316804,
    'dropout_mlp': 0.3937283675371147,
    'learning_rate': 0.004222524354917754 # Note: LR is a training param, not usually model architecture
}

# Hybrid GNN Model
# HYBRID_MODEL_BASE_URL is now imported from paths.py
HYBRID_GNN_MODEL_FILENAME = "gnn.pth" # Assumed name, user needs to verify/provide
HYBRID_MLP1_MODEL_FILENAME = "mlp.pth"   # Assumed name
HYBRID_COMBINED_MODEL_FILENAME = "hybrid.pth" # Assumed name

# User-provided params for the Hybrid model components
HYBRID_MODEL_PARAMS = {
    'embedding_size': 128,
    'dense_neuron': 256,       # Used for GIN_hybrid output and CombinedMLP input part 1
    'inner_dense_neuron': 64,  # Used for MLP1 output and CombinedMLP input part 2
    'outer_dense_neuron': 512, # Used for CombinedMLP hidden layer
    'num_gin_layers': 2,       # For GIN_hybrid
    'inner_num_layers': 2,     # For MLP1
    'outer_num_layers': 1,     # For CombinedMLP
    'dropout_value': 0.10003159585180633,    # For GIN_hybrid
    'inner_dropout_mlp': 0.1576451710515307, # For MLP1
    'outer_dropout_mlp': 0.21937877385535667, # For CombinedMLP
    'learning_rate': 0.00326166166242257 # Training param
}
NUM_DOCKING_FEATURES = 15 # Expected number of docking scores for MLP1

def mol_to_graph_data_for_prediction(smiles_str, featurizer):
    # ... (keep implementation from previous response) ...
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None: st.warning(f"Could not parse SMILES: {smiles_str}"); return None
    try: features = featurizer._featurize(mol)
    except Exception as e: st.warning(f"Could not featurize SMILES '{smiles_str}': {e}"); return None
    if features.node_features is None or features.node_features.size == 0 : st.warning(f"No node features generated for SMILES: {smiles_str}"); return None
    node_features = torch.tensor(features.node_features, dtype=torch.float)
    edge_index = torch.tensor(features.edge_index, dtype=torch.long)
    edge_features = torch.tensor(features.edge_features, dtype=torch.float) if features.edge_features is not None else torch.empty(0, dtype=torch.float)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, smiles=smiles_str, y=torch.tensor([float('nan')], dtype=torch.float))


def create_graph_data_list_from_smiles(standardized_smiles_list):
    # ... (keep implementation from previous response) ...
    data_list = []
    featurizer = MolGraphConvFeaturizer(use_edges=True)
    for smiles_str in standardized_smiles_list:
        graph_data = mol_to_graph_data_for_prediction(smiles_str, featurizer)
        if graph_data is not None: data_list.append(graph_data)
    return data_list

@st.cache_resource
def download_gnn_model_file(model_url, model_save_dir, model_filename):
    # ... (keep implementation from previous response) ...
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_local_path = model_save_dir / model_filename
    if not model_local_path.exists():
        st.info(f"Downloading GNN model from {model_url} to {model_local_path}...")
        try:
            response = requests.get(model_url, stream=True); response.raise_for_status()
            with open(model_local_path, 'wb') as f: shutil.copyfileobj(response.raw, f) # Corrected download
            st.success(f"Model file {model_filename} downloaded successfully.")
        except Exception as e: st.error(f"Failed to download GNN model {model_filename}: {e}"); return None
    else: st.info(f"GNN Model file {model_filename} already exists at {model_local_path}.")
    return model_local_path

def run_original_gnn_prediction(standardized_smiles_list):
    if not standardized_smiles_list: st.warning("No SMILES for Original GNN."); return pd.DataFrame()
    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"Original GNN using device: {device}")

    with st.spinner("Creating graph data..."):
        graph_data_list = create_graph_data_list_from_smiles(standardized_smiles_list)
    if not graph_data_list: st.error("Graph data creation failed."); return pd.DataFrame()
    st.success(f"Graph data created for {len(graph_data_list)} SMILES.")
    graph_loader = DataLoader(graph_data_list, batch_size=32, shuffle=False, num_workers=0)

    model_path = download_gnn_model_file(ORIGINAL_GNN_MODEL_URL, WORKSPACE_PARENT_DIR / "original_gnn", ORIGINAL_GNN_MODEL_FILENAME)
    if not model_path: return pd.DataFrame()

    try:
        # Pass ORIGINAL_GIN_PARAMS to your GIN constructor
        # The GIN constructor in gnn_architecture.py needs to accept feature_size and model_params
        model_architecture = GIN(
            feature_size=DEFAULT_NODE_FEATURE_SIZE, #This is num_node_features
            model_params=ORIGINAL_GIN_PARAMS # Pass the whole dictionary
        ).to(device)
        trained_model = load_model(model_architecture, str(model_path), device) # from gnn_train.py
        if trained_model is None: st.error("Original GNN model loading failed."); return pd.DataFrame()
        st.success("Original GNN model loaded.")
    except Exception as e: st.error(f"Error initializing/loading Original GNN: {e}"); return pd.DataFrame()

    with st.spinner("Predicting pIC50 with Original GNN..."):
        try:
            df_predictions = predict_pic50_gnn(trained_model, graph_loader, device) # from gnn_train.py
        except Exception as e: st.error(f"Error during Original GNN prediction: {e}"); return pd.DataFrame()
            
    if df_predictions.empty: st.warning("Original GNN prediction is empty.")
    else: st.success("Original GNN prediction complete.")
    return df_predictions

def create_num_features_dataloader(docking_scores_df, device):
    """Creates a DataLoader from numerical docking scores."""
    if not isinstance(docking_scores_df, pd.DataFrame):
        st.error("Docking scores must be a pandas DataFrame.")
        return None
        
    # Assuming all columns except a potential 'SMILES' like column are scores
    score_columns = [col for col in docking_scores_df.columns if col not in ['SMILES', 'Ligand ID / SMILES', 'Ligand Base Name', 'standardized_smiles']]

    if len(score_columns) != NUM_DOCKING_FEATURES:
        st.error(f"Expected {NUM_DOCKING_FEATURES} docking score columns, but found {len(score_columns)}. Columns found: {score_columns}")
        return None
    
    X_tensor = torch.tensor(docking_scores_df[score_columns].values, dtype=torch.float).to(device)
    
    # Create placeholder Y tensor as it's not used for prediction but TensorDataset needs it
    Y_placeholder = torch.full_like(X_tensor[:,0], float('nan')).unsqueeze(1).to(device) # Match rows, single column of NaNs
    
    dataset = TensorDataset(X_tensor, Y_placeholder)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    return dataloader


def run_hybrid_gnn_prediction(standardized_smiles_list, docking_scores_df_aligned):
    if not standardized_smiles_list: st.warning("No SMILES for Hybrid GNN."); return pd.DataFrame()
    if docking_scores_df_aligned.empty: st.warning("No docking scores for Hybrid GNN."); return pd.DataFrame()

    set_seed(42); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"Hybrid GNN using device: {device}")

    # 1. Create graph data loader
    with st.spinner("Hybrid: Creating graph data..."):
        graph_data_list = create_graph_data_list_from_smiles(standardized_smiles_list)
    if not graph_data_list: st.error("Hybrid: Graph data creation failed."); return pd.DataFrame()
    st.success(f"Hybrid: Graph data created for {len(graph_data_list)} SMILES.")
    graph_loader = DataLoader(graph_data_list, batch_size=32, shuffle=False, num_workers=0)

    # 2. Create numerical features data loader (docking scores)
    with st.spinner("Hybrid: Creating numerical features data loader from docking scores..."):
        # The docking_scores_df_aligned should already have the correct 15 score columns
        # and be aligned with standardized_smiles_list.
        num_feat_loader = create_num_features_dataloader(docking_scores_df_aligned, device)
    if not num_feat_loader: st.error("Hybrid: Numerical features (docking scores) DataLoader creation failed."); return pd.DataFrame()
    st.success("Hybrid: Numerical features DataLoader created.")

    # 3. Download model components
    # Ensure HYBRID_MODEL_BASE_URL ends with a '/'
    gnn_comp_url = HYBRID_MODEL_BASE_URL + HYBRID_GNN_MODEL_FILENAME
    mlp1_comp_url = HYBRID_MODEL_BASE_URL + HYBRID_MLP1_MODEL_FILENAME
    combined_comp_url = HYBRID_MODEL_BASE_URL + HYBRID_COMBINED_MODEL_FILENAME
    
    model_save_dir = WORKSPACE_PARENT_DIR / "hybrid_gnn_vina125_15"

    gnn_model_path = download_gnn_model_file(gnn_comp_url, model_save_dir, HYBRID_GNN_MODEL_FILENAME)
    mlp1_model_path = download_gnn_model_file(mlp1_comp_url, model_save_dir, HYBRID_MLP1_MODEL_FILENAME)
    combined_model_path = download_gnn_model_file(combined_comp_url, model_save_dir, HYBRID_COMBINED_MODEL_FILENAME)

    if not (gnn_model_path and mlp1_model_path and combined_model_path):
        st.error("Hybrid: Failed to download one or more model components."); return pd.DataFrame()

    # 4. Initialize and load models
    try:
        # GIN_hybrid: input is graph features (node_features=30)
        gin_hybrid_model_arch = GIN_hybrid(
            feature_size=DEFAULT_NODE_FEATURE_SIZE, 
            model_params=HYBRID_MODEL_PARAMS
        ).to(device)
        trained_gin_hybrid_model = load_model(gin_hybrid_model_arch, str(gnn_model_path), device)

        # MLP1: input is docking scores (input_dim=15)
        mlp1_model_arch = MLP1(
            input_dim=NUM_DOCKING_FEATURES, 
            model_params=HYBRID_MODEL_PARAMS
        ).to(device)
        trained_mlp1_model = load_model(mlp1_model_arch, str(mlp1_model_path), device)

        # CombinedMLP: input is concatenation of GIN_hybrid output and MLP1 output
        # Output dim of GIN_hybrid is params['dense_neuron']
        # Output dim of MLP1 is params['inner_dense_neuron']
        combined_mlp_input_dim = HYBRID_MODEL_PARAMS['dense_neuron'] + HYBRID_MODEL_PARAMS['inner_dense_neuron']
        combined_mlp_model_arch = CombinedMLP(
            input_dim=combined_mlp_input_dim, # Corrected from model_params=params['dense_neuron'] + params['inner_dense_neuron']
            model_params=HYBRID_MODEL_PARAMS
        ).to(device)
        trained_combined_mlp_model = load_model(combined_mlp_model_arch, str(combined_model_path), device)

        if not (trained_gin_hybrid_model and trained_mlp1_model and trained_combined_mlp_model):
            st.error("Hybrid: Failed to load one or more trained model components."); return pd.DataFrame()
        st.success("Hybrid: All model components loaded successfully.")

    except Exception as e: st.error(f"Error initializing/loading Hybrid GNN components: {e}"); return pd.DataFrame()

    # 5. Predict
    with st.spinner("Predicting pIC50 with Hybrid GNN model..."):
        try:
            df_predictions = predict_pic50_hybrid(
                trained_gin_hybrid_model, 
                trained_mlp1_model, 
                trained_combined_mlp_model,
                graph_loader, 
                num_feat_loader, 
                device
            )
        except Exception as e: st.error(f"Error during Hybrid GNN prediction: {e}"); return pd.DataFrame()
            
    if df_predictions.empty: st.warning("Hybrid GNN prediction is empty.")
    else: st.success("Hybrid GNN prediction complete.")
    return df_predictions
