import streamlit as st
import subprocess
import os
import stat
import requests
import zipfile
import shutil
from urllib.parse import urljoin
from pathlib import Path
import sys
import pandas as pd

# --- Configuration ---
APP_VERSION = "1.2.5" # Restructured sidebar, results table, cleaner STDERR & UI

# GitHub URL for RECEPTORS and CONFIGS
BASE_GITHUB_URL_FOR_DATA = "https://raw.githubusercontent.com/HenryChritopher02/bace1/main/ensemble-docking/"
GH_API_BASE_URL = "https://api.github.com/repos/"
GH_OWNER = "HenryChritopher02"
GH_REPO = "bace1"
GH_BRANCH = "main"
GH_ENSEMBLE_DOCKING_ROOT_PATH = "ensemble-docking"
RECEPTOR_SUBDIR_GH = "ensemble_protein/"
CONFIG_SUBDIR_GH = "config/"

# --- LOCAL Paths for Helper Scripts and Vina ---
APP_ROOT = Path(".")
ENSEMBLE_DOCKING_DIR_LOCAL = APP_ROOT / "ensemble_docking"
LIGAND_PREPROCESSING_SUBDIR_LOCAL = ENSEMBLE_DOCKING_DIR_LOCAL / "ligand_preprocessing"
SCRUB_PY_LOCAL_PATH = LIGAND_PREPROCESSING_SUBDIR_LOCAL / "scrub.py"
MK_PREPARE_LIGAND_PY_LOCAL_PATH = LIGAND_PREPROCESSING_SUBDIR_LOCAL / "mk_prepare_ligand.py"
VINA_SCREENING_PL_LOCAL_PATH = ENSEMBLE_DOCKING_DIR_LOCAL / "Vina_screening.pl"

VINA_DIR_LOCAL = APP_ROOT / "vina"
VINA_EXECUTABLE_NAME = "vina_1.2.5_linux_x86_64"
VINA_PATH_LOCAL = VINA_DIR_LOCAL / VINA_EXECUTABLE_NAME

WORKSPACE_PARENT_DIR = APP_ROOT / "autodock_workspace"
RECEPTOR_DIR_LOCAL = WORKSPACE_PARENT_DIR / "fetched_receptors"
CONFIG_DIR_LOCAL = WORKSPACE_PARENT_DIR / "fetched_configs"
LIGAND_PREP_DIR_LOCAL = WORKSPACE_PARENT_DIR / "prepared_ligands"
LIGAND_UPLOAD_TEMP_DIR = WORKSPACE_PARENT_DIR / "uploaded_ligands_temp"
ZIP_EXTRACT_DIR_LOCAL = WORKSPACE_PARENT_DIR / "zip_extracted_ligands"
DOCKING_OUTPUT_DIR_LOCAL = APP_ROOT / "autodock_outputs"

# --- Helper Functions (Core Functionality) ---

def initialize_directories():
    dirs_to_create = [
        WORKSPACE_PARENT_DIR, RECEPTOR_DIR_LOCAL, CONFIG_DIR_LOCAL,
        LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR,
        ZIP_EXTRACT_DIR_LOCAL, DOCKING_OUTPUT_DIR_LOCAL
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

def list_files_from_github_repo_dir(owner: str, repo: str, dir_path_in_repo: str, branch: str, file_extension: str = None) -> list[str]:
    api_url = f"{GH_API_BASE_URL}{owner}/{repo}/contents/{dir_path_in_repo}?ref={branch}"
    filenames = []
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        contents = response.json()
        if not isinstance(contents, list):
            st.sidebar.error(f"API Error for {dir_path_in_repo}: Expected list.")
            if isinstance(contents, dict) and 'message' in contents: st.sidebar.error(f"GitHub: {contents['message']}")
            return []
        for item in contents:
            if item.get('type') == 'file':
                if file_extension:
                    if item.get('name', '').lower().endswith(file_extension.lower()):
                        filenames.append(item['name'])
                else:
                    filenames.append(item['name'])
        if not filenames and file_extension:
             st.sidebar.warning(f"No files matching '{file_extension}' found in '{dir_path_in_repo}'.")
    except Exception as e:
        st.sidebar.error(f"Error listing files from GitHub ({dir_path_in_repo}): {e}")
    return filenames

def download_file_from_github(raw_download_base_url, relative_path_segment, local_filename, local_save_dir):
    full_url = urljoin(raw_download_base_url, relative_path_segment)
    local_file_path = Path(local_save_dir) / local_filename
    try:
        response = requests.get(full_url, stream=True, timeout=15)
        response.raise_for_status()
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
        return str(local_file_path)
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error downloading {local_filename}: {e}")
        return None

def make_file_executable(filepath_str):
    if not filepath_str or not os.path.exists(filepath_str): return False
    try:
        os.chmod(filepath_str, os.stat(filepath_str).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return True
    except Exception: # Keep silent on success for most, only show error
        # st.sidebar.error(f"Error making {Path(filepath_str).name} executable.") # Verbose
        return False

def check_script_exists(script_path: Path, script_name: str, is_critical: bool = True):
    if script_path.exists() and script_path.is_file():
        # st.sidebar.caption(f"Found: {script_name}") # Keep sidebar cleaner
        return True
    else:
        msg_func = st.sidebar.error if is_critical else st.sidebar.warning
        msg_func(f"{'CRITICAL: ' if is_critical else ''}`{script_name}` NOT FOUND at `{script_path}`.")
        return False

def check_vina_binary(show_success=True): # Default to showing success for this critical component
    if not VINA_PATH_LOCAL.exists():
        st.sidebar.error(f"Vina exe NOT FOUND at `{VINA_PATH_LOCAL}`. Ensure `{VINA_EXECUTABLE_NAME}` is in `{VINA_DIR_LOCAL}`.")
        return False
    if show_success: st.sidebar.success(f"Vina binary found: `{VINA_PATH_LOCAL.name}`")

    if os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK):
        if show_success: st.sidebar.success("Vina binary is executable.")
        return True
    
    st.sidebar.warning("Vina binary NOT executable. Attempting permission set...")
    if make_file_executable(str(VINA_PATH_LOCAL)) and os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK):
        st.sidebar.success("Execute permission set for Vina.")
        return True
    
    st.sidebar.error("Failed to make Vina executable.")
    st.sidebar.markdown(f"**Action:** `git add --chmod=+x {VINA_DIR_LOCAL.name}/{VINA_EXECUTABLE_NAME}` in repo.")
    return False

def get_smiles_from_pubchem_inchikey(inchikey_str):
    api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey_str}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status(); data = response.json()
        return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except Exception as e: st.warning(f"PubChem API/parse error for {inchikey_str}: {e}"); return None

def run_ligand_prep_script(script_local_path_str, script_args, process_name, ligand_name_for_log):
    if not script_local_path_str: st.error(f"{process_name}: Script path undefined."); return False
    absolute_script_path = str(Path(script_local_path_str).resolve())
    if not os.path.exists(absolute_script_path):
        st.error(f"{process_name} script NOT FOUND: {absolute_script_path}"); return False
    
    command = [sys.executable, absolute_script_path] + [str(arg) for arg in script_args]
    cwd_path_resolved = str(WORKSPACE_PARENT_DIR.resolve())
    if not os.path.exists(cwd_path_resolved):
        st.error(f"Working directory {cwd_path_resolved} for {process_name} missing."); return False
    
    try:
        st.info(f"Running {process_name} for {ligand_name_for_log}...")
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd_path_resolved)
        if result.stdout.strip():
            with st.expander(f"{process_name} STDOUT for {ligand_name_for_log}", expanded=False): st.text(result.stdout)
        # Only show script's STDERR if it's not empty and an actual error occurred (check=True raises CalledProcessError)
        # This will be caught by the except block below.
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error during {process_name} for {ligand_name_for_log} (RC: {e.returncode}):")
        with st.expander(f"{process_name} Details (on error)", expanded=True):
            st.error(f"Command: `{' '.join(e.cmd)}`")
            st.text("STDOUT:\n" + (e.stdout.strip() or "No STDOUT."))
            st.text("STDERR:\n" + (e.stderr.strip() or "No STDERR.")) # Show script STDERR on script error
        return False
    except Exception as e: st.error(f"Unexpected error running {process_name} for {ligand_name_for_log}: {e}"); return False

def convert_smiles_to_pdbqt(smiles_str, ligand_name_base, output_dir_path_for_final_pdbqt, ph_val, skip_taut, skip_acidbase, local_scrub_script_path, local_mk_prepare_script_path):
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)
    relative_sdf_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}_scrubbed.sdf"
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"
    absolute_pdbqt_path = WORKSPACE_PARENT_DIR / relative_pdbqt_filename
    scrub_options = ["--ph", str(ph_val)]
    if skip_taut: scrub_options.append("--skip_tautomer")
    if skip_acidbase: scrub_options.append("--skip_acidbase")
    scrub_args = [smiles_str, "-o", str(relative_sdf_filename)] + scrub_options
    if not run_ligand_prep_script(str(local_scrub_script_path), scrub_args, "scrub.py", ligand_name_base): return None
    mk_prepare_args = ["-i", str(relative_sdf_filename), "-o", str(relative_pdbqt_filename)]
    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base): return None
    return {"id": smiles_str, "pdbqt_path": str(absolute_pdbqt_path), "base_name": ligand_name_base} if absolute_pdbqt_path.exists() else None

def convert_ligand_file_to_pdbqt(input_ligand_file_path_absolute, original_filename, output_dir_path_for_final_pdbqt, local_mk_prepare_script_path):
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)
    ligand_name_base = Path(original_filename).stem
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"
    absolute_pdbqt_path = WORKSPACE_PARENT_DIR / relative_pdbqt_filename
    mk_prepare_args = ["-i", str(Path(input_ligand_file_path_absolute).resolve()), "-o", str(relative_pdbqt_filename)]
    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base): return None
    return {"id": original_filename, "pdbqt_path": str(absolute_pdbqt_path), "base_name": ligand_name_base} if absolute_pdbqt_path.exists() else None

def find_paired_config_for_protein(protein_base_name: str, all_config_paths: list[str]) -> Path | None:
    if not all_config_paths: return None
    patterns_to_try = [f"{protein_base_name}.txt", f"config_{protein_base_name}.txt", f"{protein_base_name}_config.txt"]
    for pattern in patterns_to_try:
        for cfg_path_str in all_config_paths:
            cfg_file = Path(cfg_path_str)
            if cfg_file.name.lower() == pattern.lower(): return cfg_file
    for cfg_path_str in all_config_paths: # Fallback substring match
        cfg_file = Path(cfg_path_str)
        if cfg_file.suffix.lower() == ".txt" and protein_base_name.lower() in cfg_file.stem.lower():
            if "config" in cfg_file.stem.lower() or cfg_file.stem.lower() == protein_base_name.lower(): return cfg_file 
    return None

def parse_vina_log(log_file_path: str) -> float | None:
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f: # Added encoding
            best_score = float('inf') 
            found_score_in_table = False
            in_results_table = False
            for line in f:
                line = line.strip()
                if line.startswith("mode | affinity") or line.startswith("----"): # Vina 1.1.2 or 1.2.x table start
                    in_results_table = True
                    continue
                if "Writing output" in line and in_results_table: # Table definitely over if we were in it
                    break
                if in_results_table and line:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        try:
                            score = float(parts[1])
                            return score # Vina lists scores from best to worst. First one is best.
                        except ValueError: continue
            return None # Should have returned earlier if table with scores found
    except FileNotFoundError: st.warning(f"Log file not found: {log_file_path}"); return None
    except Exception as e: st.warning(f"Error parsing log {log_file_path}: {e}"); return None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- App Pages/Procedures ---
def display_ensemble_docking_procedure():
    st.header(f"Ensemble AutoDock Vina Docking (App v{APP_VERSION})")
    st.markdown("---")
    initialize_directories()

    if 'prepared_ligand_details_list' not in st.session_state: st.session_state.prepared_ligand_details_list = []
    if 'docking_run_outputs' not in st.session_state: st.session_state.docking_run_outputs = []

    with st.sidebar:
        st.header("Docking Setup")
        st.caption("Core Components:")
        scrub_py_ok = check_script_exists(SCRUB_PY_LOCAL_PATH, "scrub.py")
        if scrub_py_ok: make_file_executable(str(SCRUB_PY_LOCAL_PATH))
        mk_prepare_ligand_py_ok = check_script_exists(MK_PREPARE_LIGAND_PY_LOCAL_PATH, "mk_prepare_ligand.py")
        if mk_prepare_ligand_py_ok: make_file_executable(str(MK_PREPARE_LIGAND_PY_LOCAL_PATH))
        vina_screening_pl_ok = check_script_exists(VINA_SCREENING_PL_LOCAL_PATH, "Vina_screening.pl", is_critical=False)
        if vina_screening_pl_ok: make_file_executable(str(VINA_SCREENING_PL_LOCAL_PATH))
        vina_ready = check_vina_binary(show_success=True)
        st.markdown("---")

        st.subheader("Receptor(s)")
        receptor_fetch_method = st.radio(
            "Select Receptors:", ("Fetch ALL .pdbqt from GitHub", "Specify Filenames from GitHub"),
            key="receptor_fetch_method_dockpage", horizontal=True, label_visibility="collapsed"
        )
        receptor_dir_in_repo = f"{GH_ENSEMBLE_DOCKING_ROOT_PATH}/{RECEPTOR_SUBDIR_GH.strip('/')}"
        if receptor_fetch_method == "Fetch ALL .pdbqt from GitHub":
            if st.button("Fetch All Receptors", key="fetch_all_receptors_auto_dockpage", help=f"Fetches all .pdbqt from .../{receptor_dir_in_repo}"):
                st.session_state.fetched_receptor_paths = [] 
                with st.spinner(f"Listing .pdbqt files..."):
                    receptor_filenames = list_files_from_github_repo_dir(GH_OWNER, GH_REPO, receptor_dir_in_repo, GH_BRANCH, ".pdbqt")
                if receptor_filenames:
                    temp_paths = []
                    with st.spinner(f"Downloading {len(receptor_filenames)} receptor(s)..."):
                        for r_name in receptor_filenames:
                            dl_path = f"{RECEPTOR_SUBDIR_GH.strip('/')}/{r_name}"
                            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, dl_path, r_name, RECEPTOR_DIR_LOCAL)
                            if path: temp_paths.append(path)
                    if temp_paths: st.success(f"Fetched {len(temp_paths)} receptors."); st.session_state.fetched_receptor_paths = temp_paths
                    else: st.error("No receptors downloaded.")
                else: st.warning(f"No .pdbqt files found in GitHub directory.")
        elif receptor_fetch_method == "Specify Filenames from GitHub":
            receptor_names_input = st.text_area("Receptor PDBQT Filenames (one per line):", key="receptor_filenames_manual_dockpage", help=f"From .../{receptor_dir_in_repo}/")
            if st.button("Fetch Specified Receptors", key="fetch_specified_receptors_dockpage"):
                if receptor_names_input.strip():
                    names = [n.strip() for n in receptor_names_input.splitlines() if n.strip()]
                    st.session_state.fetched_receptor_paths = []
                    temp_paths = []
                    with st.spinner(f"Downloading {len(names)} receptor(s)..."):
                        for r_name in names:
                            dl_path = f"{RECEPTOR_SUBDIR_GH.strip('/')}/{r_name}"
                            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, dl_path, r_name, RECEPTOR_DIR_LOCAL)
                            if path: temp_paths.append(path)
                    if temp_paths: st.success(f"Fetched {len(temp_paths)} receptors."); st.session_state.fetched_receptor_paths = temp_paths
                    else: st.error("No specified receptors downloaded.")
                else: st.warning("Enter receptor filenames.")
        if 'fetched_receptor_paths' in st.session_state and st.session_state.fetched_receptor_paths:
            exp = st.expander(f"{len(st.session_state.fetched_receptor_paths)} Receptor(s) Ready", expanded=False)
            for p_str in st.session_state.fetched_receptor_paths: exp.caption(f"- {Path(p_str).name}")
        st.markdown("---")

        st.subheader("Vina Config File(s)")
        config_fetch_method = st.radio(
            "Select Config Files:",("Fetch ALL .txt from GitHub", "Specify Filenames from GitHub"),
            key="config_fetch_method_dockpage", horizontal=True, label_visibility="collapsed"
        )
        config_dir_in_repo = f"{GH_ENSEMBLE_DOCKING_ROOT_PATH}/{CONFIG_SUBDIR_GH.strip('/')}"
        if config_fetch_method == "Fetch ALL .txt from GitHub":
            if st.button("Fetch All Configs", key="fetch_all_configs_auto_dockpage", help=f"Fetches all .txt from .../{config_dir_in_repo}"):
                st.session_state.fetched_config_paths = []
                with st.spinner(f"Listing .txt files..."):
                    config_filenames = list_files_from_github_repo_dir(GH_OWNER, GH_REPO, config_dir_in_repo, GH_BRANCH, ".txt")
                if config_filenames:
                    temp_paths = []
                    with st.spinner(f"Downloading {len(config_filenames)} config(s)..."):
                        for c_name in config_filenames:
                            dl_path = f"{CONFIG_SUBDIR_GH.strip('/')}/{c_name}"
                            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, dl_path, c_name, CONFIG_DIR_LOCAL)
                            if path: temp_paths.append(path)
                    if temp_paths: st.success(f"Fetched {len(temp_paths)} configs."); st.session_state.fetched_config_paths = temp_paths
                    else: st.error("No configs downloaded.")
                else: st.warning(f"No .txt files found in GitHub dir.")
        elif config_fetch_method == "Specify Filenames from GitHub":
            config_names_input = st.text_area("Vina Config Filenames (one per line):", key="config_filenames_manual_dockpage", help=f"From .../{config_dir_in_repo}/")
            if st.button("Fetch Specified Configs", key="fetch_specified_configs_dockpage"):
                if config_names_input.strip():
                    names = [n.strip() for n in config_names_input.splitlines() if n.strip()]
                    st.session_state.fetched_config_paths = []
                    temp_paths = []
                    with st.spinner(f"Downloading {len(names)} config(s)..."):
                        for c_name in names:
                            dl_path = f"{CONFIG_SUBDIR_GH.strip('/')}/{c_name}"
                            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, dl_path, c_name, CONFIG_DIR_LOCAL)
                            if path: temp_paths.append(path)
                    if temp_paths: st.success(f"Fetched {len(temp_paths)} configs."); st.session_state.fetched_config_paths = temp_paths
                    else: st.error("No specified configs downloaded.")
                else: st.warning("Enter config filenames.")
        if 'fetched_config_paths' in st.session_state and st.session_state.fetched_config_paths:
            exp = st.expander(f"{len(st.session_state.fetched_config_paths)} Config(s) Ready", expanded=False)
            for p_str in st.session_state.fetched_config_paths: exp.caption(f"- {Path(p_str).name}")
        st.markdown("---")

    # --- Main Page Content for Ensemble Docking ---
    st.subheader("🔬 Ligand Input & Preparation")
    ligand_input_method = st.radio(
        "Choose ligand input method:",
        ("SMILES String", "SMILES File (.txt)", "PDBQT File(s)", "Other Ligand File(s)", "ZIP Archive"),
        key="ligand_method_radio_mainpage", horizontal=True
    )
    current_preparation_ligand_details = [] # Stores details for the current prep action

    if ligand_input_method in ["SMILES String", "SMILES File (.txt)"]:
        with st.expander("SMILES Protonation Options", expanded=True):
            col_ph, col_taut, col_ab = st.columns(3)
            g_ph_val = col_ph.number_input("pH", value=7.4, key="g_ph_val_main_ph")
            g_skip_tautomer = col_taut.checkbox("Skip tautomers", key="g_skip_taut_main_taut")
            g_skip_acidbase = col_ab.checkbox("Skip protomers", key="g_skip_ab_main_ab")

    if ligand_input_method == "SMILES String":
        inchikey_or_smiles_val = st.text_input("InChIKey or SMILES string:", key="smiles_input_main_val")
        use_inchikey = st.checkbox("Input is InChIKey", value=False, key="use_inchikey_main_cb")
        lig_name = st.text_input("Ligand Base Name:", value="lig_smiles", key="lig_name_main_name")
        if st.button("Prepare This SMILES Ligand", key="prep_smiles_main_btn"):
            if inchikey_or_smiles_val and lig_name:
                actual_smiles = inchikey_or_smiles_val
                if use_inchikey:
                    with st.spinner("Fetching SMILES..."): fetched_s = get_smiles_from_pubchem_inchikey(inchikey_or_smiles_val)
                    if fetched_s: actual_smiles = fetched_s; st.info(f"Fetched SMILES: {actual_smiles}")
                    else: st.error("Could not get SMILES."); actual_smiles = None
                if actual_smiles and scrub_py_ok and mk_prepare_ligand_py_ok:
                    detail = convert_smiles_to_pdbqt(actual_smiles, lig_name, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                    if detail: current_preparation_ligand_details.append(detail); st.success(f"Prepared: {Path(detail['pdbqt_path']).name}")
                elif not (scrub_py_ok and mk_prepare_ligand_py_ok): st.error("Prep scripts missing.")
            else: st.warning("Provide SMILES/InChIKey and name.")
            st.session_state.prepared_ligand_details_list = current_preparation_ligand_details

    elif ligand_input_method == "SMILES File (.txt)":
        uploaded_smiles_file = st.file_uploader("Upload .txt (one SMILES per line):", type="txt", key="smiles_file_uploader_main_file")
        if uploaded_smiles_file and st.button("Prepare Ligands from SMILES File", key="prep_smiles_file_main_btn"):
            current_preparation_ligand_details = []
            smiles_list = [line.strip() for line in uploaded_smiles_file.read().decode().splitlines() if line.strip()]
            if not smiles_list: st.warning("No SMILES found in file.")
            elif not (scrub_py_ok and mk_prepare_ligand_py_ok): st.error("Ligand prep scripts missing.")
            else:
                st.info(f"Found {len(smiles_list)} SMILES. Preparing...")
                progress_bar = st.progress(0)
                for i, smiles_str in enumerate(smiles_list):
                    lig_name = f"lig_from_file_{i+1}"
                    detail = convert_smiles_to_pdbqt(smiles_str, lig_name, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                    if detail: current_preparation_ligand_details.append(detail)
                    progress_bar.progress((i + 1) / len(smiles_list))
                st.success(f"Prepared {len(current_preparation_ligand_details)} PDBQTs from SMILES file.")
            st.session_state.prepared_ligand_details_list = current_preparation_ligand_details

    elif ligand_input_method == "PDBQT File(s)":
        uploaded_pdbqt_files = st.file_uploader("Upload PDBQT ligand(s):", type="pdbqt", accept_multiple_files=True, key="pdbqt_uploader_main_file")
        if uploaded_pdbqt_files:
            current_preparation_ligand_details = []
            for up_file in uploaded_pdbqt_files:
                dest_path = LIGAND_PREP_DIR_LOCAL / up_file.name
                with open(dest_path, "wb") as f: f.write(up_file.getbuffer())
                current_preparation_ligand_details.append({"id": up_file.name, "pdbqt_path": str(dest_path), "base_name": Path(up_file.name).stem})
            st.session_state.prepared_ligand_details_list = current_preparation_ligand_details
            st.info(f"Using {len(st.session_state.prepared_ligand_details_list)} uploaded PDBQT file(s).")

    elif ligand_input_method == "Other Ligand File(s)":
        uploaded_other_files = st.file_uploader("Upload other format ligand(s) (e.g. SDF, MOL2):", accept_multiple_files=True, key="other_lig_uploader_main_file")
        if uploaded_other_files and st.button("Convert Uploaded File(s) to PDBQT", key="convert_other_main_btn"):
            current_preparation_ligand_details = []
            if not mk_prepare_ligand_py_ok: st.error("mk_prepare_ligand.py script missing.")
            else:
                st.info(f"Processing {len(uploaded_other_files)} file(s) for conversion...")
                progress_bar = st.progress(0)
                for i, up_file in enumerate(uploaded_other_files):
                    temp_save_path = LIGAND_UPLOAD_TEMP_DIR / up_file.name 
                    temp_save_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(temp_save_path, "wb") as f: f.write(up_file.getbuffer())
                    detail = convert_ligand_file_to_pdbqt(temp_save_path, up_file.name, LIGAND_PREP_DIR_LOCAL, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                    if detail: current_preparation_ligand_details.append(detail)
                    progress_bar.progress((i + 1) / len(uploaded_other_files))
                st.success(f"Prepared {len(current_preparation_ligand_details)} PDBQTs from other files.")
            st.session_state.prepared_ligand_details_list = current_preparation_ligand_details

    elif ligand_input_method == "ZIP Archive":
        uploaded_zip_file = st.file_uploader("Upload ZIP of ligand files:", type="zip", key="zip_uploader_main_file")
        if uploaded_zip_file and st.button("Process Ligands from ZIP", key="process_zip_main_btn"):
            current_preparation_ligand_details = []
            shutil.rmtree(ZIP_EXTRACT_DIR_LOCAL, ignore_errors=True)
            ZIP_EXTRACT_DIR_LOCAL.mkdir(exist_ok=True)
            with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref: zip_ref.extractall(ZIP_EXTRACT_DIR_LOCAL)
            st.info(f"Extracted ZIP to `{ZIP_EXTRACT_DIR_LOCAL}`.")
            files_in_zip = list(p for p in Path(ZIP_EXTRACT_DIR_LOCAL).rglob("*") if p.is_file())
            if not files_in_zip: st.warning("No files in ZIP.")
            else:
                st.info(f"Found {len(files_in_zip)} file(s). Processing...")
                progress_bar = st.progress(0)
                for i, item_path in enumerate(files_in_zip):
                    if item_path.suffix.lower() == ".pdbqt":
                        dest_path = LIGAND_PREP_DIR_LOCAL / item_path.name
                        shutil.copy(item_path, dest_path)
                        current_preparation_ligand_details.append({"id": item_path.name, "pdbqt_path": str(dest_path), "base_name": item_path.stem})
                    elif mk_prepare_ligand_py_ok:
                        detail = convert_ligand_file_to_pdbqt(item_path, item_path.name, LIGAND_PREP_DIR_LOCAL, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                        if detail: current_preparation_ligand_details.append(detail)
                    else: st.warning(f"Skipping {item_path.name} (mk_prepare_ligand.py missing).")
                    progress_bar.progress((i + 1) / len(files_in_zip))
                st.success(f"Prepared {len(current_preparation_ligand_details)} PDBQTs from ZIP.")
            st.session_state.prepared_ligand_details_list = current_preparation_ligand_details
    
    # Display current list of prepared ligands from session state
    if st.session_state.get('prepared_ligand_details_list', []): # Check if list in session_state has items
        exp_lig = st.expander(f"**{len(st.session_state.prepared_ligand_details_list)} Ligand(s) Ready for Docking**", expanded=True)
        for lig_info in st.session_state.prepared_ligand_details_list:
            exp_lig.caption(f"- ID: {lig_info.get('id', 'N/A')}, File: `{Path(lig_info.get('pdbqt_path', '')).name}`")
    st.markdown("---")

    # --- Docking Execution Section ---
    st.subheader("🚀 Docking Execution")
    final_ligand_details_list = st.session_state.get('prepared_ligand_details_list', [])
    current_receptors = st.session_state.get('fetched_receptor_paths', [])
    current_configs = st.session_state.get('fetched_config_paths', [])

    if not vina_ready: st.error("Vina executable not set up.")
    elif not current_receptors: st.warning("No receptors available.")
    elif not final_ligand_details_list: st.warning("No ligands prepared.")
    elif not current_configs: st.warning("No Vina configs available.")
    else:
        st.info(f"Ready for {len(final_ligand_details_list)} ligand(s) vs {len(current_receptors)} receptor(s).")
        use_vina_screening_perl = False
        if len(final_ligand_details_list) > 1 and vina_screening_pl_ok and VINA_SCREENING_PL_LOCAL_PATH.exists():
            use_vina_screening_perl = st.checkbox("Use `Vina_screening.pl` (Strict Protein-Config Pairing)?", value=True, key="use_perl_dockpage_main")
        
        if st.button("Start Docking Run", key="start_docking_main_btn_main", type="primary"):
            st.session_state.docking_run_outputs = [] 
            DOCKING_OUTPUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

            if use_vina_screening_perl:
                st.markdown("##### Docking via `Vina_screening.pl` (Strict Protein-Config Pairing)")
                if not (vina_screening_pl_ok and VINA_SCREENING_PL_LOCAL_PATH.exists() and os.access(VINA_SCREENING_PL_LOCAL_PATH, os.X_OK)):
                    st.error("Local `Vina_screening.pl` script not available or not executable.")
                else:
                    ligand_list_file_for_perl = WORKSPACE_PARENT_DIR / "ligands_for_perl.txt"
                    with open(ligand_list_file_for_perl, "w") as f:
                        for lig_detail in final_ligand_details_list:
                            f.write(str(Path(lig_detail['pdbqt_path']).resolve()) + "\n")
                    
                    overall_docking_progress = st.progress(0)
                    receptors_processed_count = 0; skipped_receptor_count = 0
                    for i, receptor_path_str in enumerate(current_receptors):
                        receptor_file = Path(receptor_path_str); protein_base = receptor_file.stem
                        st.markdown(f"--- \n**Receptor: `{receptor_file.name}` (Perl Mode)**")
                        config_to_use = None
                        if not current_configs: st.error(f"No Vina configs for {receptor_file.name}."); skipped_receptor_count +=1; continue
                        elif len(current_configs) == 1: config_to_use = Path(current_configs[0])
                        else: config_to_use = find_paired_config_for_protein(protein_base, current_configs)
                        if not config_to_use: st.warning(f"No paired config for `{receptor_file.name}`. Skipping."); skipped_receptor_count +=1; overall_docking_progress.progress((receptors_processed_count + skipped_receptor_count) / len(current_receptors)); continue
                        st.caption(f"Using config: `{config_to_use.name}`")
                        
                        temp_receptor_path = WORKSPACE_PARENT_DIR / receptor_file.name
                        shutil.copy(receptor_file, temp_receptor_path)
                        cmd_perl = ["perl", str(VINA_SCREENING_PL_LOCAL_PATH.resolve()),
                                    str(VINA_PATH_LOCAL.resolve()), str(temp_receptor_path.resolve()),
                                    str(config_to_use.resolve()), protein_base]
                        try:
                            with open(ligand_list_file_for_perl, "r") as f_ligs: ligand_stdin = f_ligs.read()
                            proc = subprocess.Popen(cmd_perl, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(WORKSPACE_PARENT_DIR.resolve()))
                            stdout_p, stderr_p = proc.communicate(input=ligand_stdin) # Pass actual content
                            if proc.returncode != 0: st.error(f"Perl script failed for `{protein_base}` (RC: {proc.returncode}).")
                            
                            if stdout_p.strip():
                                with st.expander(f"Perl STDOUT for {protein_base}", expanded=False): st.text(stdout_p)
                            if stderr_p.strip() and proc.returncode != 0: 
                                with st.expander(f"Perl STDERR for {protein_base}", expanded=True): st.text(stderr_p)
                            
                            perl_protein_out_dir = WORKSPACE_PARENT_DIR / protein_base # As defined in your more complete Perl script
                            if perl_protein_out_dir.is_dir():
                                for lig_detail in final_ligand_details_list:
                                    log_fn = f"{lig_detail['base_name']}_{protein_base}_log.txt" 
                                    expected_log_file = perl_protein_out_dir / log_fn
                                    if not expected_log_file.exists(): expected_log_file = perl_protein_out_dir / f"{lig_detail['base_name']}_log.txt" # Fallback
                                    
                                    if expected_log_file.exists():
                                        score = parse_vina_log(str(expected_log_file))
                                        if score is not None: st.session_state.docking_run_outputs.append({"ligand_id": lig_detail["id"], "protein_stem": protein_base, "config_stem": config_to_use.stem, "score": score})
                            else: st.warning(f"Perl output dir not found: {perl_protein_out_dir}. Scores may not be parsed for this protein.")
                        except Exception as e_p: st.error(f"Error running Perl script for `{protein_base}`: {e_p}")
                        finally: 
                            if temp_receptor_path.exists(): temp_receptor_path.unlink(missing_ok=True)
                        receptors_processed_count += 1
                        overall_docking_progress.progress((receptors_processed_count + skipped_receptor_count) / len(current_receptors))
                    if skipped_receptor_count > 0: st.warning(f"{skipped_receptor_count} receptor(s) skipped.")

            else: # Direct Vina calls
                st.markdown("##### Docking via Direct Vina Calls (Strict Protein-Config Pairing)")
                planned_docking_jobs = []
                skipped_receptors_direct_mode = set()
                for rec_path_str in current_receptors:
                    receptor_file = Path(rec_path_str); protein_base = receptor_file.stem
                    paired_config_file = None
                    if not current_configs:
                        if protein_base not in skipped_receptors_direct_mode: st.warning(f"No configs for '{receptor_file.name}'. Skip."); skipped_receptors_direct_mode.add(protein_base)
                        continue
                    elif len(current_configs) == 1: paired_config_file = Path(current_configs[0])
                    else: paired_config_file = find_paired_config_for_protein(protein_base, current_configs)
                    if paired_config_file:
                        for lig_detail in final_ligand_details_list:
                            planned_docking_jobs.append({"ligand_info": lig_detail, "receptor": receptor_file, "config": paired_config_file})
                    elif protein_base not in skipped_receptors_direct_mode: st.warning(f"No matching config for '{receptor_file.name}'. Skip."); skipped_receptors_direct_mode.add(protein_base)
                
                num_total_direct_jobs = len(planned_docking_jobs)
                st.info(f"Planning {num_total_direct_jobs} Vina jobs (protein-config pairs).")
                if num_total_direct_jobs > 0:
                    job_counter = 0; overall_docking_progress = st.progress(0)
                    for job_spec in planned_docking_jobs:
                        lig_info, receptor_file, config_file = job_spec["ligand_info"], job_spec["receptor"], job_spec["config"]
                        ligand_file = Path(lig_info["pdbqt_path"])
                        job_counter += 1
                        st.markdown(f"--- \n**Job {job_counter}/{num_total_direct_jobs}:** `{receptor_file.name}` + `{ligand_file.name}` (Config: `{config_file.name}`)")
                        output_base = f"{receptor_file.stem}_{lig_info['base_name']}_{config_file.stem}"
                        output_pdbqt_docked = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base}_out.pdbqt"
                        output_log_file = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base}_log.txt"
                        cmd_vina = [str(VINA_PATH_LOCAL.resolve()), "--receptor", str(receptor_file.resolve()),
                                      "--ligand", str(ligand_file.resolve()), "--config", str(config_file.resolve()),
                                      "--out", str(output_pdbqt_docked.resolve()), "--log", str(output_log_file.resolve())]
                        try:
                            res_vina = subprocess.run(cmd_vina, capture_output=True, text=True, check=True, cwd=str(WORKSPACE_PARENT_DIR.resolve()))
                            st.success("Vina job OK!")
                            if res_vina.stdout.strip():
                                with st.expander("Vina STDOUT", expanded=False): st.text(res_vina.stdout)
                            # Vina STDERR is hidden on success for Direct calls
                            score = parse_vina_log(str(output_log_file))
                            if score is not None:
                                st.session_state.docking_run_outputs.append({
                                    "ligand_id": lig_info["id"], "protein_stem": receptor_file.stem,
                                    "config_stem": config_file.stem, "score": score })
                            if output_pdbqt_docked.exists():
                                with open(output_pdbqt_docked, "rb") as fp: st.download_button(f"DL Docked PDBQT", fp, output_pdbqt_docked.name, "application/octet-stream", key=f"dl_pdbqt_{job_counter}")
                            if output_log_file.exists():
                                with open(output_log_file, "r", encoding='utf-8') as fp: st.download_button(f"DL Log", fp.read(), output_log_file.name, "text/plain",  key=f"dl_log_{job_counter}")
                        except subprocess.CalledProcessError as e_vina:
                            st.error(f"Vina job FAILED (RC: {e_vina.returncode}).")
                            with st.expander("Vina Error Details", expanded=True):
                                st.error(f"Cmd: `{' '.join(e_vina.cmd)}`"); st.text("STDOUT:\n" + (e_vina.stdout.strip() or "No STDOUT.")); st.text("STDERR:\n" + (e_vina.stderr.strip() or "No STDERR."))
                        except Exception as e_gen: st.error(f"Unexpected error in Vina job: {e_gen}")
                        if num_total_direct_jobs > 0: overall_docking_progress.progress(job_counter / num_total_direct_jobs)
            
            if st.session_state.docking_run_outputs:
                st.markdown("---"); st.subheader("📊 Docking Results Summary")
                try:
                    df_detailed = pd.DataFrame(st.session_state.docking_run_outputs)
                    if not df_detailed.empty and 'ligand_id' in df_detailed.columns and 'score' in df_detailed.columns:
                        best_score_indices = df_detailed.loc[df_detailed.groupby('ligand_id')['score'].idxmin()]
                        summary_data = [{"Input SMILES/Ligand ID": row["ligand_id"],
                                         "Best Docking Score (kcal/mol)": row["score"],
                                         "Achieved with Protein": row["protein_stem"],
                                         "Using Config": row["config_stem"]} for _, row in best_score_indices.iterrows()]
                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df_summary.set_index("Input SMILES/Ligand ID"))
                        csv_summary = convert_df_to_csv(df_summary)
                        st.download_button("Download Summary (CSV)", csv_summary, "docking_summary.csv", "text/csv", key="dl_summary_csv")
                    elif not df_detailed.empty:
                        st.warning("Could not display best score summary (missing expected columns). Displaying all parsed scores:"); st.dataframe(df_detailed)
                    else: st.info("No docking scores were successfully parsed/recorded.")
                except Exception as e_df:
                    st.error(f"Error generating summary table: {e_df}")
                    st.caption("Raw results (first 5):"); st.json(st.session_state.docking_run_outputs[:5])
            else: st.info("No docking outputs recorded to summarize.")
            st.balloons()
            st.header("🏁 Docking Run Finished 🏁")
            st.caption(f"Outputs are in `{DOCKING_OUTPUT_DIR_LOCAL}` or `{WORKSPACE_PARENT_DIR}` subdirectories.")

def display_about_page():
    st.header("About This Application")
    st.markdown(f"**Ensemble AutoDock Vina App - v{APP_VERSION}**")
    st.markdown("""
    This application facilitates molecular docking simulations using AutoDock Vina, 
    allowing for ensemble docking approaches.

    **Features:**
    - Preparation of ligands from SMILES strings or various file formats.
    - Docking against one or multiple receptor structures.
    - Utilization of specific or multiple Vina configuration files.
    - Options for using a Perl-based screening script or direct Vina calls.
    - Summarization of best docking scores.

    **Instructions:**
    1.  **Setup (Sidebar):** Ensure Vina & helper scripts are found. Fetch Receptors & Configs.
    2.  **Ligand Input (Main Page):** Provide SMILES or upload files. Click "Prepare...".
    3.  **Docking (Main Page):** Choose mode & click "Start Docking Run".
    4.  **Results (Main Page):** Summary table will appear. Download links for outputs.

    **Repository Structure:** (Place these in your app's GitHub repo)
    - `your_app_root/`
        - `streamlit_app.py` (this file)
        - `ensemble_docking/` (subfolder)
            - `ligand_preprocessing/scrub.py` (Ensure it has `from molscrub import Scrub`)
            - `ligand_preprocessing/mk_prepare_ligand.py`
            - `Vina_screening.pl` (Ensure it's adapted for new arguments)
        - `vina/vina_1.2.5_linux_x86_64` (Vina executable, `chmod +x`)
        - `requirements.txt` (e.g., `streamlit`, `requests`, `pandas`, `molscrub`)
        - `packages.txt` (if using Perl script, add `perl`)
    """)

# --- Main App Execution Logic ---
def main():
    st.sidebar.image("https://raw.githubusercontent.com/HenryChritopher02/bace1/main/logo.png", width=100) 
    st.sidebar.title("Docking Suite")
    
    app_mode = st.sidebar.radio(
        "Select Procedure:",
        ("Ensemble Docking", "About"),
        key="app_mode_selector_main",
        horizontal=True, label_visibility="collapsed"
    )
    st.sidebar.markdown("---")

    if app_mode == "Ensemble Docking":
        display_ensemble_docking_procedure()
    elif app_mode == "About":
        display_about_page()

if __name__ == "__main__":
    main()
