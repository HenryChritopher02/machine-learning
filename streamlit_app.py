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

# Added RDKit imports for the standardize function
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

APP_VERSION = "1.2.5"

BASE_GITHUB_URL_FOR_DATA = "https://raw.githubusercontent.com/HenryChritopher02/bace1/main/ensemble-docking/"
GH_API_BASE_URL = "https://api.github.com/repos/"
GH_OWNER = "HenryChritopher02"
GH_REPO = "bace1"
GH_BRANCH = "main"
GH_ENSEMBLE_DOCKING_ROOT_PATH = "ensemble-docking"
RECEPTOR_SUBDIR_GH = "ensemble_protein/"
CONFIG_SUBDIR_GH = "config/"

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

# --- New Standardize Function ---
def standardize_smiles_rdkit(smiles, invalid_smiles_list): # Renamed to avoid conflict if user has other 'standardize'
    """Standardizes a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)  # Sanitize = True by default
        if mol is None:
            # st.warning(f"RDKit MolFromSmiles returned None for: {smiles}") # Use st.warning for Streamlit UI
            invalid_smiles_list.append(smiles)
            return None # Explicitly return None if MolFromSmiles fails

        # Standard RDKit sanitization, kekulization, and hydrogen removal
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol) # Kekulize before removing Hs if needed for aromaticity perception
        mol = Chem.RemoveHs(mol) # Remove explicit hydrogens

        # Neutralize charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Reionize (optional, depending on desired protonation state handling before Dimorphite-DL/scrub.py)
        # For now, we align with the general standardization steps.
        # If specific ionization is handled by scrub.py, this step might be redundant or need careful consideration.
        # mol = rdMolStandardize.Reionizer().reionize(mol) # Example if reionization is needed

        # Disconnect metal atoms
        metal_disconnector = rdMolStandardize.MetalDisconnector()
        mol = metal_disconnector.Disconnect(mol)


        # Get the largest fragment (parent molecule)
        mol = rdMolStandardize.FragmentParent(mol)

        # Ensure stereochemistry is assigned
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        # Generate canonical SMILES (non-isomeric, kekulized for consistency if that's the goal)
        # isomericSmiles=False will remove stereo info from SMILES, which might be desired for some workflows
        # but often retaining it (isomericSmiles=True) is preferred up until 3D generation.
        # kekuleSmiles=True can sometimes help with consistency but may not always be necessary.
        standardized_smiles_out = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) # Retain isomerism by default
        return standardized_smiles_out
    except Exception as e:
        st.warning(f"Error standardizing SMILES '{smiles}': {e}")
        invalid_smiles_list.append(smiles)
        return None
# --- End of Standardize Function ---

def initialize_directories():
    dirs_to_create = [
        WORKSPACE_PARENT_DIR, RECEPTOR_DIR_LOCAL, CONFIG_DIR_LOCAL,
        LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR,
        ZIP_EXTRACT_DIR_LOCAL, DOCKING_OUTPUT_DIR_LOCAL,
        ENSEMBLE_DOCKING_DIR_LOCAL, LIGAND_PREPROCESSING_SUBDIR_LOCAL, VINA_DIR_LOCAL
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
            st.sidebar.caption(f"No files matching '{file_extension}' found in '{dir_path_in_repo}'.")
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
        st.sidebar.error(f"Error downloading {local_filename} from {full_url}: {e}")
        return None

def make_file_executable(filepath_str):
    if not filepath_str or not os.path.exists(filepath_str):
        #st.sidebar.warning(f"Make executable: File not found at {filepath_str}") # Can be noisy
        return False
    try:
        os.chmod(filepath_str, os.stat(filepath_str).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to make {filepath_str} executable: {e}")
        return False

def check_script_exists(script_path: Path, script_name: str, is_critical: bool = True):
    if script_path.exists() and script_path.is_file(): return True
    msg_func = st.sidebar.error if is_critical else st.sidebar.warning
    msg_func(f"{'CRITICAL: ' if is_critical else ''}`{script_name}` NOT FOUND at `{script_path}`.")
    return False

def check_vina_binary(show_success=True):
    if not VINA_PATH_LOCAL.exists():
        st.sidebar.error(f"Vina exe NOT FOUND at `{VINA_PATH_LOCAL}`. Ensure `{VINA_EXECUTABLE_NAME}` is in `{VINA_DIR_LOCAL}`.")
        return False
    if show_success: st.sidebar.success(f"Vina binary found: `{VINA_PATH_LOCAL.name}`")

    if os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK):
        if show_success: st.sidebar.success("Vina binary is executable.")
        return True
    else:
        st.sidebar.warning("Vina binary NOT executable by os.access. Attempting permission set...")
        if make_file_executable(str(VINA_PATH_LOCAL)):
            if os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK):
                st.sidebar.success("Execute permission successfully set for Vina and verified.")
                return True
            else:
                st.sidebar.error("Failed to make Vina executable (os.access still fails after chmod).")
                st.sidebar.markdown(f"**Manual Action Needed:** `git add --chmod=+x {VINA_DIR_LOCAL.name}/{VINA_EXECUTABLE_NAME}` in your local repo, commit, and push. Or ensure the file has execute permissions in your deployment environment.")
                return False
        else:
            st.sidebar.error("Failed to make Vina executable (chmod call failed).")
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

    if not os.access(absolute_script_path, os.X_OK):
        #st.info(f"Attempting to make {process_name} script executable: {absolute_script_path}") # Can be noisy
        if not make_file_executable(absolute_script_path) or not os.access(absolute_script_path, os.X_OK):
            st.error(f"Failed to make {process_name} script executable. Cannot run.")
            return False

    command = [sys.executable, absolute_script_path] + [str(arg) for arg in script_args]
    cwd_path_resolved = str(WORKSPACE_PARENT_DIR.resolve())
    if not os.path.exists(cwd_path_resolved):
        st.error(f"Working directory {cwd_path_resolved} for {process_name} missing."); return False
    try:
        #st.info(f"Running {process_name} for {ligand_name_for_log}...") # Can be noisy
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd_path_resolved)
        if result.stdout.strip():
            with st.expander(f"{process_name} STDOUT for {ligand_name_for_log}", expanded=False): st.text(result.stdout)
        # Only show STDERR if it seems like an error or significant warning
        # if result.stderr.strip() and ("error" in result.stderr.lower() or "warning" in result.stderr.lower() or "fail" in result.stderr.lower()):
        #     with st.expander(f"{process_name} STDERR for {ligand_name_for_log}", expanded=True): st.text(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error during {process_name} for {ligand_name_for_log} (RC: {e.returncode}):")
        with st.expander(f"{process_name} Details (on error)", expanded=True):
            st.error(f"Command: `{' '.join(e.cmd)}`")
            st.text("STDOUT:\n" + (e.stdout.strip() or "No STDOUT."))
            st.text("STDERR:\n" + (e.stderr.strip() or "No STDERR."))
        return False
    except Exception as e: st.error(f"Unexpected error running {process_name} for {ligand_name_for_log}: {e}"); return False

def convert_smiles_to_pdbqt(smiles_str, ligand_name_base, output_dir_path_for_final_pdbqt, ph_val, skip_taut, skip_acidbase, local_scrub_script_path, local_mk_prepare_script_path):
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)
    relative_sdf_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}_scrubbed.sdf"
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"

    absolute_sdf_path_for_check = WORKSPACE_PARENT_DIR / relative_sdf_filename
    absolute_pdbqt_path_for_return = WORKSPACE_PARENT_DIR / relative_pdbqt_filename

    scrub_options = ["--ph", str(ph_val)]
    if skip_taut: scrub_options.append("--skip_tautomer")
    if skip_acidbase: scrub_options.append("--skip_acidbase")
    # `smiles_str` here is the already standardized SMILES
    scrub_args = [smiles_str, "-o", str(relative_sdf_filename)] + scrub_options

    if not run_ligand_prep_script(str(local_scrub_script_path), scrub_args, "scrub.py", ligand_name_base): return None
    if not absolute_sdf_path_for_check.exists():
        st.error(f"scrub.py did not produce expected output: {absolute_sdf_path_for_check}")
        return None

    mk_prepare_args = ["-i", str(relative_sdf_filename), "-o", str(relative_pdbqt_filename)]
    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base): return None

    return {"id": smiles_str, "pdbqt_path": str(absolute_pdbqt_path_for_return), "base_name": ligand_name_base} if absolute_pdbqt_path_for_return.exists() else None

def convert_ligand_file_to_pdbqt(input_ligand_file_path_absolute, original_filename, output_dir_path_for_final_pdbqt, local_mk_prepare_script_path):
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)
    ligand_name_base = Path(original_filename).stem

    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"
    absolute_pdbqt_path_for_return = WORKSPACE_PARENT_DIR / relative_pdbqt_filename

    mk_prepare_args = ["-i", str(Path(input_ligand_file_path_absolute).resolve()), "-o", str(relative_pdbqt_filename)]

    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base): return None
    return {"id": original_filename, "pdbqt_path": str(absolute_pdbqt_path_for_return), "base_name": ligand_name_base} if absolute_pdbqt_path_for_return.exists() else None

def find_paired_config_for_protein(protein_base_name: str, all_config_paths: list[str]) -> Path | None:
    if not all_config_paths: return None
    patterns_to_try = [f"{protein_base_name}.txt", f"config_{protein_base_name}.txt", f"{protein_base_name}_config.txt"]
    for pattern in patterns_to_try:
        for cfg_path_str in all_config_paths:
            cfg_file = Path(cfg_path_str)
            if cfg_file.name.lower() == pattern.lower(): return cfg_file
    for cfg_path_str in all_config_paths:
        cfg_file = Path(cfg_path_str)
        if cfg_file.suffix.lower() == ".txt" and protein_base_name.lower() in cfg_file.stem.lower():
            if "config" in cfg_file.stem.lower() or cfg_file.stem.lower() == protein_base_name.lower(): return cfg_file
    return None

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def parse_score_from_pdbqt(pdbqt_file_path: str) -> float | None:
    try:
        resolved_path = Path(pdbqt_file_path).resolve()
        if not resolved_path.exists():
            st.warning(f"PDBQT file for score parsing not found: {resolved_path}")
            return None
        if resolved_path.stat().st_size == 0:
            st.warning(f"PDBQT file for score parsing is empty: {resolved_path}")
            return None

        with open(resolved_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line_content in enumerate(lines):
            current_line_stripped = line_content.strip()
            if current_line_stripped.upper().startswith("REMARK VINA RESULT:"):
                parts = current_line_stripped.split(':', 1)
                if len(parts) > 1:
                    score_values_str = parts[1].strip()
                    score_values_list = score_values_str.split()
                    if score_values_list:
                        try:
                            score = float(score_values_list[0])
                            return score
                        except ValueError:
                            st.warning(f"ValueError converting score part '{score_values_list[0]}' to float in {resolved_path.name} on line {i+1}.")
                            return None
                    else:
                        st.warning(f"Score part list is empty after split for VINA RESULT line in {resolved_path.name}.")
                        return None
                else:
                    st.warning(f"'REMARK VINA RESULT:' found, but line splitting by ':' failed in {resolved_path.name}.")
                    return None

        st.warning(f"Could not find 'REMARK VINA RESULT:' in {resolved_path.name}.")
        return None

    except Exception as e:
        st.error(f"Unexpected error parsing PDBQT file {Path(pdbqt_file_path).name}: {e}")
        return None

def display_ensemble_docking_procedure():
    st.header(f"Ensemble AutoDock Vina Docking (App v{APP_VERSION})")
    st.markdown("---")
    initialize_directories()

    if 'prepared_ligand_details_list' not in st.session_state:
        st.session_state.prepared_ligand_details_list = []
    if 'docking_run_outputs' not in st.session_state:
        st.session_state.docking_run_outputs = []
    if 'invalid_smiles_during_standardization' not in st.session_state: # To track SMILES that failed standardization
        st.session_state.invalid_smiles_during_standardization = []


    with st.sidebar:
        st.header("⚙️ Docking Setup")
        st.caption("Core Components Status:")
        scrub_py_ok = check_script_exists(SCRUB_PY_LOCAL_PATH, "scrub.py")
        if scrub_py_ok: make_file_executable(str(SCRUB_PY_LOCAL_PATH))
        mk_prepare_ligand_py_ok = check_script_exists(MK_PREPARE_LIGAND_PY_LOCAL_PATH, "mk_prepare_ligand.py")
        if mk_prepare_ligand_py_ok: make_file_executable(str(MK_PREPARE_LIGAND_PY_LOCAL_PATH))
        vina_screening_pl_ok = check_script_exists(VINA_SCREENING_PL_LOCAL_PATH, "Vina_screening.pl", is_critical=False)
        if vina_screening_pl_ok: make_file_executable(str(VINA_SCREENING_PL_LOCAL_PATH))

        vina_ready = check_vina_binary(show_success=True)
        st.markdown("---")

        # Receptor and Config fetching UI ( 그대로 유지 )
        st.subheader(" Receptor(s)")
        receptor_fetch_method = st.radio("Fetch Receptors:", ("All from GitHub", "Specify from GitHub"), key="receptor_fetch_method_dockpage", horizontal=True, label_visibility="collapsed")
        receptor_dir_in_repo = f"{GH_ENSEMBLE_DOCKING_ROOT_PATH}/{RECEPTOR_SUBDIR_GH.strip('/')}"
        if receptor_fetch_method == "All from GitHub":
            if st.button("Fetch All Receptors", key="fetch_all_receptors_auto_dockpage", help=f"Fetches all .pdbqt from .../{receptor_dir_in_repo}"):
                st.session_state.fetched_receptor_paths = []
                with st.spinner(f"Listing .pdbqt files..."): receptor_filenames = list_files_from_github_repo_dir(GH_OWNER, GH_REPO, receptor_dir_in_repo, GH_BRANCH, ".pdbqt")
                if receptor_filenames:
                    temp_paths = []; st.success(f"Found {len(receptor_filenames)} receptors. Downloading...")
                    with st.spinner(f"Downloading..."):
                        for r_name in receptor_filenames:
                            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"{RECEPTOR_SUBDIR_GH.strip('/')}/{r_name}", r_name, RECEPTOR_DIR_LOCAL)
                            if path: temp_paths.append(path)
                    if temp_paths: st.success(f"Fetched {len(temp_paths)} receptors."); st.session_state.fetched_receptor_paths = temp_paths
                    else: st.error("No receptors downloaded.")
                else: st.warning(f"No .pdbqt files found in GitHub directory.")
        else: 
            receptor_names_input = st.text_area("Receptor Filenames (one per line):", key="receptor_filenames_manual_dockpage", height=100, help=f"From .../{receptor_dir_in_repo}/")
            if st.button("Fetch Specified Receptors", key="fetch_specified_receptors_dockpage"):
                if receptor_names_input.strip():
                    names = [n.strip() for n in receptor_names_input.splitlines() if n.strip()]
                    st.session_state.fetched_receptor_paths = []; temp_paths = []
                    with st.spinner(f"Downloading {len(names)} receptor(s)..."):
                        for r_name in names:
                            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"{RECEPTOR_SUBDIR_GH.strip('/')}/{r_name}", r_name, RECEPTOR_DIR_LOCAL)
                            if path: temp_paths.append(path)
                    if temp_paths: st.success(f"Fetched {len(temp_paths)} receptors."); st.session_state.fetched_receptor_paths = temp_paths
                    else: st.error("No specified receptors downloaded.")
                else: st.warning("Enter receptor filenames.")
        if st.session_state.get('fetched_receptor_paths'):
            exp = st.expander(f"{len(st.session_state.fetched_receptor_paths)} Receptor(s) Ready", expanded=False)
            for p_str in st.session_state.fetched_receptor_paths: exp.caption(f"- {Path(p_str).name}")
        st.markdown("---")

        st.subheader("Vina Config File(s)")
        config_fetch_method = st.radio("Fetch Configs:",("All .txt from GitHub", "Specify from GitHub"), key="config_fetch_method_dockpage", horizontal=True, label_visibility="collapsed")
        config_dir_in_repo = f"{GH_ENSEMBLE_DOCKING_ROOT_PATH}/{CONFIG_SUBDIR_GH.strip('/')}"
        if config_fetch_method == "All .txt from GitHub":
            if st.button("Fetch All Configs", key="fetch_all_configs_auto_dockpage", help=f"Fetches all .txt from .../{config_dir_in_repo}"):
                st.session_state.fetched_config_paths = []
                with st.spinner(f"Listing .txt files..."): config_filenames = list_files_from_github_repo_dir(GH_OWNER, GH_REPO, config_dir_in_repo, GH_BRANCH, ".txt")
                if config_filenames:
                    temp_paths = []; st.success(f"Found {len(config_filenames)} configs. Downloading...")
                    with st.spinner(f"Downloading..."):
                        for c_name in config_filenames:
                            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"{CONFIG_SUBDIR_GH.strip('/')}/{c_name}", c_name, CONFIG_DIR_LOCAL)
                            if path: temp_paths.append(path)
                    if temp_paths: st.success(f"Fetched {len(temp_paths)} configs."); st.session_state.fetched_config_paths = temp_paths
                    else: st.error("No configs downloaded.")
                else: st.warning(f"No .txt files found in GitHub directory.")
        else: 
            config_names_input = st.text_area("Config Filenames (one per line):", key="config_filenames_manual_dockpage", height=100, help=f"From .../{config_dir_in_repo}/")
            if st.button("Fetch Specified Configs", key="fetch_specified_configs_dockpage"):
                if config_names_input.strip():
                    names = [n.strip() for n in config_names_input.splitlines() if n.strip()]
                    st.session_state.fetched_config_paths = []; temp_paths = []
                    with st.spinner(f"Downloading {len(names)} config(s)..."):
                        for c_name in names:
                            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"{CONFIG_SUBDIR_GH.strip('/')}/{c_name}", c_name, CONFIG_DIR_LOCAL)
                            if path: temp_paths.append(path)
                    if temp_paths: st.success(f"Fetched {len(temp_paths)} configs."); st.session_state.fetched_config_paths = temp_paths
                    else: st.error("No specified configs downloaded.")
                else: st.warning("Enter config filenames.")
        if st.session_state.get('fetched_config_paths'):
            exp = st.expander(f"{len(st.session_state.fetched_config_paths)} Config(s) Ready", expanded=False)
            for p_str in st.session_state.fetched_config_paths: exp.caption(f"- {Path(p_str).name}")
        st.markdown("---")

        if st.button("Clear All Prepared Ligands", key="clear_ligands_btn"):
            st.session_state.prepared_ligand_details_list = []
            st.session_state.invalid_smiles_during_standardization = [] # Also clear invalid SMILES list
            st.success("All prepared ligands and invalid SMILES records have been cleared.")
        st.markdown("---")


    st.subheader("🔬 Ligand Input & Preparation")
    ligand_input_method = st.radio(
        "Choose ligand input method:",
        ("SMILES String", "SMILES File (.txt)", "PDBQT File(s)", "Other Ligand File(s)", "ZIP Archive"),
        key="ligand_method_radio_mainpage", horizontal=True )

    g_ph_val, g_skip_tautomer, g_skip_acidbase = 7.4, False, False
    if ligand_input_method in ["SMILES String", "SMILES File (.txt)"]:
        with st.expander("SMILES Protonation Options (via scrub.py)", expanded=False): # Clarified this is for scrub.py
            g_ph_val = st.number_input("pH for scrub.py", value=7.4, key="g_ph_val_main_ph", format="%.1f")
            g_skip_tautomer = st.checkbox("scrub.py: Skip tautomers", key="g_skip_taut_main_taut")
            g_skip_acidbase = st.checkbox("scrub.py: Skip protomers", key="g_skip_ab_main_ab")

    if ligand_input_method == "SMILES String":
        inchikey_or_smiles_val = st.text_input("InChIKey or SMILES string:", key="smiles_input_main_val_lig")
        use_inchikey = st.checkbox("Input is InChIKey (will fetch SMILES from PubChem)", value=False, key="use_inchikey_main_cb_lig")
        lig_name_base_input = st.text_input("Ligand Base Name:", value="ligand_smiles", key="lig_name_main_name_lig")
        if st.button("Prepare & Add This SMILES Ligand", key="prep_add_smiles_main_btn_lig"):
            _current_batch_processed_details = []
            st.session_state.invalid_smiles_during_standardization = [] # Reset for this batch

            if not inchikey_or_smiles_val.strip():
                st.warning("Please enter a SMILES string or InChIKey.")
            elif not lig_name_base_input.strip():
                st.warning("Please enter a base name for the ligand.")
            elif not scrub_py_ok or not mk_prepare_ligand_py_ok:
                st.error("Ligand preparation scripts (scrub.py/mk_prepare_ligand.py) are not ready.")
            else:
                original_input_smiles = inchikey_or_smiles_val
                actual_smiles_to_process = inchikey_or_smiles_val

                if use_inchikey:
                    with st.spinner(f"Fetching SMILES for InChIKey {inchikey_or_smiles_val}..."):
                        actual_smiles_to_process = get_smiles_from_pubchem_inchikey(inchikey_or_smiles_val)
                    if not actual_smiles_to_process:
                        st.error(f"Could not retrieve SMILES for InChIKey: {inchikey_or_smiles_val}")
                        st.session_state.invalid_smiles_during_standardization.append(original_input_smiles + " (InChIKey fetch failed)")


                if actual_smiles_to_process:
                    st.info(f"Original SMILES for {lig_name_base_input}: {actual_smiles_to_process}")
                    standardized_s = standardize_smiles_rdkit(actual_smiles_to_process, st.session_state.invalid_smiles_during_standardization)
                    if standardized_s:
                        st.info(f"Standardized SMILES for {lig_name_base_input}: {standardized_s}")
                        detail = convert_smiles_to_pdbqt(standardized_s, lig_name_base_input, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                        if detail:
                            detail['id'] = standardized_s # Use standardized SMILES as ID
                            _current_batch_processed_details.append(detail)
                            st.success(f"SMILES ligand '{lig_name_base_input}' prepared: {Path(detail['pdbqt_path']).name}")
                        else:
                            st.error(f"Failed to prepare PDBQT for ligand '{lig_name_base_input}' from standardized SMILES: {standardized_s}")
                    else:
                        st.warning(f"Standardization failed for SMILES '{actual_smiles_to_process}'. Ligand '{lig_name_base_input}' not prepared.")
            
            if _current_batch_processed_details:
                st.session_state.prepared_ligand_details_list.extend(_current_batch_processed_details)
                st.success(f"Added {len(_current_batch_processed_details)} ligand(s). Total ready: {len(st.session_state.prepared_ligand_details_list)}")
            if st.session_state.invalid_smiles_during_standardization:
                 with st.expander(f"{len(st.session_state.invalid_smiles_during_standardization)} SMILES failed standardization/fetch", expanded=True):
                    for failed_smi in st.session_state.invalid_smiles_during_standardization: st.caption(f"- {failed_smi}")
            elif not (not inchikey_or_smiles_val.strip() or not lig_name_base_input.strip() or (not scrub_py_ok or not mk_prepare_ligand_py_ok) or not actual_smiles_to_process) and not _current_batch_processed_details:
                st.warning("No ligands were successfully prepared and added in this step.")


    elif ligand_input_method == "SMILES File (.txt)":
        uploaded_smiles_file = st.file_uploader("Upload SMILES file (.txt, one SMILES per line):", type="txt", key="smiles_uploader_main_file_lig")
        if st.button("Process & Add SMILES File", key="process_add_smiles_file_btn"):
            _current_batch_processed_details = []
            st.session_state.invalid_smiles_during_standardization = [] # Reset for this batch

            if not uploaded_smiles_file:
                st.warning("Please upload a SMILES file first.")
            elif not scrub_py_ok or not mk_prepare_ligand_py_ok:
                st.error("Ligand preparation scripts (scrub.py/mk_prepare_ligand.py) are not ready.")
            else:
                try:
                    smiles_strings_original = uploaded_smiles_file.getvalue().decode("utf-8").splitlines()
                    smiles_strings_original = [s.strip() for s in smiles_strings_original if s.strip()]
                    if not smiles_strings_original:
                        st.warning("SMILES file is empty or contains no valid strings.")
                    else:
                        st.info(f"Found {len(smiles_strings_original)} SMILES string(s) in the file for processing.")
                        for i, original_smi in enumerate(smiles_strings_original):
                            lig_name_base = f"{Path(uploaded_smiles_file.name).stem}_lig{i+1}"
                            st.markdown(f"--- \nProcessing original SMILES: `{original_smi}` as `{lig_name_base}`")
                            
                            standardized_s = standardize_smiles_rdkit(original_smi, st.session_state.invalid_smiles_during_standardization)
                            if standardized_s:
                                st.info(f"Standardized SMILES for {lig_name_base}: {standardized_s}")
                                detail = convert_smiles_to_pdbqt(standardized_s, lig_name_base, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                                if detail:
                                    detail['id'] = standardized_s # Use standardized SMILES as ID
                                    _current_batch_processed_details.append(detail)
                                    st.success(f"SMILES ligand '{lig_name_base}' prepared: {Path(detail['pdbqt_path']).name}")
                                else:
                                    st.error(f"Failed to prepare PDBQT for ligand '{lig_name_base}' from standardized SMILES: {standardized_s}")
                            else:
                                st.warning(f"Standardization failed for original SMILES '{original_smi}'. Ligand '{lig_name_base}' not prepared.")
                except Exception as e:
                    st.error(f"Error reading or processing SMILES file: {e}")
            
            if _current_batch_processed_details:
                st.session_state.prepared_ligand_details_list.extend(_current_batch_processed_details)
                st.success(f"Added {len(_current_batch_processed_details)} ligand(s) from file. Total ready: {len(st.session_state.prepared_ligand_details_list)}")
            if st.session_state.invalid_smiles_during_standardization:
                with st.expander(f"{len(st.session_state.invalid_smiles_during_standardization)} SMILES failed standardization", expanded=True):
                    for failed_smi in st.session_state.invalid_smiles_during_standardization: st.caption(f"- {failed_smi}")
            elif uploaded_smiles_file and not _current_batch_processed_details : 
                st.warning("No ligands were successfully prepared and added from the file.")


    elif ligand_input_method == "PDBQT File(s)":
        uploaded_pdbqt_files = st.file_uploader("Upload PDBQT ligand(s):", type="pdbqt", accept_multiple_files=True, key="pdbqt_uploader_main_file_lig_btn")
        if st.button("Add Uploaded PDBQT(s)", key="add_pdbqt_btn"):
            _current_batch_processed_details = []
            if not uploaded_pdbqt_files:
                st.warning("No PDBQT files uploaded to add.")
            else:
                for up_file in uploaded_pdbqt_files:
                    dest_path = LIGAND_PREP_DIR_LOCAL / up_file.name 
                    with open(dest_path, "wb") as f: f.write(up_file.getbuffer())
                    _current_batch_processed_details.append({"id": up_file.name, "pdbqt_path": str(dest_path), "base_name": Path(up_file.name).stem})
            
            if _current_batch_processed_details:
                st.session_state.prepared_ligand_details_list.extend(_current_batch_processed_details)
                st.success(f"Added {len(_current_batch_processed_details)} PDBQT file(s). Total ready: {len(st.session_state.prepared_ligand_details_list)}")
            elif uploaded_pdbqt_files:
                st.warning("No PDBQT files were added in this step (already added or none selected).")

    elif ligand_input_method == "Other Ligand File(s)":
        uploaded_other_files = st.file_uploader("Upload other ligand file(s) (e.g., SDF, MOL2, PDB):", accept_multiple_files=True, key="other_lig_uploader_main_file_lig_btn", help="Files will be converted to PDBQT.")
        if st.button("Process & Add Other Ligand Files", key="process_add_other_files_btn"):
            _current_batch_processed_details = []
            if not uploaded_other_files:
                st.warning("No files uploaded to process.")
            elif not mk_prepare_ligand_py_ok:
                st.error("Ligand preparation script (mk_prepare_ligand.py) is not ready.")
            else:
                for up_file in uploaded_other_files:
                    st.markdown(f"--- \nProcessing file: `{up_file.name}`")
                    temp_ligand_path = LIGAND_UPLOAD_TEMP_DIR / up_file.name 
                    with open(temp_ligand_path, "wb") as f:
                        f.write(up_file.getbuffer())
                    detail = convert_ligand_file_to_pdbqt(str(temp_ligand_path), up_file.name, LIGAND_PREP_DIR_LOCAL, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                    if detail:
                        _current_batch_processed_details.append(detail)
                        st.success(f"Ligand '{up_file.name}' converted to PDBQT: {Path(detail['pdbqt_path']).name}")
                    else:
                        st.error(f"Failed to convert ligand file: {up_file.name}")
                    if temp_ligand_path.exists(): temp_ligand_path.unlink(missing_ok=True) 
            
            if _current_batch_processed_details:
                st.session_state.prepared_ligand_details_list.extend(_current_batch_processed_details)
                st.success(f"Added {len(_current_batch_processed_details)} converted ligand(s). Total ready: {len(st.session_state.prepared_ligand_details_list)}")
            elif uploaded_other_files:
                st.warning("No ligands were successfully prepared and added from the uploaded files.")

    elif ligand_input_method == "ZIP Archive":
        uploaded_zip_file = st.file_uploader("Upload ZIP archive of ligand files (PDBQT, SDF, MOL2, PDB):", type="zip", key="zip_uploader_main_file_lig_btn")
        if st.button("Process & Add ZIP Archive", key="process_add_zip_archive_btn"):
            _current_batch_processed_details = []
            if not uploaded_zip_file:
                st.warning("No ZIP archive uploaded.")
            else:
                if ZIP_EXTRACT_DIR_LOCAL.exists(): 
                    for item in ZIP_EXTRACT_DIR_LOCAL.iterdir():
                        if item.is_file(): item.unlink()
                        elif item.is_dir(): shutil.rmtree(item)
                ZIP_EXTRACT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

                zip_file_path = LIGAND_UPLOAD_TEMP_DIR / uploaded_zip_file.name
                with open(zip_file_path, "wb") as f:
                    f.write(uploaded_zip_file.getbuffer())
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(ZIP_EXTRACT_DIR_LOCAL)
                    st.info(f"Extracted files from '{uploaded_zip_file.name}'. Processing...")

                    for extracted_item_path in ZIP_EXTRACT_DIR_LOCAL.rglob('*'): 
                        if extracted_item_path.is_file():
                            original_filename = extracted_item_path.name
                            st.markdown(f"--- \nProcessing extracted file: `{original_filename}`")
                            if original_filename.lower().endswith(".pdbqt"):
                                dest_path = LIGAND_PREP_DIR_LOCAL / original_filename
                                shutil.copy(extracted_item_path, dest_path)
                                _current_batch_processed_details.append({"id": original_filename, "pdbqt_path": str(dest_path), "base_name": Path(original_filename).stem})
                                st.success(f"PDBQT ligand '{original_filename}' added directly.")
                            elif mk_prepare_ligand_py_ok: 
                                detail = convert_ligand_file_to_pdbqt(str(extracted_item_path), original_filename, LIGAND_PREP_DIR_LOCAL, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                                if detail:
                                    _current_batch_processed_details.append(detail)
                                    st.success(f"Ligand '{original_filename}' converted: {Path(detail['pdbqt_path']).name}")
                                else:
                                    st.error(f"Failed to convert extracted ligand: {original_filename}")
                            else: 
                                st.warning(f"Skipping '{original_filename}': Not a PDBQT and mk_prepare_ligand.py is not ready.")
                except zipfile.BadZipFile: st.error("Uploaded file is not a valid ZIP archive.")
                except Exception as e: st.error(f"Error processing ZIP archive: {e}")
                finally:
                    if zip_file_path.exists(): zip_file_path.unlink(missing_ok=True)
            
            if _current_batch_processed_details:
                st.session_state.prepared_ligand_details_list.extend(_current_batch_processed_details)
                st.success(f"Added {len(_current_batch_processed_details)} ligand(s) from ZIP. Total ready: {len(st.session_state.prepared_ligand_details_list)}")
            elif uploaded_zip_file:
                st.warning("No ligands were successfully prepared and added from the ZIP archive.")

    if st.session_state.get('prepared_ligand_details_list', []):
        exp_lig = st.expander(f"**{len(st.session_state.prepared_ligand_details_list)} Ligand(s) Ready for Docking**", expanded=True)
        for i, lig_info in enumerate(st.session_state.prepared_ligand_details_list):
            exp_lig.caption(f"{i+1}. ID (SMILES/Filename): {lig_info.get('id', 'N/A')}, Base: {lig_info.get('base_name', 'N/A')}, File: `{Path(lig_info.get('pdbqt_path', '')).name}`")
    st.markdown("---")

    st.subheader("🚀 Docking Execution")
    
    current_receptors_for_ui = st.session_state.get('fetched_receptor_paths', [])
    current_configs_for_ui = st.session_state.get('fetched_config_paths', [])
    current_ligands_for_ui = st.session_state.get('prepared_ligand_details_list', [])

    show_perl_checkbox = len(current_ligands_for_ui) >= 1 and vina_screening_pl_ok and VINA_SCREENING_PL_LOCAL_PATH.exists()
    use_perl_default = True 
    if show_perl_checkbox:
        st.checkbox("Use `Vina_screening.pl` (Strict Protein-Config Pairing)?", value=use_perl_default, key="use_perl_dockpage_main_cb_persistent")
    
    if st.button("Start Docking Run", key="start_docking_main_btn_main_page", type="primary"):
        final_ligand_details_list_for_run = list(st.session_state.get('prepared_ligand_details_list', []))
        current_receptors_for_run = st.session_state.get('fetched_receptor_paths', [])
        current_configs_for_run = st.session_state.get('fetched_config_paths', [])

        if not vina_ready: 
            st.error("Vina executable not set up. Cannot start docking.")
        elif not current_receptors_for_run: 
            st.warning("No receptors available for docking. Please fetch receptors.")
        elif not final_ligand_details_list_for_run: 
            st.warning("No ligands prepared and added for docking. Please prepare ligands.")
        elif not current_configs_for_run: 
            st.warning("No Vina configs available for docking. Please fetch configs.")
        else:
            st.info(f"Starting docking for {len(final_ligand_details_list_for_run)} ligand(s) vs {len(current_receptors_for_run)} receptor(s).")
            st.session_state.docking_run_outputs = [] 
            DOCKING_OUTPUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

            use_vina_screening_perl_for_run_actual = False
            if show_perl_checkbox: 
                use_vina_screening_perl_for_run_actual = st.session_state.get("use_perl_dockpage_main_cb_persistent", use_perl_default)
            
            if use_vina_screening_perl_for_run_actual:
                st.markdown("##### Docking via `Vina_screening.pl` (Strict Protein-Config Pairing)")
                if not (vina_screening_pl_ok and VINA_SCREENING_PL_LOCAL_PATH.exists() and os.access(VINA_SCREENING_PL_LOCAL_PATH, os.X_OK)):
                    st.error(f"Local `Vina_screening.pl` script not available or not executable at {VINA_SCREENING_PL_LOCAL_PATH}.")
                else:
                    ligand_list_file_for_perl = WORKSPACE_PARENT_DIR / "ligands_for_perl.txt"
                    with open(ligand_list_file_for_perl, "w") as f_list:
                        for lig_detail in final_ligand_details_list_for_run:
                            f_list.write(str(Path(lig_detail['pdbqt_path']).resolve()) + "\n")
                    #st.caption(f"Perl ligand list file created: '{ligand_list_file_for_perl.name}'")

                    overall_docking_progress = st.progress(0)
                    receptors_processed_count = 0; skipped_receptor_count = 0
                    for i_rec, receptor_path_str in enumerate(current_receptors_for_run):
                        receptor_file = Path(receptor_path_str); protein_base = receptor_file.stem
                        st.markdown(f"--- \n**Receptor: `{receptor_file.name}` (Perl Mode)**")
                        config_to_use = None 
                        if not current_configs_for_run: st.error(f"No Vina configs for {receptor_file.name}."); skipped_receptor_count +=1; overall_docking_progress.progress((i_rec + 1) / len(current_receptors_for_run)); continue
                        elif len(current_configs_for_run) == 1: config_to_use = Path(current_configs_for_run[0])
                        else: config_to_use = find_paired_config_for_protein(protein_base, current_configs_for_run)

                        if not config_to_use: st.warning(f"No paired config for `{receptor_file.name}`. Skipping."); skipped_receptor_count +=1; overall_docking_progress.progress((i_rec + 1) / len(current_receptors_for_run)); continue
                        #st.caption(f"Using config: `{config_to_use.name}`")

                        temp_receptor_path_for_perl = WORKSPACE_PARENT_DIR / receptor_file.name
                        shutil.copy(receptor_file, temp_receptor_path_for_perl)
                        
                        cmd_perl = ["perl", str(VINA_SCREENING_PL_LOCAL_PATH.resolve()),
                                    str(VINA_PATH_LOCAL.resolve()), 
                                    str(temp_receptor_path_for_perl.name), 
                                    str(config_to_use.resolve()), 
                                    protein_base]
                        try:
                            path_to_ligand_list_for_perl_stdin = str(ligand_list_file_for_perl.resolve()) + "\n"
                            #st.caption(f"Path being passed to Perl STDIN: {path_to_ligand_list_for_perl_stdin.strip()}")
                            
                            proc = subprocess.run(cmd_perl, 
                                                  input=path_to_ligand_list_for_perl_stdin, 
                                                  capture_output=True, 
                                                  text=True, 
                                                  check=False, 
                                                  cwd=str(WORKSPACE_PARENT_DIR.resolve()))
                            
                            return_code_perl = proc.returncode
                            stdout_p = proc.stdout
                            stderr_p = proc.stderr
                            
                            #st.info(f"Perl script for `{protein_base}` completed with RC: {return_code_perl}.")
                            if stdout_p.strip():
                                with st.expander(f"Perl STDOUT for {protein_base}", expanded=False): st.text(stdout_p)
                            # if stderr_p.strip(): 
                            #     st.warning(f"Perl script for `{protein_base}` produced STDERR (RC: {return_code_perl}):")
                            #     with st.expander(f"Perl STDERR for {protein_base}", expanded=True): st.text(stderr_p)
                            
                            if return_code_perl != 0: 
                                st.error(f"Perl script execution failed for `{protein_base}` (RC: {return_code_perl}). Review STDOUT/STDERR above for details.")
                            
                            perl_protein_out_dir = WORKSPACE_PARENT_DIR / protein_base 
                            if perl_protein_out_dir.is_dir():
                                #st.info(f"Perl: Output directory for {protein_base} found: {perl_protein_out_dir}")
                                for lig_detail in final_ligand_details_list_for_run:
                                    score = None
                                    possible_pdbqt_names = [
                                        f"{lig_detail['base_name']}-{protein_base}_out.pdbqt", 
                                        f"{lig_detail['base_name']}_{protein_base}_out.pdbqt", 
                                        f"{lig_detail['base_name']}_out.pdbqt",             
                                        f"{Path(lig_detail['pdbqt_path']).stem}_{protein_base}_out.pdbqt" 
                                    ]
                                    
                                    expected_pdbqt_file_found_path = None
                                    for pdbqt_name_pattern in possible_pdbqt_names:
                                        current_expected_pdbqt_file = perl_protein_out_dir / pdbqt_name_pattern
                                        if current_expected_pdbqt_file.exists():
                                            #st.caption(f"Perl: Found PDBQT '{current_expected_pdbqt_file.name}', attempting to parse.")
                                            score = parse_score_from_pdbqt(str(current_expected_pdbqt_file))
                                            if score is not None:
                                                expected_pdbqt_file_found_path = current_expected_pdbqt_file
                                                break 
                                    
                                    if score is not None and expected_pdbqt_file_found_path:
                                        #st.info(f"Perl: Score {score} for '{lig_detail['base_name']}' (protein '{protein_base}') from PDBQT '{expected_pdbqt_file_found_path.name}'.")
                                        st.session_state.docking_run_outputs.append({
                                            "ligand_id": lig_detail["id"],
                                            "ligand_base_name": lig_detail["base_name"],
                                            "protein_stem": protein_base,
                                            "config_stem": config_to_use.stem, 
                                            "score": score
                                        })
                                    else:
                                        st.warning(f"Perl: Score not obtained for '{lig_detail['base_name']}' with '{protein_base}'. Searched in '{perl_protein_out_dir}'.")
                            elif return_code_perl == 0 : 
                                st.warning(f"Perl output dir NOT found: {perl_protein_out_dir}, though script RC=0 for {protein_base}. Check Perl script's output behavior.")

                        except Exception as e_p: 
                            st.error(f"Error during Perl script processing for `{protein_base}`: {type(e_p).__name__} - {e_p}")
                        finally:
                            if temp_receptor_path_for_perl.exists(): 
                                temp_receptor_path_for_perl.unlink(missing_ok=True)
                        
                        receptors_processed_count += 1
                        overall_docking_progress.progress((receptors_processed_count + skipped_receptor_count) / len(current_receptors_for_run))
                    
                    if ligand_list_file_for_perl.exists(): 
                        ligand_list_file_for_perl.unlink(missing_ok=True)
                    if skipped_receptor_count > 0: 
                        st.warning(f"{skipped_receptor_count} receptor(s) skipped in Perl mode.")

            else: # Direct Vina calls
                st.markdown("##### Docking via Direct Vina Calls (Strict Protein-Config Pairing)")
                planned_docking_jobs = []
                skipped_receptors_direct_mode = set()
                for rec_path_str in current_receptors_for_run:
                    receptor_file = Path(rec_path_str); protein_base = receptor_file.stem
                    paired_config_file = None
                    if not current_configs_for_run:
                        if protein_base not in skipped_receptors_direct_mode: st.warning(f"No configs for '{receptor_file.name}'. Skip."); skipped_receptors_direct_mode.add(protein_base)
                        continue
                    elif len(current_configs_for_run) == 1: paired_config_file = Path(current_configs_for_run[0])
                    else: paired_config_file = find_paired_config_for_protein(protein_base, current_configs_for_run)

                    if paired_config_file:
                        for lig_detail in final_ligand_details_list_for_run:
                            planned_docking_jobs.append({"ligand_info": lig_detail, "receptor": receptor_file, "config": paired_config_file})
                    elif protein_base not in skipped_receptors_direct_mode: st.warning(f"No matching config for '{receptor_file.name}'. Skip."); skipped_receptors_direct_mode.add(protein_base)
                
                num_total_direct_jobs = len(planned_docking_jobs)
                if num_total_direct_jobs > 0:
                    job_counter = 0; overall_docking_progress = st.progress(0)
                    for job_spec in planned_docking_jobs:
                        lig_info, receptor_file, config_file = job_spec["ligand_info"], job_spec["receptor"], job_spec["config"]
                        ligand_file = Path(lig_info["pdbqt_path"]) 
                        job_counter += 1
                        st.markdown(f"--- \n**Job {job_counter}/{num_total_direct_jobs}:** `{receptor_file.name}` + `{ligand_file.name}` (Config: `{config_file.name}`)")

                        output_base_name = f"{lig_info['base_name']}_{receptor_file.stem}_{config_file.stem}"
                        output_pdbqt_docked = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base_name}_out.pdbqt"
                        
                        cmd_vina = [str(VINA_PATH_LOCAL.resolve()), 
                                    "--receptor", str(receptor_file.resolve()),
                                    "--ligand", str(ligand_file.resolve()), 
                                    "--config", str(config_file.resolve()),
                                    "--out", str(output_pdbqt_docked.resolve())]
                        try:
                            for f_path_str, f_name in [(str(receptor_file.resolve()), "Receptor"), (str(ligand_file.resolve()), "Ligand"), (str(config_file.resolve()), "Config")]:
                                if not Path(f_path_str).exists():
                                    st.error(f"Vina input file MISSING: {f_name} at {f_path_str}")
                                elif Path(f_path_str).stat().st_size == 0:
                                    st.warning(f"Vina input file IS EMPTY: {f_name} at {f_path_str}")

                            res_vina = subprocess.run(cmd_vina, capture_output=True, text=True, check=True, cwd=str(WORKSPACE_PARENT_DIR.resolve()))
                            
                            st.success(f"Vina job OK for {output_base_name}!")
                            if res_vina.stdout.strip():
                                with st.expander(f"Vina STDOUT for {output_base_name}", expanded=False): st.text(res_vina.stdout)
                            if res_vina.stderr.strip(): 
                                with st.expander(f"Vina STDERR for {output_base_name} (Warnings/Info)", expanded=False): 
                                    st.text(res_vina.stderr)

                            if output_pdbqt_docked.exists() and output_pdbqt_docked.stat().st_size > 0:
                                score = parse_score_from_pdbqt(str(output_pdbqt_docked.resolve()))
                                if score is not None:
                                    st.session_state.docking_run_outputs.append({
                                        "ligand_id": lig_info["id"],
                                        "ligand_base_name": lig_info["base_name"],
                                        "protein_stem": receptor_file.stem,
                                        "config_stem": config_file.stem,
                                        "score": score
                                    })
                                else:
                                    st.warning(f"Score was None after parsing {output_pdbqt_docked.name}.")
                            else:
                                st.error(f"Vina output file MISSING or EMPTY: {output_pdbqt_docked.name} after Vina run.")

                            if output_pdbqt_docked.exists() and output_pdbqt_docked.stat().st_size > 0: 
                                 with open(output_pdbqt_docked, "rb") as fp: st.download_button(f"DL Docked PDBQT ({output_base_name})", fp, output_pdbqt_docked.name, "application/octet-stream", key=f"dl_pdbqt_{job_counter}_direct")

                        except subprocess.CalledProcessError as e_vina:
                            st.error(f"VINA JOB FAILED for {output_base_name} (RC: {e_vina.returncode}).")
                            with st.expander(f"Vina Error Details for {output_base_name}", expanded=True):
                                st.error(f"Cmd: `{' '.join(e_vina.cmd)}`"); st.text("STDOUT:\n" + (e_vina.stdout.strip() or "No STDOUT.")); st.text("STDERR:\n" + (e_vina.stderr.strip() or "No STDERR."))
                        except FileNotFoundError as e_fnf: 
                            st.error(f"FILE NOT FOUND error during Vina subprocess call for {output_base_name}: {e_fnf}. Check Vina path: {VINA_PATH_LOCAL.resolve()}")
                        except Exception as e_gen:
                            st.error(f"Unexpected error in Vina job for {output_base_name}: {e_gen}")
                        if num_total_direct_jobs > 0: overall_docking_progress.progress(job_counter / num_total_direct_jobs)
                else: 
                    st.info("No direct Vina jobs were planned.")

            if st.session_state.docking_run_outputs: 
                st.markdown("---"); st.subheader("📊 Docking Results Summary")
                try:
                    df_flat = pd.DataFrame(st.session_state.docking_run_outputs)
                    if df_flat.empty: 
                        st.info("No docking scores were recorded (DataFrame is empty).")
                    else:
                        df_flat['score'] = pd.to_numeric(df_flat['score'], errors='coerce')
                        if df_flat['score'].isnull().any():
                            st.warning("Some scores could not be converted to numeric (NaN). These rows will show 'N/A'.")
                        df_flat['Protein-Config'] = df_flat['protein_stem'] + '_' + df_flat['config_stem']
                        df_pivot = df_flat.pivot_table(index=['ligand_id', 'ligand_base_name'], columns='Protein-Config', values='score', aggfunc='min')
                        df_summary = df_pivot.reset_index()
                        new_column_names = {'ligand_id': 'Ligand ID / SMILES', 'ligand_base_name': 'Ligand Base Name'}
                        for col in df_pivot.columns: new_column_names[col] = f"{col} Score (kcal/mol)"
                        df_summary = df_summary.rename(columns=new_column_names)
                        for col_name in df_summary.columns:
                            if col_name.endswith("Score (kcal/mol)"):
                                df_summary[col_name] = df_summary[col_name].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                        st.dataframe(df_summary)
                        csv_summary = convert_df_to_csv(df_summary)
                        st.download_button("Download Summary (CSV)", csv_summary, "docking_summary_per_ligand.csv", "text/csv", key="dl_summary_per_ligand_csv")
                except Exception as e_df:
                    st.error(f"Error generating results summary table: {e_df}")
                    st.caption("Raw results data (first 5 if available):"); st.json(st.session_state.docking_run_outputs[:5])
            else: 
                st.info("No docking outputs were recorded to summarize.") 
            st.balloons()
            st.header("🏁 Docking Run Finished 🏁")
            st.caption(f"Docked PDBQTs (direct Vina) in `{DOCKING_OUTPUT_DIR_LOCAL.name}/`. Perl script outputs in `{WORKSPACE_PARENT_DIR.name}/<protein_base_name>/`.")

def display_about_page():
    st.header("About This Application")
    st.markdown(f"**Ensemble AutoDock Vina App - v{APP_VERSION}**")
    st.markdown("""
    This application facilitates molecular docking simulations using AutoDock Vina,
    allowing for ensemble docking approaches.
    **Features:**
    - Preparation of ligands from SMILES strings (with RDKit standardization) or various file formats.
    - Docking against one or multiple receptor structures.
    - Utilization of specific or multiple Vina configuration files.
    - Options for using a Perl-based screening script or direct Vina calls.
    - Summarization of best docking scores per ligand.
    **Repository Structure Example:**
    - `your_app_root/`
        - `streamlit_app.py` (this file)
        - `ensemble_docking/`
            - `ligand_preprocessing/scrub.py`
            - `ligand_preprocessing/mk_prepare_ligand.py`
            - `Vina_screening.pl`
        - `vina/vina_1.2.5_linux_x86_64` (Vina executable, `chmod +x`)
        - `autodock_workspace/` (created for temporary files, fetched assets)
        - `autodock_outputs/` (created for PDBQT outputs from direct Vina calls)
        - `requirements.txt` (must include `rdkit-pypi` if using RDKit standardization)
        - `packages.txt` (e.g., for Streamlit Community Cloud: `perl`)
    """)
    st.markdown(f"**Key Local Paths Used (resolved):**\n"
                f"- App Root: `{APP_ROOT.resolve()}`\n"
                f"- Workspace Parent: `{WORKSPACE_PARENT_DIR.resolve()}`\n"
                f"- Vina Executable: `{VINA_PATH_LOCAL.resolve()}`\n"
                f"- Direct Vina Output PDBQTs: `{DOCKING_OUTPUT_DIR_LOCAL.resolve()}`")

def main():
    st.set_page_config(layout="wide", page_title=f"Ensemble Vina Docking v{APP_VERSION}")
    st.sidebar.image("https://raw.githubusercontent.com/HenryChritopher02/bace1/main/logo.png", width=100)
    st.sidebar.title("Docking Suite")

    app_mode_options = ("Ensemble Docking", "About")
    app_mode_default = app_mode_options[0] 
    app_mode = st.sidebar.radio(
        "Select Procedure:", app_mode_options,
        index=app_mode_options.index(st.session_state.get("app_mode_select", app_mode_default)), 
        key="app_mode_selector_main_radio", 
    )
    st.session_state.app_mode_select = app_mode 
    st.sidebar.markdown("---")

    if app_mode == "Ensemble Docking":
        display_ensemble_docking_procedure()
    elif app_mode == "About":
        display_about_page()

if __name__ == "__main__":
    main()
