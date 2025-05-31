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

# --- Configuration ---
APP_VERSION = "1.2.5" # Cleaned UI, Dual Fetch Options

# GitHub URL for RECEPTORS and CONFIGS (if they are still remote)
BASE_GITHUB_URL_FOR_DATA = "https://raw.githubusercontent.com/HenryChritopher02/bace1/main/ensemble-docking/"
GH_API_BASE_URL = "https://api.github.com/repos/"
GH_OWNER = "HenryChritopher02"
GH_REPO = "bace1"
GH_BRANCH = "main"
GH_ENSEMBLE_DOCKING_ROOT_PATH = "ensemble-docking" # Path within the repo to the root of these files
RECEPTOR_SUBDIR_GH = "ensemble_protein/" # Relative to GH_ENSEMBLE_DOCKING_ROOT_PATH on GitHub for API listing
CONFIG_SUBDIR_GH = "config/"             # Relative to GH_ENSEMBLE_DOCKING_ROOT_PATH on GitHub for API listing

# --- LOCAL Paths for Helper Scripts and Vina ---
APP_ROOT = Path(".")
ENSEMBLE_DOCKING_DIR_LOCAL = APP_ROOT / "ensemble_docking" # This folder is IN your app repo
LIGAND_PREPROCESSING_SUBDIR_LOCAL = ENSEMBLE_DOCKING_DIR_LOCAL / "ligand_preprocessing"
SCRUB_PY_LOCAL_PATH = LIGAND_PREPROCESSING_SUBDIR_LOCAL / "scrub.py"
MK_PREPARE_LIGAND_PY_LOCAL_PATH = LIGAND_PREPROCESSING_SUBDIR_LOCAL / "mk_prepare_ligand.py"
VINA_SCREENING_PL_LOCAL_PATH = ENSEMBLE_DOCKING_DIR_LOCAL / "Vina_screening.pl"

VINA_DIR_LOCAL = APP_ROOT / "vina" # This folder is IN your app repo
VINA_EXECUTABLE_NAME = "vina_1.2.5_linux_x86_64"
VINA_PATH_LOCAL = VINA_DIR_LOCAL / VINA_EXECUTABLE_NAME

# Local workspace directories (will be created under the app root)
WORKSPACE_PARENT_DIR = APP_ROOT / "autodock_workspace"
RECEPTOR_DIR_LOCAL = WORKSPACE_PARENT_DIR / "fetched_receptors" # For downloaded remote receptors
CONFIG_DIR_LOCAL = WORKSPACE_PARENT_DIR / "fetched_configs"     # For downloaded remote configs
LIGAND_PREP_DIR_LOCAL = WORKSPACE_PARENT_DIR / "prepared_ligands"
LIGAND_UPLOAD_TEMP_DIR = WORKSPACE_PARENT_DIR / "uploaded_ligands_temp"
ZIP_EXTRACT_DIR_LOCAL = WORKSPACE_PARENT_DIR / "zip_extracted_ligands"
DOCKING_OUTPUT_DIR_LOCAL = APP_ROOT / "autodock_outputs"

# --- Helper Functions ---

def list_files_from_github_repo_dir(owner: str, repo: str, dir_path_in_repo: str, branch: str, file_extension: str = None) -> list[str]:
    """Lists files in a GitHub repository directory using the GitHub API."""
    api_url = f"{GH_API_BASE_URL}{owner}/{repo}/contents/{dir_path_in_repo}?ref={branch}"
    # st.sidebar.caption(f"API URL: {api_url}") # Debug
    filenames = []
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        contents = response.json()
        if not isinstance(contents, list):
            st.sidebar.error(f"API Error for {dir_path_in_repo}: Expected list, got {type(contents)}")
            if isinstance(contents, dict) and 'message' in contents: st.sidebar.error(f"GitHub: {contents['message']}")
            return []
        for item in contents:
            if item.get('type') == 'file':
                if file_extension:
                    if item.get('name', '').lower().endswith(file_extension.lower()):
                        filenames.append(item['name'])
                else:
                    filenames.append(item['name'])
        if not filenames and file_extension: # Only warn if a filter was applied and nothing found
             st.sidebar.warning(f"No files matching '{file_extension}' found in '{dir_path_in_repo}'.")
    except requests.exceptions.Timeout:
        st.sidebar.error(f"Timeout listing files from GitHub: {dir_path_in_repo}")
    except requests.exceptions.HTTPError as e:
        st.sidebar.error(f"HTTP error listing files from {dir_path_in_repo}: {e.status_code}")
        if e.response is not None: st.sidebar.caption(f"Detail: {e.response.text[:100]}")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Request error listing files from {dir_path_in_repo}: {e}")
    except ValueError as e: 
        st.sidebar.error(f"JSON decode error for {dir_path_in_repo}: {e}")
    return filenames

def initialize_directories():
    """Creates necessary local workspace and output directories."""
    dirs_to_create = [
        WORKSPACE_PARENT_DIR, RECEPTOR_DIR_LOCAL, CONFIG_DIR_LOCAL,
        LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR,
        ZIP_EXTRACT_DIR_LOCAL, DOCKING_OUTPUT_DIR_LOCAL
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

def download_file_from_github(raw_download_base_url, relative_path_segment, local_filename, local_save_dir):
    """Downloads a file using the raw GitHub content URL."""
    # raw_download_base_url is like https://raw.githubusercontent.com/.../ensemble-docking/
    # relative_path_segment is like ensemble_protein/receptor.pdbqt or config/conf.txt
    full_url = urljoin(raw_download_base_url, relative_path_segment)
    local_file_path = Path(local_save_dir) / local_filename
    try:
        # Spinner context removed from here to be managed by the calling function for batch operations
        response = requests.get(full_url, stream=True, timeout=15)
        response.raise_for_status()
        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # st.sidebar.success(f"Downloaded: {local_filename}") # Too verbose for batch
        return str(local_file_path)
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error downloading {local_filename} from {full_url}: {e}")
        return None

def make_file_executable(filepath_str):
    """Makes a file executable."""
    if not filepath_str or not os.path.exists(filepath_str):
        # st.sidebar.warning(f"Cannot make executable: File not found at {filepath_str}") # Too verbose
        return False
    try:
        current_mode = os.stat(filepath_str).st_mode
        os.chmod(filepath_str, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return True
    except Exception: # Keep silent on success, only show error
        st.sidebar.error(f"Error making {Path(filepath_str).name} executable.")
        return False

def check_script_exists(script_path: Path, script_name: str, is_critical: bool = True):
    """Checks if a local script file exists. Shows error if critical and not found."""
    if script_path.exists() and script_path.is_file():
        return True
    else:
        if is_critical:
            st.sidebar.error(f"CRITICAL: `{script_name}` NOT FOUND at `{script_path}`. App may not function.")
        else:
            st.sidebar.warning(f"Optional script `{script_name}` not found at `{script_path}`.")
        return False

def check_vina_binary(show_success=False):
    """Checks for Vina binary and its permissions."""
    if not VINA_PATH_LOCAL.exists():
        st.sidebar.error(
            f"Vina exe NOT FOUND at `{VINA_PATH_LOCAL}`. "
            f"Ensure `{VINA_EXECUTABLE_NAME}` is in `{VINA_DIR_LOCAL}` folder in your app repo."
        )
        return False
    if show_success: st.sidebar.success(f"Vina binary found: `{VINA_PATH_LOCAL.name}`")

    is_executable = os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK)
    if is_executable:
        if show_success: st.sidebar.success("Vina binary is executable.")
        return True
    else:
        st.sidebar.warning("Vina binary is NOT executable. Attempting to set permission...")
        if make_file_executable(str(VINA_PATH_LOCAL)):
            if os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK):
                st.sidebar.success("Execute permission set for Vina.")
                return True
        st.sidebar.error("Failed to make Vina executable.")
        st.sidebar.markdown(f"**Action:** `git add --chmod=+x {VINA_DIR_LOCAL.name}/{VINA_EXECUTABLE_NAME}` in your repo.")
        return False

def get_smiles_from_pubchem_inchikey(inchikey_str):
    api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey_str}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except requests.exceptions.RequestException as e:
        st.warning(f"PubChem API error for InChIKey {inchikey_str}: {e}")
    except (KeyError, IndexError, TypeError) as e: # Added TypeError for non-JSON response
        st.warning(f"Could not parse SMILES from PubChem response for {inchikey_str}: {e}")
    return None

def run_ligand_prep_script(script_local_path_str, script_args, process_name, ligand_name_for_log):
    if not script_local_path_str:
        st.error(f"{process_name}: Script path is not defined.")
        return False
    absolute_script_path = str(Path(script_local_path_str).resolve())
    if not os.path.exists(absolute_script_path):
        st.error(f"{process_name} script not found at {absolute_script_path} (Original: {script_local_path_str})")
        return False
    command = [sys.executable, absolute_script_path] + [str(arg) for arg in script_args] # Ensure all args are strings
    cwd_path_resolved = str(WORKSPACE_PARENT_DIR.resolve())
    if not os.path.exists(cwd_path_resolved):
        st.error(f"Working directory {cwd_path_resolved} for {process_name} does not exist.")
        return False
    try:
        st.info(f"Running {process_name} for {ligand_name_for_log}...")
        # st.caption(f"Cmd: {command[0]} ... CWD: {cwd_path_resolved}") # Concise debug
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd_path_resolved)
        with st.expander(f"{process_name} STDOUT for {ligand_name_for_log}", expanded=False):
            st.text(result.stdout if result.stdout.strip() else "No standard output.")
        if result.stderr.strip(): 
             with st.expander(f"{process_name} STDERR for {ligand_name_for_log} (check for warnings/errors)", expanded=True):
                st.text(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error during {process_name} for {ligand_name_for_log} (Return Code: {e.returncode}):")
        with st.expander(f"{process_name} Details (on error)", expanded=True):
            st.error(f"Command: `{' '.join(e.cmd)}`")
            st.text("STDOUT:\n" + (e.stdout if e.stdout.strip() else "No standard output."))
            st.text("STDERR:\n" + (e.stderr if e.stderr.strip() else "No standard error output."))
        return False
    except Exception as e:
        st.error(f"Unexpected error running {process_name} for {ligand_name_for_log}: {e}")
        return False

def convert_smiles_to_pdbqt(smiles_str, ligand_name_base, output_dir_path_for_final_pdbqt, ph_val, skip_taut, skip_acidbase, local_scrub_script_path, local_mk_prepare_script_path):
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)
    relative_sdf_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}_scrubbed.sdf"
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"
    absolute_pdbqt_path = WORKSPACE_PARENT_DIR / relative_pdbqt_filename
    scrub_options = ["--ph", str(ph_val)]
    if skip_taut: scrub_options.append("--skip_tautomer")
    if skip_acidbase: scrub_options.append("--skip_acidbase")
    scrub_args = [smiles_str, "-o", str(relative_sdf_filename)] + scrub_options
    if not run_ligand_prep_script(str(local_scrub_script_path), scrub_args, "scrub.py", ligand_name_base):
        return None
    mk_prepare_args = ["-i", str(relative_sdf_filename), "-o", str(relative_pdbqt_filename)]
    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base):
        return None
    return str(absolute_pdbqt_path) if absolute_pdbqt_path.exists() else None

def convert_ligand_file_to_pdbqt(input_ligand_file_path_absolute, ligand_name_base, output_dir_path_for_final_pdbqt, local_mk_prepare_script_path):
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"
    absolute_pdbqt_path = WORKSPACE_PARENT_DIR / relative_pdbqt_filename
    mk_prepare_args = ["-i", str(Path(input_ligand_file_path_absolute).resolve()), 
                       "-o", str(relative_pdbqt_filename)]
    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base):
        return None
    return str(absolute_pdbqt_path) if absolute_pdbqt_path.exists() else None

def find_paired_config_for_protein(protein_base_name: str, all_config_paths: list[str]) -> Path | None:
    if not all_config_paths: return None
    patterns_to_try = [
        f"{protein_base_name}.txt", f"config_{protein_base_name}.txt", f"{protein_base_name}_config.txt",
    ]
    for pattern in patterns_to_try:
        for cfg_path_str in all_config_paths:
            cfg_file = Path(cfg_path_str)
            if cfg_file.name.lower() == pattern.lower(): return cfg_file
    for cfg_path_str in all_config_paths: # Fallback substring match
        cfg_file = Path(cfg_path_str)
        if cfg_file.suffix.lower() == ".txt" and protein_base_name.lower() in cfg_file.stem.lower():
            if "config" in cfg_file.stem.lower() or cfg_file.stem.lower() == protein_base_name.lower() : return cfg_file 
    return None

# --- Main Application ---
st.set_page_config(page_title="Ensemble AutoDock Vina", layout="wide")
st.title(f"🧬 Ensemble AutoDock Vina Docking App (v{APP_VERSION})")

initialize_directories()

# --- Sidebar for Setup & Global Parameters ---
st.sidebar.header("🔧 Setup")

# --- Core Components Status (Concise) ---
# These checks are important for app functionality but keep UI clean on success.
st.sidebar.markdown("---")
st.sidebar.caption("Core Components:")
scrub_py_ok = check_script_exists(SCRUB_PY_LOCAL_PATH, "scrub.py")
if scrub_py_ok: make_file_executable(str(SCRUB_PY_LOCAL_PATH))

mk_prepare_ligand_py_ok = check_script_exists(MK_PREPARE_LIGAND_PY_LOCAL_PATH, "mk_prepare_ligand.py")
if mk_prepare_ligand_py_ok: make_file_executable(str(MK_PREPARE_LIGAND_PY_LOCAL_PATH))

vina_screening_pl_ok = check_script_exists(VINA_SCREENING_PL_LOCAL_PATH, "Vina_screening.pl", is_critical=False)
if vina_screening_pl_ok: make_file_executable(str(VINA_SCREENING_PL_LOCAL_PATH))

vina_ready = check_vina_binary(show_success=False) # Only show errors
st.sidebar.markdown("---")


# --- Receptor Data Source ---
st.sidebar.subheader("Receptor(s)")
receptor_fetch_method = st.sidebar.radio(
    "Select Receptors:", ("Fetch ALL .pdbqt from GitHub", "Specify Filenames from GitHub"),
    key="receptor_fetch_method", horizontal=True, label_visibility="collapsed"
)
receptor_dir_in_repo = f"{GH_ENSEMBLE_DOCKING_ROOT_PATH}/{RECEPTOR_SUBDIR_GH.strip('/')}"

if receptor_fetch_method == "Fetch ALL .pdbqt from GitHub":
    if st.sidebar.button("Fetch All Receptors", key="fetch_all_receptors_auto", help=f"Fetches all .pdbqt files from .../{receptor_dir_in_repo} on GitHub."):
        st.session_state.fetched_receptor_paths = [] 
        with st.spinner(f"Listing .pdbqt files in GitHub dir..."):
            receptor_filenames = list_files_from_github_repo_dir(
                GH_OWNER, GH_REPO, receptor_dir_in_repo, GH_BRANCH, ".pdbqt"
            )
        if receptor_filenames:
            fetched_receptor_paths_temp = []
            with st.spinner(f"Downloading {len(receptor_filenames)} receptor(s)..."):
                for r_name in receptor_filenames:
                    dl_path_segment = f"{RECEPTOR_SUBDIR_GH.strip('/')}/{r_name}" # Path relative to ensemble-docking/
                    path = download_file_from_github(
                        BASE_GITHUB_URL_FOR_DATA, dl_path_segment, r_name, RECEPTOR_DIR_LOCAL
                    )
                    if path: fetched_receptor_paths_temp.append(path)
            if fetched_receptor_paths_temp:
                st.sidebar.success(f"Fetched {len(fetched_receptor_paths_temp)} receptors.")
                st.session_state.fetched_receptor_paths = fetched_receptor_paths_temp
            else: st.sidebar.error("No receptors downloaded.")
        else: st.sidebar.warning(f"No .pdbqt files found in GitHub directory.")

elif receptor_fetch_method == "Specify Filenames from GitHub":
    receptor_file_names_input = st.sidebar.text_area(
        "Receptor PDBQT Filenames (one per line):", key="receptor_filenames_manual",
        help=f"Enter exact filenames from .../{receptor_dir_in_repo}/ on GitHub."
    )
    if st.sidebar.button("Fetch Specified Receptors", key="fetch_specified_receptors"):
        if receptor_file_names_input.strip():
            receptor_names = [name.strip() for name in receptor_file_names_input.splitlines() if name.strip()]
            st.session_state.fetched_receptor_paths = [] 
            fetched_receptor_paths_temp = []
            with st.spinner(f"Downloading {len(receptor_names)} specified receptor(s)..."):
                for r_name in receptor_names:
                    dl_path_segment = f"{RECEPTOR_SUBDIR_GH.strip('/')}/{r_name}"
                    path = download_file_from_github(
                        BASE_GITHUB_URL_FOR_DATA, dl_path_segment, r_name, RECEPTOR_DIR_LOCAL
                    )
                    if path: fetched_receptor_paths_temp.append(path)
            if fetched_receptor_paths_temp:
                st.sidebar.success(f"Fetched {len(fetched_receptor_paths_temp)} specified receptors.")
                st.session_state.fetched_receptor_paths = fetched_receptor_paths_temp
            else: st.sidebar.error("No specified receptors downloaded. Check filenames.")
        else: st.sidebar.warning("Please enter receptor filenames.")

if 'fetched_receptor_paths' in st.session_state and st.session_state.fetched_receptor_paths:
    exp = st.sidebar.expander(f"{len(st.session_state.fetched_receptor_paths)} Receptor(s) Ready", expanded=False)
    for p_str in st.session_state.fetched_receptor_paths: exp.caption(f"- {Path(p_str).name}")
st.sidebar.markdown("---")

# --- Vina Config File Data Source ---
st.sidebar.subheader("Vina Config File(s)")
config_fetch_method = st.sidebar.radio(
    "Select Config Files:",("Fetch ALL .txt from GitHub", "Specify Filenames from GitHub"),
    key="config_fetch_method", horizontal=True, label_visibility="collapsed"
)
config_dir_in_repo = f"{GH_ENSEMBLE_DOCKING_ROOT_PATH}/{CONFIG_SUBDIR_GH.strip('/')}"

if config_fetch_method == "Fetch ALL .txt from GitHub":
    if st.sidebar.button("Fetch All Configs", key="fetch_all_configs_auto", help=f"Fetches all .txt files from .../{config_dir_in_repo} on GitHub."):
        st.session_state.fetched_config_paths = []
        with st.spinner(f"Listing .txt files in GitHub dir..."):
            config_filenames = list_files_from_github_repo_dir(
                GH_OWNER, GH_REPO, config_dir_in_repo, GH_BRANCH, ".txt"
            )
        if config_filenames:
            fetched_config_paths_temp = []
            with st.spinner(f"Downloading {len(config_filenames)} config file(s)..."):
                for c_name in config_filenames:
                    dl_path_segment = f"{CONFIG_SUBDIR_GH.strip('/')}/{c_name}"
                    path = download_file_from_github(
                        BASE_GITHUB_URL_FOR_DATA, dl_path_segment, c_name, CONFIG_DIR_LOCAL
                    )
                    if path: fetched_config_paths_temp.append(path)
            if fetched_config_paths_temp:
                st.sidebar.success(f"Fetched {len(fetched_config_paths_temp)} config files.")
                st.session_state.fetched_config_paths = fetched_config_paths_temp
            else: st.sidebar.error("No config files downloaded.")
        else: st.sidebar.warning(f"No .txt config files found in GitHub directory.")

elif config_fetch_method == "Specify Filenames from GitHub":
    config_file_names_input = st.sidebar.text_area(
        "Vina Config Filenames (one per line):", key="config_filenames_manual",
        help=f"Enter exact filenames from .../{config_dir_in_repo}/ on GitHub."
    )
    if st.sidebar.button("Fetch Specified Configs", key="fetch_specified_configs"):
        if config_file_names_input.strip():
            config_names = [name.strip() for name in config_file_names_input.splitlines() if name.strip()]
            st.session_state.fetched_config_paths = []
            fetched_config_paths_temp = []
            with st.spinner(f"Downloading {len(config_names)} specified config file(s)..."):
                for c_name in config_names:
                    dl_path_segment = f"{CONFIG_SUBDIR_GH.strip('/')}/{c_name}"
                    path = download_file_from_github(
                        BASE_GITHUB_URL_FOR_DATA, dl_path_segment, c_name, CONFIG_DIR_LOCAL
                    )
                    if path: fetched_config_paths_temp.append(path)
            if fetched_config_paths_temp:
                st.sidebar.success(f"Fetched {len(fetched_config_paths_temp)} specified config files.")
                st.session_state.fetched_config_paths = fetched_config_paths_temp
            else: st.sidebar.error("No specified config files downloaded. Check filenames.")
        else: st.sidebar.warning("Please enter config filenames.")

if 'fetched_config_paths' in st.session_state and st.session_state.fetched_config_paths:
    exp = st.sidebar.expander(f"{len(st.session_state.fetched_config_paths)} Config(s) Ready", expanded=False)
    for p_str in st.session_state.fetched_config_paths: exp.caption(f"- {Path(p_str).name}")
st.sidebar.markdown("---")

# --- Ligand Input Section ---
st.header("🔬 Ligand Input & Preparation")
ligand_input_method = st.radio(
    "Choose your ligand input method:",
    ("SMILES String", "SMILES File (.txt)", "PDBQT File(s)", "Other Ligand File(s) (e.g., SDF, MOL2)", "ZIP Archive of Ligands"),
    key="ligand_method_radio"
)
prepared_ligand_pdbqt_paths = st.session_state.get('prepared_ligand_pdbqt_paths', [])

if ligand_input_method in ["SMILES String", "SMILES File (.txt)"]:
    with st.expander("SMILES Protonation Options", expanded=True):
        col_ph, col_taut, col_ab = st.columns(3)
        g_ph_val = col_ph.number_input("pH for protonation", value=7.4, min_value=0.0, max_value=14.0, step=0.1, key="global_ph")
        g_skip_tautomer = col_taut.checkbox("Skip tautomer enumeration", key="global_skip_taut")
        g_skip_acidbase = col_ab.checkbox("Skip acid-base (protomer) enumeration", key="global_skip_ab")

if ligand_input_method == "SMILES String":
    inchikey_or_smiles_val = st.text_input("Enter InChIKey or SMILES string:", key="smiles_input_text")
    use_inchikey_lookup = st.checkbox("Input is InChIKey (lookup SMILES on PubChem)", value=False, key="use_inchikey_cb")
    single_ligand_name = st.text_input("Ligand Name (for output files)", value="ligand_from_smiles", key="single_lig_name")

    if st.button("Prepare Ligand from SMILES", key="prep_single_smiles_btn"):
        prepared_ligand_pdbqt_paths = [] 
        if inchikey_or_smiles_val and single_ligand_name:
            actual_smiles = ""
            if use_inchikey_lookup:
                with st.spinner(f"Fetching SMILES for InChIKey {inchikey_or_smiles_val}..."):
                    actual_smiles = get_smiles_from_pubchem_inchikey(inchikey_or_smiles_val)
                if actual_smiles: st.info(f"Fetched SMILES: {actual_smiles}")
                else: st.error("Could not fetch SMILES. Please provide SMILES directly.");
            else:
                actual_smiles = inchikey_or_smiles_val
            if actual_smiles and scrub_py_ok and mk_prepare_ligand_py_ok:
                pdbqt_p = convert_smiles_to_pdbqt(actual_smiles, single_ligand_name, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                if pdbqt_p:
                    prepared_ligand_pdbqt_paths.append(pdbqt_p)
                    st.success(f"Ligand '{single_ligand_name}' prepared: `{Path(pdbqt_p).name}`")
            elif not (scrub_py_ok and mk_prepare_ligand_py_ok):
                st.error("Local ligand preparation scripts are missing or not found.")
        else:
            st.warning("Please provide both SMILES/InChIKey and a ligand name.")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

elif ligand_input_method == "SMILES File (.txt)":
    uploaded_smiles_file = st.file_uploader("Upload a .txt file (one SMILES per line):", type="txt", key="smiles_file_uploader")
    if uploaded_smiles_file and st.button("Prepare Ligands from SMILES File", key="prep_smiles_file_btn"):
        prepared_ligand_pdbqt_paths = [] 
        smiles_list = [line.strip() for line in uploaded_smiles_file.read().decode().splitlines() if line.strip()]
        if not smiles_list: st.warning("No SMILES strings found in the uploaded file.")
        elif not (scrub_py_ok and mk_prepare_ligand_py_ok): st.error("Local ligand preparation scripts are missing or not found.")
        else:
            st.info(f"Found {len(smiles_list)} SMILES string(s). Preparing...")
            progress_bar_smiles = st.progress(0)
            for i, smiles_str in enumerate(smiles_list):
                lig_name = f"ligand_file_{i+1}"
                pdbqt_p = convert_smiles_to_pdbqt(smiles_str, lig_name, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                if pdbqt_p: prepared_ligand_pdbqt_paths.append(pdbqt_p)
                progress_bar_smiles.progress((i + 1) / len(smiles_list))
            st.success(f"Finished processing. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

elif ligand_input_method == "PDBQT File(s)":
    uploaded_pdbqt_files = st.file_uploader("Upload PDBQT ligand file(s):", type="pdbqt", accept_multiple_files=True, key="pdbqt_uploader")
    if uploaded_pdbqt_files:
        prepared_ligand_pdbqt_paths = [] 
        for up_file in uploaded_pdbqt_files:
            dest_path = LIGAND_PREP_DIR_LOCAL / up_file.name
            with open(dest_path, "wb") as f: f.write(up_file.getbuffer())
            prepared_ligand_pdbqt_paths.append(str(dest_path))
        st.info(f"Using {len(prepared_ligand_pdbqt_paths)} uploaded PDBQT file(s).")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

elif ligand_input_method == "Other Ligand File(s) (e.g., SDF, MOL2)":
    uploaded_other_files = st.file_uploader("Upload other ligand format file(s):", accept_multiple_files=True, key="other_lig_uploader")
    if uploaded_other_files and st.button("Convert Uploaded Ligand File(s) to PDBQT", key="convert_other_btn"):
        prepared_ligand_pdbqt_paths = [] 
        if not mk_prepare_ligand_py_ok: st.error("Local mk_prepare_ligand.py script is missing, cannot convert.")
        else:
            st.info(f"Processing {len(uploaded_other_files)} file(s) for conversion...")
            progress_bar_other = st.progress(0)
            for i, up_file in enumerate(uploaded_other_files):
                temp_save_path = LIGAND_UPLOAD_TEMP_DIR / up_file.name
                with open(temp_save_path, "wb") as f: f.write(up_file.getbuffer())
                lig_name_base = Path(up_file.name).stem
                pdbqt_p = convert_ligand_file_to_pdbqt(temp_save_path, lig_name_base, LIGAND_PREP_DIR_LOCAL, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                if pdbqt_p: prepared_ligand_pdbqt_paths.append(pdbqt_p)
                progress_bar_other.progress((i + 1) / len(uploaded_other_files))
            st.success(f"Finished conversion. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

elif ligand_input_method == "ZIP Archive of Ligands":
    uploaded_zip_file = st.file_uploader("Upload a ZIP archive of ligand files:", type="zip", key="zip_uploader")
    if uploaded_zip_file and st.button("Process Ligands from ZIP Archive", key="process_zip_btn"):
        prepared_ligand_pdbqt_paths = [] 
        shutil.rmtree(ZIP_EXTRACT_DIR_LOCAL, ignore_errors=True)
        ZIP_EXTRACT_DIR_LOCAL.mkdir(exist_ok=True)
        with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref: zip_ref.extractall(ZIP_EXTRACT_DIR_LOCAL)
        st.info(f"Extracted ZIP to `{ZIP_EXTRACT_DIR_LOCAL}`.")
        files_in_zip = list(p for p in Path(ZIP_EXTRACT_DIR_LOCAL).rglob("*") if p.is_file())
        if not files_in_zip: st.warning("No files found in extracted ZIP.")
        else:
            st.info(f"Found {len(files_in_zip)} file(s) in ZIP. Processing...")
            progress_bar_zip = st.progress(0)
            for i, item_path in enumerate(files_in_zip):
                lig_name_base = item_path.stem
                if item_path.suffix.lower() == ".pdbqt":
                    dest_path = LIGAND_PREP_DIR_LOCAL / item_path.name
                    shutil.copy(item_path, dest_path)
                    prepared_ligand_pdbqt_paths.append(str(dest_path))
                elif mk_prepare_ligand_py_ok:
                    pdbqt_p = convert_ligand_file_to_pdbqt(item_path, lig_name_base, LIGAND_PREP_DIR_LOCAL, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                    if pdbqt_p: prepared_ligand_pdbqt_paths.append(pdbqt_p)
                else: st.warning(f"Skipping conversion of {item_path.name} (mk_prepare_ligand.py missing).")
                progress_bar_zip.progress((i + 1) / len(files_in_zip))
            st.success(f"Finished ZIP. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

current_prepared_ligands = st.session_state.get('prepared_ligand_pdbqt_paths', [])
if current_prepared_ligands:
    exp_lig = st.expander(f"**{len(current_prepared_ligands)} Ligand(s) Ready for Docking**", expanded=False)
    for p_path_str in current_prepared_ligands: exp_lig.caption(f"- `{Path(p_path_str).name}`")

# --- Docking Execution Section ---
st.header("🚀 Docking Execution")

final_ligand_list_for_docking = st.session_state.get('prepared_ligand_pdbqt_paths', [])
current_receptors = st.session_state.get('fetched_receptor_paths', [])
current_configs = st.session_state.get('fetched_config_paths', [])

if not vina_ready:
    st.error("Vina executable is not properly set up. Cannot run docking.")
elif not current_receptors:
    st.warning("No receptors available. Please fetch/select receptors from the sidebar.")
elif not final_ligand_list_for_docking:
    st.warning("No ligands prepared or uploaded. Please prepare/upload ligands first.")
elif not current_configs:
    st.warning("No Vina configuration files available. Please fetch/select config files from the sidebar.")
else:
    st.info(f"Ready to dock {len(final_ligand_list_for_docking)} ligand(s) against {len(current_receptors)} receptor(s).")
    st.caption(f"{len(current_configs)} config file(s) available: {[Path(c).name for c in current_configs]}")

    use_vina_screening_perl = False
    if len(final_ligand_list_for_docking) > 1 and vina_screening_pl_ok and VINA_SCREENING_PL_LOCAL_PATH.exists():
        use_vina_screening_perl = st.checkbox(
            "Use `Vina_screening.pl` (Strict Protein-Config Pairing)?", value=True, 
            key="use_perl_script_cb", 
            help="Recommended for many ligands. Enforces strict protein-config pairing."
        )

    if st.button("Start Docking Run", key="start_docking_main_btn", type="primary"):
        st.markdown("---")
        DOCKING_OUTPUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

        if use_vina_screening_perl:
            st.subheader("Docking via Local `Vina_screening.pl` (Strict Protein-Config Pairing)")
            if not (vina_screening_pl_ok and VINA_SCREENING_PL_LOCAL_PATH.exists() and os.access(VINA_SCREENING_PL_LOCAL_PATH, os.X_OK)):
                st.error("Local `Vina_screening.pl` script is not available or not executable.")
            else:
                ligand_list_file_for_perl = WORKSPACE_PARENT_DIR / "ligands_for_perl.txt"
                with open(ligand_list_file_for_perl, "w") as f:
                    for p_path_str in final_ligand_list_for_docking:
                        f.write(str(Path(p_path_str).resolve()) + "\n")
                st.info(f"Created ligand list for Perl script: `{ligand_list_file_for_perl}`")

                overall_docking_progress = st.progress(0)
                receptors_processed_count = 0
                skipped_receptor_count = 0

                for i, receptor_path_str in enumerate(current_receptors):
                    receptor_file = Path(receptor_path_str)
                    protein_base = receptor_file.stem
                    st.markdown(f"--- \n#### Receptor: `{receptor_file.name}` (Perl Script Mode)")

                    config_to_use_for_perl = None
                    if not current_configs:
                        st.error(f"No Vina configs. Cannot dock {receptor_file.name}.")
                        skipped_receptor_count +=1
                        overall_docking_progress.progress((receptors_processed_count + skipped_receptor_count) / len(current_receptors))
                        continue
                    elif len(current_configs) == 1:
                        config_to_use_for_perl = Path(current_configs[0])
                        st.info(f"Using single available config: `{config_to_use_for_perl.name}`.")
                    else: 
                        config_to_use_for_perl = find_paired_config_for_protein(protein_base, current_configs)
                        if config_to_use_for_perl:
                            st.info(f"Using paired config: `{config_to_use_for_perl.name}`.")
                        else:
                            st.warning(f"No specific paired config found for `{receptor_file.name}`. Skipping.")
                            skipped_receptor_count +=1
                            overall_docking_progress.progress((receptors_processed_count + skipped_receptor_count) / len(current_receptors))
                            continue
                    
                    temp_receptor_for_perl_path = WORKSPACE_PARENT_DIR / receptor_file.name
                    shutil.copy(receptor_file, temp_receptor_for_perl_path)
                    
                    cmd_perl = [
                        "perl", str(VINA_SCREENING_PL_LOCAL_PATH.resolve()),
                        str(VINA_PATH_LOCAL.resolve()), str(temp_receptor_for_perl_path.resolve()),
                        str(config_to_use_for_perl.resolve()), protein_base
                    ]
                    # st.code(f"echo {str(ligand_list_file_for_perl.resolve())} | {' '.join(cmd_perl)}") # Debug

                    try:
                        with open(ligand_list_file_for_perl, "r") as f_ligands:
                            ligand_paths_content_for_stdin = f_ligands.read()
                        process = subprocess.Popen(
                            cmd_perl, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(WORKSPACE_PARENT_DIR.resolve())
                        )
                        stdout_perl, stderr_perl = process.communicate(input=ligand_paths_content_for_stdin)
                        if process.returncode == 0: st.success(f"`Vina_screening.pl` completed for `{receptor_file.name}`.")
                        else: st.error(f"`Vina_screening.pl` failed for `{receptor_file.name}` (RC: {process.returncode}).")
                        
                        exp_stdout = st.expander(f"Perl STDOUT for {receptor_file.name}", expanded=False)
                        exp_stdout.text(stdout_perl if stdout_perl.strip() else "No STDOUT.")
                        exp_stderr = st.expander(f"Perl STDERR for {receptor_file.name}", expanded=not(process.returncode == 0 and not stderr_perl.strip()))
                        exp_stderr.text(stderr_perl if stderr_perl.strip() else "No STDERR.")
                        st.caption(f"Perl script outputs for `{protein_base}` expected in `{WORKSPACE_PARENT_DIR / protein_base}`.")
                    except Exception as e_perl:
                        st.error(f"Error executing `Vina_screening.pl` for `{receptor_file.name}`: {e_perl}")
                    finally:
                        if temp_receptor_for_perl_path.exists(): temp_receptor_for_perl_path.unlink(missing_ok=True)
                    receptors_processed_count += 1
                    overall_docking_progress.progress((receptors_processed_count + skipped_receptor_count) / len(current_receptors))
                if skipped_receptor_count > 0: st.warning(f"{skipped_receptor_count} receptor(s) skipped (no paired config for Perl mode).")

        else: # Direct Vina calls - MODIFIED FOR STRICT PROTEIN-CONFIG PAIRING
            st.subheader("Docking via Direct Vina Calls (Strict Protein-Config Pairing)")
            
            planned_docking_jobs = []
            skipped_receptors_direct_mode = set() # To keep track of skipped receptors for a summary message

            for rec_path_str in current_receptors:
                receptor_file = Path(rec_path_str)
                protein_base = receptor_file.stem
                
                # Find the paired config for this specific protein
                paired_config_file = None
                if not current_configs: # Should be caught by initial checks, but defensive
                    if protein_base not in skipped_receptors_direct_mode:
                        st.warning(f"No config files available to pair with receptor '{receptor_file.name}'. Skipping.")
                        skipped_receptors_direct_mode.add(protein_base)
                    continue
                elif len(current_configs) == 1:
                    paired_config_file = Path(current_configs[0])
                    # st.caption(f"Using single available config '{paired_config_file.name}' for receptor '{receptor_file.name}'.") # Can be verbose
                else:
                    paired_config_file = find_paired_config_for_protein(protein_base, current_configs)
                
                if paired_config_file:
                    # st.caption(f"Receptor '{receptor_file.name}' will use paired config '{paired_config_file.name}'.") # Can be verbose
                    for lig_path_str in final_ligand_list_for_docking:
                        planned_docking_jobs.append({
                            "receptor": receptor_file,
                            "ligand": Path(lig_path_str),
                            "config": paired_config_file, # Use the specifically found paired config
                        })
                else:
                    if protein_base not in skipped_receptors_direct_mode:
                        st.warning(f"No matching config file found for receptor '{receptor_file.name}'. It will be skipped for direct Vina calls.")
                        skipped_receptors_direct_mode.add(protein_base)
            
            num_total_direct_jobs = len(planned_docking_jobs)
            st.info(f"Planning {num_total_direct_jobs} individual Vina docking jobs based on protein-config pairs.")

            if num_total_direct_jobs == 0:
                st.warning("No valid protein-config pairs found for direct Vina calls with the selected files.")
            else:
                job_counter = 0
                overall_docking_progress = st.progress(0)

                for job_spec in planned_docking_jobs:
                    receptor_file = job_spec["receptor"]
                    ligand_file = job_spec["ligand"]
                    config_file = job_spec["config"] # This is the paired config

                    job_counter += 1
                    st.markdown(f"--- \n**Job {job_counter}/{num_total_direct_jobs}:** "
                                f"`{receptor_file.name}` + `{ligand_file.name}` (Paired Config: `{config_file.name}`)")

                    output_base = f"{receptor_file.stem}_{ligand_file.stem}_{config_file.stem}" # Name includes specific config
                    output_pdbqt_docked = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base}_out.pdbqt"
                    output_log_file = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base}_log.txt"

                    cmd_vina_direct = [
                        str(VINA_PATH_LOCAL.resolve()),
                        "--receptor", str(receptor_file.resolve()),
                        "--ligand", str(ligand_file.resolve()),
                        "--config", str(config_file.resolve()), 
                        "--out", str(output_pdbqt_docked.resolve()),
                        "--log", str(output_log_file.resolve())
                    ]
                    # st.code(" ".join(cmd_vina_direct)) # Debug

                    try:
                        vina_run_result = subprocess.run(cmd_vina_direct, capture_output=True, text=True, check=True, cwd=str(WORKSPACE_PARENT_DIR.resolve()))
                        st.success("Vina job completed!")
                        with st.expander("Vina STDOUT", expanded=False):
                            st.text(vina_run_result.stdout if vina_run_result.stdout.strip() else "No STDOUT.")
                        if vina_run_result.stderr.strip():
                            with st.expander("Vina STDERR", expanded=True): st.text(vina_run_result.stderr)
                        
                        if output_pdbqt_docked.exists():
                            with open(output_pdbqt_docked, "rb") as fp:
                                st.download_button(f"DL Docked PDBQT ({output_pdbqt_docked.name})", fp, output_pdbqt_docked.name, "application/octet-stream", key=f"dl_pdbqt_{job_counter}") # Changed mime for pdbqt
                        if output_log_file.exists():
                            with open(output_log_file, "r", encoding='utf-8') as fp:
                                 st.download_button(f"DL Log ({output_log_file.name})", fp.read(), output_log_file.name, "text/plain",  key=f"dl_log_{job_counter}")
                    except subprocess.CalledProcessError as e_vina_direct:
                        st.error(f"Vina job FAILED (RC: {e_vina_direct.returncode}).")
                        with st.expander("Vina Error Details", expanded=True):
                            st.error(f"Cmd: `{' '.join(e_vina_direct.cmd)}`")
                            st.text("STDOUT:\n" + (e_vina_direct.stdout if e_vina_direct.stdout.strip() else "No STDOUT."))
                            st.text("STDERR:\n" + (e_vina_direct.stderr if e_vina_direct.stderr.strip() else "No STDERR."))
                    except Exception as e_generic: st.error(f"Unexpected error in Vina job: {e_generic}")
                    
                    if num_total_direct_jobs > 0 : 
                        overall_docking_progress.progress(job_counter / num_total_direct_jobs)
        
        st.balloons()
        st.header("🏁 Docking Run Finished 🏁")
        st.info(f"Outputs are in `{DOCKING_OUTPUT_DIR_LOCAL}` (relative to app root).")

st.markdown("---")
st.caption(f"Docking App v{APP_VERSION} | AutoDock Vina Ensemble Docker")
