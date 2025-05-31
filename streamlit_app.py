import streamlit as st
import subprocess
import os
import stat
import requests # Still needed for PubChem and optionally remote receptors/configs
import zipfile
# import tempfile # Might not be needed if LIGAND_UPLOAD_TEMP_DIR is used directly
import shutil
from urllib.parse import urljoin # Still needed for optional remote receptors/configs
from pathlib import Path
import sys # <--- IMPORT SYS TO USE SYS.EXECUTABLE

# --- Configuration ---
APP_VERSION = "1.2.5" # For tracking changes (added sys.executable and debug)

# GitHub URL for RECEPTORS and CONFIGS (if they are still remote)
BASE_GITHUB_URL_FOR_DATA = "https://raw.githubusercontent.com/HenryChritopher02/bace1/main/ensemble-docking/"
RECEPTOR_SUBDIR_GH = "ensemble_protein/" # Relative to BASE_GITHUB_URL_FOR_DATA
CONFIG_SUBDIR_GH = "config/"           # Relative to BASE_GITHUB_URL_FOR_DATA

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

# --- Helper Functions ---

def initialize_directories():
    """Creates necessary local workspace and output directories."""
    dirs_to_create = [
        WORKSPACE_PARENT_DIR, RECEPTOR_DIR_LOCAL, CONFIG_DIR_LOCAL,
        LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR,
        ZIP_EXTRACT_DIR_LOCAL, DOCKING_OUTPUT_DIR_LOCAL
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

def download_file_from_github(github_base_url, relative_path_on_github, local_filename, local_save_dir):
    """Downloads a file from a specified GitHub repo to a local path."""
    full_url = urljoin(github_base_url, relative_path_on_github)
    local_file_path = Path(local_save_dir) / local_filename
    try:
        with st.spinner(f"Downloading {local_filename} from {full_url}..."):
            response = requests.get(full_url, stream=True)
            response.raise_for_status()
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.sidebar.success(f"Downloaded: {local_filename}")
        return str(local_file_path)
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error downloading {relative_path_on_github}: {e}")
        return None

def make_file_executable(filepath_str):
    """Makes a file executable."""
    if not filepath_str or not os.path.exists(filepath_str):
        st.sidebar.warning(f"Cannot make executable: File not found at {filepath_str}")
        return False
    try:
        current_mode = os.stat(filepath_str).st_mode
        os.chmod(filepath_str, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        # st.sidebar.info(f"Made {Path(filepath_str).name} executable.") # Reduced verbosity
        return True
    except Exception as e:
        st.sidebar.error(f"Error making {Path(filepath_str).name} executable: {e}")
        return False

def check_script_exists(script_path: Path, script_name: str):
    """Checks if a local script file exists and displays status."""
    if script_path.exists() and script_path.is_file():
        st.sidebar.caption(f"Found: `{script_name}`") # Reduced verbosity
        return True
    else:
        st.sidebar.error(f"NOT FOUND: `{script_name}` (expected at `{script_path}`)")
        st.sidebar.caption("Ensure `ensemble-docking` folder is in your app repo.")
        return False

def check_vina_binary():
    """Checks for Vina binary and its permissions."""
    st.sidebar.subheader("Vina Binary Status")
    # st.sidebar.write(f"Expected Vina path: `{VINA_PATH_LOCAL.resolve()}`") # Verbose
    if not VINA_PATH_LOCAL.exists():
        st.sidebar.error(
            f"Vina exe NOT FOUND at `{VINA_PATH_LOCAL}`. "
            f"Ensure `{VINA_EXECUTABLE_NAME}` is in `{VINA_DIR_LOCAL}`."
        )
        return False

    st.sidebar.success(f"Vina binary found: `{VINA_PATH_LOCAL.name}`")
    is_executable = os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK)
    if is_executable:
        st.sidebar.success("Vina binary is executable.")
        return True
    else:
        st.sidebar.warning("Vina binary is NOT executable.")
        st.sidebar.info("Attempting to set execute permission...")
        if make_file_executable(str(VINA_PATH_LOCAL)):
            if os.access(str(VINA_PATH_LOCAL.resolve()), os.X_OK):
                st.sidebar.success("Execute permission set successfully for Vina.")
                return True
            else:
                st.sidebar.error("Failed to set execute permission (still not executable).")
        else:
            st.sidebar.error("Failed to set execute permission (chmod call failed).")
        st.sidebar.markdown(
            f"**Action:** `git add --chmod=+x {VINA_DIR_LOCAL.name}/{VINA_EXECUTABLE_NAME}`, commit, push."
        )
        return False

def get_smiles_from_pubchem_inchikey(inchikey_str):
    api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchikey_str}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except requests.exceptions.RequestException as e:
        st.warning(f"PubChem API error for InChIKey {inchikey_str}: {e}")
    except (KeyError, IndexError) as e:
        st.warning(f"Could not parse SMILES from PubChem response for {inchikey_str}: {e}")
    return None

def run_ligand_prep_script(script_local_path_str, script_args, process_name, ligand_name_for_log):
    if not script_local_path_str:
        st.error(f"{process_name}: Script path is not defined.")
        return False

    absolute_script_path = str(Path(script_local_path_str).resolve())
    if not os.path.exists(absolute_script_path):
        st.error(f"{process_name} script not found at resolved absolute path: {absolute_script_path}")
        st.error(f"(Original path provided: {script_local_path_str})")
        return False

    # Use sys.executable to ensure the correct Python interpreter is used
    command = [sys.executable, absolute_script_path] + script_args # <--- CHANGED TO SYS.EXECUTABLE
    
    cwd_path_resolved = str(WORKSPACE_PARENT_DIR.resolve())
    if not os.path.exists(cwd_path_resolved):
        st.error(f"Working directory {cwd_path_resolved} for {process_name} does not exist.")
        return False

    try:
        st.info(f"Running {process_name} for {ligand_name_for_log}...")
        st.caption(f"Command: {sys.executable} {absolute_script_path} {' '.join(map(str, script_args))}")
        st.caption(f"Working Directory: {cwd_path_resolved}")
        
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=cwd_path_resolved)
        
        with st.expander(f"{process_name} STDOUT for {ligand_name_for_log}", expanded=False):
            st.text(result.stdout if result.stdout.strip() else "No standard output.")
        
        if result.stderr.strip(): 
             with st.expander(f"{process_name} STDERR for {ligand_name_for_log} (check for warnings/errors)", expanded=True):
                st.text(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error during {process_name} for {ligand_name_for_log}:")
        st.error(f"Command: `{' '.join(e.cmd)}`")
        st.error(f"Return code: {e.returncode}")
        with st.expander(f"{process_name} STDOUT (on error)", expanded=True):
            st.text(e.stdout if e.stdout.strip() else "No standard output.")
        with st.expander(f"{process_name} STDERR (on error)", expanded=True):
            st.text(e.stderr if e.stderr.strip() else "No standard error output.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred running {process_name} for {ligand_name_for_log}: {e}")
        return False

def convert_smiles_to_pdbqt(smiles_str, ligand_name_base, output_dir_path_for_final_pdbqt, ph_val, skip_taut, skip_acidbase, local_scrub_script_path, local_mk_prepare_script_path):
    """Converts a single SMILES string to a PDBQT file using local scripts.
    Outputs will be relative to WORKSPACE_PARENT_DIR when scripts are run.
    """
    # output_dir_path_for_final_pdbqt is LIGAND_PREP_DIR_LOCAL

    # Ensure the final PDBQT output directory (within workspace) exists
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)

    # Define filenames RELATIVE to WORKSPACE_PARENT_DIR (which will be the CWD for the scripts)
    # The "prepared_ligands" part of the path is relative to WORKSPACE_PARENT_DIR
    relative_sdf_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}_scrubbed.sdf"
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"

    # Absolute paths are useful for checks and for the final return value
    absolute_sdf_path = WORKSPACE_PARENT_DIR / relative_sdf_filename
    absolute_pdbqt_path = WORKSPACE_PARENT_DIR / relative_pdbqt_filename # This is the same as output_dir_path_for_final_pdbqt / f"{ligand_name_base}.pdbqt"

    scrub_options = ["--ph", str(ph_val)]
    if skip_taut: scrub_options.append("--skip_tautomer")
    if skip_acidbase: scrub_options.append("--skip_acidbase")

    # Pass the RELATIVE path to scrub.py because its cwd will be WORKSPACE_PARENT_DIR
    scrub_args = [smiles_str, "-o", str(relative_sdf_filename)] + scrub_options
    if not run_ligand_prep_script(str(local_scrub_script_path), scrub_args, "scrub.py", ligand_name_base):
        return None

    # For mk_prepare_ligand.py:
    # Input is the SDF file just created. We can give its path relative to WORKSPACE_PARENT_DIR.
    # Output is the PDBQT file, also relative to WORKSPACE_PARENT_DIR.
    mk_prepare_args = ["-i", str(relative_sdf_filename), "-o", str(relative_pdbqt_filename)]
    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base):
        return None

    return str(absolute_pdbqt_path) if absolute_pdbqt_path.exists() else None

def convert_ligand_file_to_pdbqt(input_ligand_file_path_absolute, ligand_name_base, output_dir_path_for_final_pdbqt, local_mk_prepare_script_path):
    """Converts an input ligand file to PDBQT using local mk_prepare_ligand.py.
    The input_ligand_file_path_absolute is the path to the uploaded file.
    Outputs will be relative to WORKSPACE_PARENT_DIR.
    """
    # output_dir_path_for_final_pdbqt is LIGAND_PREP_DIR_LOCAL
    output_dir_path_for_final_pdbqt.mkdir(parents=True, exist_ok=True)

    # Define output PDBQT filename RELATIVE to WORKSPACE_PARENT_DIR
    relative_pdbqt_filename = Path(LIGAND_PREP_DIR_LOCAL.name) / f"{ligand_name_base}.pdbqt"
    absolute_pdbqt_path = WORKSPACE_PARENT_DIR / relative_pdbqt_filename

    # mk_prepare_ligand.py needs an input. We pass the absolute path to the uploaded file.
    # It's generally safer for tools if their input paths are unambiguous (absolute).
    # The output path is relative to its CWD (WORKSPACE_PARENT_DIR).
    mk_prepare_args = ["-i", str(Path(input_ligand_file_path_absolute).resolve()), 
                       "-o", str(relative_pdbqt_filename)]
    if not run_ligand_prep_script(str(local_mk_prepare_script_path), mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base):
        return None
    return str(absolute_pdbqt_path) if absolute_pdbqt_path.exists() else None

# --- Main Application ---
st.set_page_config(page_title="Ensemble AutoDock Vina", layout="wide")
st.title(f"🧬 Ensemble AutoDock Vina Docking App (v{APP_VERSION})")

initialize_directories()

# --- Sidebar for Setup & Global Parameters ---
st.sidebar.header("🔧 Global Setup & Parameters")

# --- DEBUGGING SECTION ---
st.sidebar.subheader("🕵️ DEBUG Information")
# Check content of deployed scrub.py
if SCRUB_PY_LOCAL_PATH.exists():
    try:
        with open(SCRUB_PY_LOCAL_PATH, "r") as f_debug_scrub:
            st.sidebar.info(f"First 250 chars of deployed {SCRUB_PY_LOCAL_PATH.name}:")
            st.sidebar.code(f_debug_scrub.read(250))
    except Exception as e_read_scrub:
        st.sidebar.error(f"Error reading {SCRUB_PY_LOCAL_PATH.name} for debug: {e_read_scrub}")
else:
    st.sidebar.error(f"DEBUG: {SCRUB_PY_LOCAL_PATH.name} does NOT exist at: {SCRUB_PY_LOCAL_PATH}")

# Check molscrub installation and importability
try:
    st.sidebar.info("Checking molscrub installation via subprocess...")
    molscrub_check_command = [
        sys.executable, "-c", # Use sys.executable here too
        "import sys; "
        "print(f'Python version: {sys.version}'); "
        "print(f'Python executable: {sys.executable}'); "
        "print(f'sys.path: {sys.path}'); "
        "try: "
        "    import scrubber; "
        "    print(f'scrubber imported. Path: {scrubber.__file__}'); "
        "    print(f'scrubber version: {scrubber.__version__ if hasattr(scrubber, \"__version__\") else \"N/A\"}'); "
        "    from scrubber import Scrub; " # Test the critical import
        "    print('*** SUCCESS: from scrubber import Scrub works ***'); "
        "except ImportError as e_i: "
        "    print(f'!!! FAILED to import scrubber or Scrub from molscrub: {e_i}'); "
        "except Exception as e_g: "
        "    print(f'!!! Some other error during scrubber check: {e_g}')"
    ]
    molscrub_result = subprocess.run(molscrub_check_command, capture_output=True, text=True, timeout=20) # Increased timeout
    debug_expander = st.sidebar.expander("`molscrub` Installation & Importability Check Details")
    debug_expander.text("molscrub check STDOUT:")
    debug_expander.code(molscrub_result.stdout if molscrub_result.stdout else "No STDOUT")
    debug_expander.text("molscrub check STDERR:")
    debug_expander.code(molscrub_result.stderr if molscrub_result.stderr else "No STDERR")
except Exception as e_sub_debug:
    st.sidebar.error(f"Error running molscrub debug subprocess: {e_sub_debug}")
st.sidebar.markdown("---") # End of debug section
# --- END DEBUGGING SECTION ---


st.sidebar.subheader("Helper Scripts Status")
scrub_py_ok = check_script_exists(SCRUB_PY_LOCAL_PATH, "scrub.py")
if scrub_py_ok:
    make_file_executable(str(SCRUB_PY_LOCAL_PATH))

mk_prepare_ligand_py_ok = check_script_exists(MK_PREPARE_LIGAND_PY_LOCAL_PATH, "mk_prepare_ligand.py")
if mk_prepare_ligand_py_ok:
    make_file_executable(str(MK_PREPARE_LIGAND_PY_LOCAL_PATH))

vina_screening_pl_ok = check_script_exists(VINA_SCREENING_PL_LOCAL_PATH, "Vina_screening.pl")
if vina_screening_pl_ok:
    make_file_executable(str(VINA_SCREENING_PL_LOCAL_PATH))

vina_ready = check_vina_binary()


st.sidebar.subheader("Receptor Data Source")
receptor_file_names_input = st.sidebar.text_area(
    f"Receptor PDBQT filenames (one per line, from remote .../{RECEPTOR_SUBDIR_GH})",
    help="Enter exact filenames of receptor PDBQT files from the GitHub repository."
)
if receptor_file_names_input.strip() and st.sidebar.button("Fetch Remote Receptors", key="fetch_receptors"):
    receptor_names = [name.strip() for name in receptor_file_names_input.splitlines() if name.strip()]
    fetched_receptor_paths_temp = []
    with st.spinner("Fetching receptor files..."):
        for r_name in receptor_names:
            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, RECEPTOR_SUBDIR_GH + r_name, r_name, RECEPTOR_DIR_LOCAL)
            if path: fetched_receptor_paths_temp.append(path)
    if fetched_receptor_paths_temp:
        st.sidebar.success(f"Successfully fetched {len(fetched_receptor_paths_temp)} remote receptors.")
        st.session_state.fetched_receptor_paths = fetched_receptor_paths_temp
    else:
        st.sidebar.error("No remote receptors fetched. Check filenames or GitHub path.")

if 'fetched_receptor_paths' in st.session_state and st.session_state.fetched_receptor_paths:
    st.sidebar.markdown(f"**{len(st.session_state.fetched_receptor_paths)} Receptors Ready:**")
    for p in st.session_state.fetched_receptor_paths:
        st.sidebar.caption(f"- {Path(p).name}")


st.sidebar.subheader("Vina Config File Data Source")
config_file_names_input = st.sidebar.text_area(
    f"Vina config filenames (one per line, from remote .../{CONFIG_SUBDIR_GH})",
    help="Enter exact filenames of Vina config TXT files from the GitHub repository."
)
if config_file_names_input.strip() and st.sidebar.button("Fetch Remote Config Files", key="fetch_configs"):
    config_names = [name.strip() for name in config_file_names_input.splitlines() if name.strip()]
    fetched_config_paths_temp = []
    with st.spinner("Fetching remote config files..."):
        for c_name in config_names:
            path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, CONFIG_SUBDIR_GH + c_name, c_name, CONFIG_DIR_LOCAL)
            if path: fetched_config_paths_temp.append(path)
    if fetched_config_paths_temp:
        st.sidebar.success(f"Successfully fetched {len(fetched_config_paths_temp)} remote config files.")
        st.session_state.fetched_config_paths = fetched_config_paths_temp
    else:
        st.sidebar.error("No remote config files fetched. Check filenames or GitHub path.")

if 'fetched_config_paths' in st.session_state and st.session_state.fetched_config_paths:
    st.sidebar.markdown(f"**{len(st.session_state.fetched_config_paths)} Configs Ready:**")
    for p in st.session_state.fetched_config_paths:
        st.sidebar.caption(f"- {Path(p).name}")

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
                    st.success(f"Ligand '{single_ligand_name}' prepared: `{pdbqt_p}`")
            elif not (scrub_py_ok and mk_prepare_ligand_py_ok):
                st.error("Local ligand preparation scripts (scrub.py, mk_prepare_ligand.py) are missing or not found.")
        else:
            st.warning("Please provide both SMILES/InChIKey and a ligand name.")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths


elif ligand_input_method == "SMILES File (.txt)":
    uploaded_smiles_file = st.file_uploader("Upload a .txt file (one SMILES per line):", type="txt", key="smiles_file_uploader")
    if uploaded_smiles_file and st.button("Prepare Ligands from SMILES File", key="prep_smiles_file_btn"):
        prepared_ligand_pdbqt_paths = [] 
        smiles_list = [line.strip() for line in uploaded_smiles_file.read().decode().splitlines() if line.strip()]
        if not smiles_list:
            st.warning("No SMILES strings found in the uploaded file.")
        elif not (scrub_py_ok and mk_prepare_ligand_py_ok):
            st.error("Local ligand preparation scripts (scrub.py, mk_prepare_ligand.py) are missing or not found.")
        else:
            st.info(f"Found {len(smiles_list)} SMILES string(s). Preparing...")
            progress_bar_smiles = st.progress(0)
            for i, smiles_str in enumerate(smiles_list):
                lig_name = f"ligand_file_{i+1}"
                pdbqt_p = convert_smiles_to_pdbqt(smiles_str, lig_name, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                if pdbqt_p: prepared_ligand_pdbqt_paths.append(pdbqt_p)
                progress_bar_smiles.progress((i + 1) / len(smiles_list))
            st.success(f"Finished processing SMILES file. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

elif ligand_input_method == "PDBQT File(s)":
    uploaded_pdbqt_files = st.file_uploader("Upload PDBQT ligand file(s):", type="pdbqt", accept_multiple_files=True, key="pdbqt_uploader")
    if uploaded_pdbqt_files:
        prepared_ligand_pdbqt_paths = [] 
        for up_file in uploaded_pdbqt_files:
            dest_path = LIGAND_PREP_DIR_LOCAL / up_file.name
            with open(dest_path, "wb") as f:
                f.write(up_file.getbuffer())
            prepared_ligand_pdbqt_paths.append(str(dest_path))
        st.info(f"Using {len(prepared_ligand_pdbqt_paths)} uploaded PDBQT file(s).")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

elif ligand_input_method == "Other Ligand File(s) (e.g., SDF, MOL2)":
    uploaded_other_files = st.file_uploader("Upload other ligand format file(s) (e.g., SDF, MOL2):", accept_multiple_files=True, key="other_lig_uploader")
    if uploaded_other_files and st.button("Convert Uploaded Ligand File(s) to PDBQT", key="convert_other_btn"):
        prepared_ligand_pdbqt_paths = [] 
        if not mk_prepare_ligand_py_ok:
            st.error("Local mk_prepare_ligand.py script is missing or not found, cannot convert.")
        else:
            st.info(f"Processing {len(uploaded_other_files)} uploaded file(s) for conversion...")
            progress_bar_other = st.progress(0)
            for i, up_file in enumerate(uploaded_other_files):
                temp_save_path = LIGAND_UPLOAD_TEMP_DIR / up_file.name
                with open(temp_save_path, "wb") as f:
                    f.write(up_file.getbuffer())
                lig_name_base = Path(up_file.name).stem
                pdbqt_p = convert_ligand_file_to_pdbqt(temp_save_path, lig_name_base, LIGAND_PREP_DIR_LOCAL, MK_PREPARE_LIGAND_PY_LOCAL_PATH)
                if pdbqt_p: prepared_ligand_pdbqt_paths.append(pdbqt_p)
                progress_bar_other.progress((i + 1) / len(uploaded_other_files))
            st.success(f"Finished conversion. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

elif ligand_input_method == "ZIP Archive of Ligands":
    uploaded_zip_file = st.file_uploader("Upload a ZIP archive containing ligand files:", type="zip", key="zip_uploader")
    if uploaded_zip_file and st.button("Process Ligands from ZIP Archive", key="process_zip_btn"):
        prepared_ligand_pdbqt_paths = [] 
        shutil.rmtree(ZIP_EXTRACT_DIR_LOCAL, ignore_errors=True)
        ZIP_EXTRACT_DIR_LOCAL.mkdir(exist_ok=True)
        with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref:
            zip_ref.extractall(ZIP_EXTRACT_DIR_LOCAL)
        st.info(f"Extracted ZIP archive to `{ZIP_EXTRACT_DIR_LOCAL}`.")
        files_in_zip = list(Path(p) for p in Path(ZIP_EXTRACT_DIR_LOCAL).rglob("*") if p.is_file())
        if not files_in_zip:
            st.warning("No files found in the extracted ZIP archive.")
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
                else:
                    st.warning(f"Skipping conversion of {item_path.name} (mk_prepare_ligand.py missing or not found).")
                progress_bar_zip.progress((i + 1) / len(files_in_zip))
            st.success(f"Finished processing ZIP. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")
        st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths

current_prepared_ligands = st.session_state.get('prepared_ligand_pdbqt_paths', [])
if current_prepared_ligands:
    st.success(f"**Total {len(current_prepared_ligands)} ligand(s) ready for docking.**")
    with st.expander("View Ready Ligands"):
        for p_path_str in current_prepared_ligands:
            st.caption(f"- `{Path(p_path_str).name}` (at `{p_path_str}`)")

# --- Docking Execution Section ---
# (Rest of the docking code remains the same as version 1.3.0 you provided)
# ... (ensure to copy the entire docking execution block from your previous full script) ...
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
    st.info(f"Ready to dock {len(final_ligand_list_for_docking)} ligand(s) against {len(current_receptors)} receptor(s) using {len(current_configs)} configuration(s).")

    use_vina_screening_perl = False
    if len(final_ligand_list_for_docking) > 1 and vina_screening_pl_ok and VINA_SCREENING_PL_LOCAL_PATH.exists():
        use_vina_screening_perl = st.checkbox(
            "Use local `Vina_screening.pl` for docking multiple ligands? (Recommended for many ligands; processes one receptor at a time)",
            value=True, key="use_perl_script_cb"
        )

    if st.button("Start Docking Run", key="start_docking_main_btn", type="primary"):
        st.markdown("---")
        DOCKING_OUTPUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

        if use_vina_screening_perl:
            st.subheader("Docking via Local `Vina_screening.pl`")
            if not (vina_screening_pl_ok and VINA_SCREENING_PL_LOCAL_PATH.exists() and os.access(VINA_SCREENING_PL_LOCAL_PATH, os.X_OK)):
                st.error("Local `Vina_screening.pl` script is not available or not executable.")
            else:
                ligand_list_file_for_perl = WORKSPACE_PARENT_DIR / "ligands_for_perl.txt"
                with open(ligand_list_file_for_perl, "w") as f:
                    for p_path_str in final_ligand_list_for_docking:
                        f.write(str(Path(p_path_str).resolve()) + "\n")
                st.info(f"Created ligand list for Perl script: `{ligand_list_file_for_perl}`")

                overall_docking_progress = st.progress(0)
                for i, receptor_path_str in enumerate(current_receptors):
                    receptor_file = Path(receptor_path_str)
                    protein_base = receptor_file.stem
                    st.markdown(f"#### Processing Receptor: `{receptor_file.name}`")

                    temp_receptor_for_perl = WORKSPACE_PARENT_DIR / receptor_file.name
                    shutil.copy(receptor_file, temp_receptor_for_perl)

                    config_for_this_protein = None
                    if len(current_configs) == 1:
                        config_for_this_protein = Path(current_configs[0])
                    else:
                        for cfg_path_str in current_configs:
                            cfg_file = Path(cfg_path_str)
                            if protein_base in cfg_file.name:
                                config_for_this_protein = cfg_file
                                break
                    
                    temp_config_for_perl = None
                    if config_for_this_protein:
                        expected_config_name_for_perl = f"{protein_base}.txt"
                        temp_config_for_perl = WORKSPACE_PARENT_DIR / expected_config_name_for_perl
                        shutil.copy(config_for_this_protein, temp_config_for_perl)
                        st.info(f"Using config `{config_for_this_protein.name}` (as `{expected_config_name_for_perl}`) for `{receptor_file.name}`.")
                    else:
                        st.warning(f"Could not determine a specific config for `{receptor_file.name}`.")

                    protein_specific_output_dir = DOCKING_OUTPUT_DIR_LOCAL / protein_base
                    protein_specific_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    cmd_perl = [
                        "perl",
                        str(VINA_SCREENING_PL_LOCAL_PATH.resolve()), # Use local Perl script path
                        protein_base
                    ]
                    st.code(f"echo {str(ligand_list_file_for_perl.resolve())} | {' '.join(cmd_perl)}")

                    try:
                        process = subprocess.Popen(
                            cmd_perl, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=str(WORKSPACE_PARENT_DIR.resolve()) # Ensure CWD is resolved
                        )
                        stdout_perl, stderr_perl = process.communicate(input=str(ligand_list_file_for_perl.resolve()) + "\n")

                        if process.returncode == 0:
                            st.success(f"`Vina_screening.pl` completed for `{receptor_file.name}`.")
                        else:
                            st.error(f"`Vina_screening.pl` failed for `{receptor_file.name}` (Return code: {process.returncode}).")
                        with st.expander(f"Perl script STDOUT for {receptor_file.name}", expanded=False):
                            st.text(stdout_perl if stdout_perl else "No standard output.")
                        with st.expander(f"Perl script STDERR for {receptor_file.name}", expanded=True):
                            st.text(stderr_perl if stderr_perl else "No standard error output.")
                        st.info(f"Docking outputs for `{receptor_file.name}` should be in `{protein_specific_output_dir}`.")

                    except Exception as e_perl:
                        st.error(f"Error executing `Vina_screening.pl` for `{receptor_file.name}`: {e_perl}")
                    finally:
                        if temp_receptor_for_perl and temp_receptor_for_perl.exists(): temp_receptor_for_perl.unlink(missing_ok=True)
                        if temp_config_for_perl and temp_config_for_perl.exists(): temp_config_for_perl.unlink(missing_ok=True)
                    
                    overall_docking_progress.progress((i + 1) / len(current_receptors))
        else: # Direct Vina calls
            st.subheader("Docking via Direct AutoDock Vina Calls")
            num_total_direct_jobs = len(current_receptors) * len(final_ligand_list_for_docking) * len(current_configs)
            st.info(f"Planning {num_total_direct_jobs} individual Vina docking jobs.")
            if num_total_direct_jobs == 0:
                st.warning("No valid combination of receptor, ligand, and config for direct Vina calls.")

            job_counter = 0
            overall_docking_progress = st.progress(0)

            for rec_path_str in current_receptors:
                receptor_file = Path(rec_path_str)
                for lig_path_str in final_ligand_list_for_docking:
                    ligand_file = Path(lig_path_str)
                    for conf_path_str in current_configs:
                        config_file = Path(conf_path_str)
                        job_counter += 1
                        st.markdown(f"--- \n**Job {job_counter}/{num_total_direct_jobs}:** "
                                    f"`{receptor_file.name}` + `{ligand_file.name}` (Config: `{config_file.name}`)")

                        output_base = f"{receptor_file.stem}_{ligand_file.stem}_{config_file.stem}"
                        output_pdbqt_docked = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base}_out.pdbqt"
                        output_log_file = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base}_log.txt"

                        cmd_vina_direct = [
                            str(VINA_PATH_LOCAL.resolve()),
                            "--receptor", str(receptor_file.resolve()),
                            "--ligand", str(ligand_file.resolve()),
                            "--config", str(config_file.resolve()),
                            "--out", str(output_pdbqt_docked.resolve())
                        ]
                        st.code(" ".join(cmd_vina_direct))

                        try:
                            vina_run_result = subprocess.run(cmd_vina_direct, capture_output=True, text=True, check=True, cwd=str(WORKSPACE_PARENT_DIR.resolve()))
                            st.success("Vina docking job completed successfully!")
                            with st.expander("Vina STDOUT", expanded=False):
                                st.text(vina_run_result.stdout if vina_run_result.stdout else "No standard output.")
                            if vina_run_result.stderr:
                                with st.expander("Vina STDERR (check for warnings)", expanded=True):
                                    st.text(vina_run_result.stderr)
                            
                            if output_pdbqt_docked.exists():
                                with open(output_pdbqt_docked, "rb") as fp:
                                    st.download_button(label=f"Download Docked PDBQT ({output_pdbqt_docked.name})", data=fp, file_name=output_pdbqt_docked.name, mime="chemical/x-pdbqt", key=f"dl_pdbqt_{job_counter}")
                            if output_log_file.exists():
                                with open(output_log_file, "r", encoding='utf-8') as fp:
                                     st.download_button(label=f"Download Log File ({output_log_file.name})", data=fp.read(), file_name=output_log_file.name, mime="text/plain",  key=f"dl_log_{job_counter}")

                        except subprocess.CalledProcessError as e_vina_direct:
                            st.error(f"Vina docking job FAILED (Return code: {e_vina_direct.returncode}).")
                            st.error(f"Command: `{' '.join(e_vina_direct.cmd)}`")
                            with st.expander("Vina STDOUT (on error)", expanded=True):
                                st.text(e_vina_direct.stdout if e_vina_direct.stdout else "No standard output.")
                            with st.expander("Vina STDERR (on error)", expanded=True):
                                st.text(e_vina_direct.stderr if e_vina_direct.stderr else "No standard error output.")
                        except Exception as e_generic:
                            st.error(f"An unexpected error occurred during this Vina job: {e_generic}")
                        
                        if num_total_direct_jobs > 0 :
                            overall_docking_progress.progress(job_counter / num_total_direct_jobs)
        
        st.balloons()
        st.header("🏁 Docking Run Finished 🏁")
        st.info(f"All docking outputs can be found in the `{DOCKING_OUTPUT_DIR_LOCAL}` directory (relative to the app's root on the server). You may need to use other tools or app features (if added) to browse/manage these files directly on the server.")

st.markdown("---")
st.caption(f"Docking App v{APP_VERSION} | Developed with Streamlit. AutoDock Vina for docking simulations.")
