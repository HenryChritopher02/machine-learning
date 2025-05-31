import streamlit as st
import subprocess
import os
import stat
import requests
import zipfile
import tempfile
import shutil
from urllib.parse import urljoin
from pathlib import Path

# --- Configuration ---
APP_VERSION = "1.2.5" # For tracking changes

# Base URL for fetching resources from the specified GitHub repository
BASE_GITHUB_URL = "https://raw.githubusercontent.com/HenryChritopher02/bace1/main/ensemble-docking/"

# Relative paths to helper scripts and directories within the GitHub repo (from BASE_GITHUB_URL)
SCRUB_PY_REL_PATH = "ligand_preprocessing/scrub.py"
MK_PREPARE_LIGAND_PY_REL_PATH = "ligand_preprocessing/mk_prepare_ligand.py"
VINA_SCREENING_PL_REL_PATH = "Vina_screening.pl"
RECEPTOR_SUBDIR_GH = "ensemble_protein/"
CONFIG_SUBDIR_GH = "config/"

# Local Vina executable details
VINA_DIR_LOCAL = Path("./vina")  # Assumes vina executable is in a 'vina' subfolder in your app's repo
VINA_EXECUTABLE_NAME = "vina_1.2.5_linux_x86_64"
VINA_PATH_LOCAL = VINA_DIR_LOCAL / VINA_EXECUTABLE_NAME

# Local workspace directories (will be created)
WORKSPACE_PARENT_DIR = Path("./autodock_workspace")
HELPER_SCRIPTS_DIR_LOCAL = WORKSPACE_PARENT_DIR / "helper_scripts"
RECEPTOR_DIR_LOCAL = WORKSPACE_PARENT_DIR / "receptors"
CONFIG_DIR_LOCAL = WORKSPACE_PARENT_DIR / "configs"
LIGAND_PREP_DIR_LOCAL = WORKSPACE_PARENT_DIR / "prepared_ligands"
LIGAND_UPLOAD_TEMP_DIR = WORKSPACE_PARENT_DIR / "uploaded_ligands_temp"
ZIP_EXTRACT_DIR_LOCAL = WORKSPACE_PARENT_DIR / "zip_extracted_ligands"
DOCKING_OUTPUT_DIR_LOCAL = Path("./autodock_outputs") # Separate from workspace for easier access to final results

# --- Helper Functions ---

def initialize_directories():
    """Creates necessary local directories."""
    dirs_to_create = [
        WORKSPACE_PARENT_DIR, HELPER_SCRIPTS_DIR_LOCAL, RECEPTOR_DIR_LOCAL,
        CONFIG_DIR_LOCAL, LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR,
        ZIP_EXTRACT_DIR_LOCAL, DOCKING_OUTPUT_DIR_LOCAL
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    # Clean specific temp directories that might contain old data from previous runs
    # shutil.rmtree(LIGAND_UPLOAD_TEMP_DIR, ignore_errors=True)
    # LIGAND_UPLOAD_TEMP_DIR.mkdir(exist_ok=True)
    # shutil.rmtree(ZIP_EXTRACT_DIR_LOCAL, ignore_errors=True)
    # ZIP_EXTRACT_DIR_LOCAL.mkdir(exist_ok=True)


def download_file_from_github(relative_path_on_github, local_filename, local_save_dir):
    """Downloads a file from the configured GitHub repo to a local path."""
    full_url = urljoin(BASE_GITHUB_URL, relative_path_on_github)
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
        st.sidebar.info(f"Made {Path(filepath_str).name} executable.")
        return True
    except Exception as e:
        st.sidebar.error(f"Error making {Path(filepath_str).name} executable: {e}")
        return False

def check_vina_binary():
    """Checks for Vina binary and its permissions."""
    st.sidebar.subheader("Vina Binary Status")
    st.sidebar.write(f"Expected Vina path: `{VINA_PATH_LOCAL.resolve()}`")
    if not VINA_PATH_LOCAL.exists():
        st.sidebar.error(
            f"AutoDock Vina executable NOT FOUND at the expected location. "
            f"Please ensure `{VINA_EXECUTABLE_NAME}` is in the `{VINA_DIR_LOCAL}` directory "
            f"of your app's repository."
        )
        return False

    st.sidebar.success("Vina binary found.")
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
            "**Action Required:** If this fails, please set permissions manually in your Git repository: "
            "`git add --chmod=+x vina/your_vina_executable_name`, commit, and push."
        )
        return False

# --- Ligand Preparation Backend Functions ---

def get_smiles_from_pubchem_inchikey(inchikey_str):
    """Fetches SMILES string from PubChem using InChIKey."""
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

def run_ligand_prep_script(script_local_path, script_args, process_name, ligand_name_for_log):
    """Runs a Python ligand preparation script (scrub.py or mk_prepare_ligand.py)."""
    if not script_local_path or not os.path.exists(script_local_path):
        st.error(f"{process_name} script not found at {script_local_path}.")
        return False

    command = ["python", script_local_path] + script_args
    try:
        st.info(f"Running {process_name} for {ligand_name_for_log}...")
        # st.code(" ".join(command)) # For debugging
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd=WORKSPACE_PARENT_DIR)
        with st.expander(f"{process_name} output for {ligand_name_for_log} (stdout)", expanded=False):
            st.text(result.stdout if result.stdout else "No standard output.")
        if result.stderr:
             with st.expander(f"{process_name} output for {ligand_name_for_log} (stderr) - check for errors", expanded=True):
                st.text(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error during {process_name} for {ligand_name_for_log}:")
        st.error(f"Command: `{' '.join(e.cmd)}`")
        st.error(f"Return code: {e.returncode}")
        st.error(f"Stdout: {e.stdout}")
        st.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred running {process_name} for {ligand_name_for_log}: {e}")
        return False

def convert_smiles_to_pdbqt(smiles_str, ligand_name_base, output_dir_path, ph_val, skip_taut, skip_acidbase, scrub_script_path, mk_prepare_script_path):
    """Converts a single SMILES string to a PDBQT file."""
    output_dir_path.mkdir(parents=True, exist_ok=True)
    sdf_path = output_dir_path / f"{ligand_name_base}_scrubbed.sdf"
    pdbqt_path = output_dir_path / f"{ligand_name_base}.pdbqt"

    scrub_options = ["--ph", str(ph_val)]
    if skip_taut: scrub_options.append("--skip_tautomer")
    if skip_acidbase: scrub_options.append("--skip_acidbase")

    scrub_args = [smiles_str, "-o", str(sdf_path)] + scrub_options
    if not run_ligand_prep_script(scrub_script_path, scrub_args, "scrub.py", ligand_name_base):
        return None

    mk_prepare_args = ["-i", str(sdf_path), "-o", str(pdbqt_path)]
    if not run_ligand_prep_script(mk_prepare_script_path, mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base):
        return None

    return str(pdbqt_path) if pdbqt_path.exists() else None

def convert_ligand_file_to_pdbqt(input_ligand_file_path, ligand_name_base, output_dir_path, mk_prepare_script_path):
    """Converts an input ligand file (SDF, MOL2, etc.) to PDBQT."""
    output_dir_path.mkdir(parents=True, exist_ok=True)
    pdbqt_path = output_dir_path / f"{ligand_name_base}.pdbqt"
    mk_prepare_args = ["-i", str(input_ligand_file_path), "-o", str(pdbqt_path)]
    if not run_ligand_prep_script(mk_prepare_script_path, mk_prepare_args, "mk_prepare_ligand.py", ligand_name_base):
        return None
    return str(pdbqt_path) if pdbqt_path.exists() else None


# --- Main Application ---
st.set_page_config(page_title="Ensemble AutoDock Vina", layout="wide")
st.title(f"🧬 Ensemble AutoDock Vina Docking App (v{APP_VERSION})")

# Initialize local directories
initialize_directories()

# --- Sidebar for Setup & Global Parameters ---
st.sidebar.header("Global Setup & Parameters")

# Download helper scripts
scrub_py_local = download_file_from_github(SCRUB_PY_REL_PATH, "scrub.py", HELPER_SCRIPTS_DIR_LOCAL)
mk_prepare_ligand_py_local = download_file_from_github(MK_PREPARE_LIGAND_PY_REL_PATH, "mk_prepare_ligand.py", HELPER_SCRIPTS_DIR_LOCAL)
vina_screening_pl_local = download_file_from_github(VINA_SCREENING_PL_REL_PATH, "Vina_screening.pl", HELPER_SCRIPTS_DIR_LOCAL)

if vina_screening_pl_local:
    make_file_executable(vina_screening_pl_local)

# Check Vina binary
vina_ready = check_vina_binary()

# Fetch Receptors
st.sidebar.subheader("Receptor Setup")
receptor_file_names_input = st.sidebar.text_area(
    f"Receptor PDBQT filenames (one per line, from .../ensemble-docking/{RECEPTOR_SUBDIR_GH})",
    help="Enter the exact filenames of your receptor PDBQT files located in the GitHub repository."
    # Example: "proteinA.pdbqt\nproteinB.pdbqt"
)
fetched_receptor_paths = []
if receptor_file_names_input.strip() and st.sidebar.button("Fetch Receptors", key="fetch_receptors"):
    receptor_names = [name.strip() for name in receptor_file_names_input.splitlines() if name.strip()]
    with st.spinner("Fetching receptor files..."):
        for r_name in receptor_names:
            path = download_file_from_github(RECEPTOR_SUBDIR_GH + r_name, r_name, RECEPTOR_DIR_LOCAL)
            if path: fetched_receptor_paths.append(path)
    if fetched_receptor_paths:
        st.sidebar.success(f"Successfully fetched {len(fetched_receptor_paths)} receptors.")
        st.session_state.fetched_receptor_paths = fetched_receptor_paths # Store in session state
    else:
        st.sidebar.error("No receptors fetched. Check filenames or GitHub path.")

if 'fetched_receptor_paths' in st.session_state and st.session_state.fetched_receptor_paths:
    st.sidebar.markdown(f"**{len(st.session_state.fetched_receptor_paths)} Receptors Ready:**")
    for p in st.session_state.fetched_receptor_paths:
        st.sidebar.caption(f"- {Path(p).name}")

# Fetch Configs
st.sidebar.subheader("Vina Config File Setup")
config_file_names_input = st.sidebar.text_area(
    f"Vina config filenames (one per line, from .../ensemble-docking/{CONFIG_SUBDIR_GH})",
    help="Enter exact filenames of Vina config TXT files. These define search space, etc."
    # Example: "config_proteinA.txt\nconfig_general.txt"
)
fetched_config_paths = []
if config_file_names_input.strip() and st.sidebar.button("Fetch Config Files", key="fetch_configs"):
    config_names = [name.strip() for name in config_file_names_input.splitlines() if name.strip()]
    with st.spinner("Fetching config files..."):
        for c_name in config_names:
            path = download_file_from_github(CONFIG_SUBDIR_GH + c_name, c_name, CONFIG_DIR_LOCAL)
            if path: fetched_config_paths.append(path)
    if fetched_config_paths:
        st.sidebar.success(f"Successfully fetched {len(fetched_config_paths)} config files.")
        st.session_state.fetched_config_paths = fetched_config_paths # Store in session state
    else:
        st.sidebar.error("No config files fetched. Check filenames or GitHub path.")

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

prepared_ligand_pdbqt_paths = [] # This will hold paths to PDBQT files ready for docking

# Common SMILES preparation parameters
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
        if inchikey_or_smiles_val and single_ligand_name:
            actual_smiles = ""
            if use_inchikey_lookup:
                with st.spinner(f"Fetching SMILES for InChIKey {inchikey_or_smiles_val}..."):
                    actual_smiles = get_smiles_from_pubchem_inchikey(inchikey_or_smiles_val)
                if actual_smiles: st.info(f"Fetched SMILES: {actual_smiles}")
                else: st.error("Could not fetch SMILES. Please provide SMILES directly.");
            else:
                actual_smiles = inchikey_or_smiles_val

            if actual_smiles and scrub_py_local and mk_prepare_ligand_py_local:
                pdbqt_p = convert_smiles_to_pdbqt(actual_smiles, single_ligand_name, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, scrub_py_local, mk_prepare_ligand_py_local)
                if pdbqt_p:
                    prepared_ligand_pdbqt_paths.append(pdbqt_p)
                    st.success(f"Ligand '{single_ligand_name}' prepared: `{pdbqt_p}`")
            elif not (scrub_py_local and mk_prepare_ligand_py_local):
                st.error("Ligand preparation scripts (scrub.py, mk_prepare_ligand.py) are missing.")
        else:
            st.warning("Please provide both SMILES/InChIKey and a ligand name.")


elif ligand_input_method == "SMILES File (.txt)":
    uploaded_smiles_file = st.file_uploader("Upload a .txt file (one SMILES per line):", type="txt", key="smiles_file_uploader")
    if uploaded_smiles_file and st.button("Prepare Ligands from SMILES File", key="prep_smiles_file_btn"):
        smiles_list = [line.strip() for line in uploaded_smiles_file.read().decode().splitlines() if line.strip()]
        if not smiles_list:
            st.warning("No SMILES strings found in the uploaded file.")
        elif not (scrub_py_local and mk_prepare_ligand_py_local):
            st.error("Ligand preparation scripts (scrub.py, mk_prepare_ligand.py) are missing.")
        else:
            st.info(f"Found {len(smiles_list)} SMILES string(s). Preparing...")
            progress_bar_smiles = st.progress(0)
            for i, smiles_str in enumerate(smiles_list):
                lig_name = f"ligand_file_{i+1}"
                pdbqt_p = convert_smiles_to_pdbqt(smiles_str, lig_name, LIGAND_PREP_DIR_LOCAL, g_ph_val, g_skip_tautomer, g_skip_acidbase, scrub_py_local, mk_prepare_ligand_py_local)
                if pdbqt_p: prepared_ligand_pdbqt_paths.append(pdbqt_p)
                progress_bar_smiles.progress((i + 1) / len(smiles_list))
            st.success(f"Finished processing SMILES file. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")


elif ligand_input_method == "PDBQT File(s)":
    uploaded_pdbqt_files = st.file_uploader("Upload PDBQT ligand file(s):", type="pdbqt", accept_multiple_files=True, key="pdbqt_uploader")
    if uploaded_pdbqt_files:
        for up_file in uploaded_pdbqt_files:
            # Save to prepared ligands directory to have a consistent location
            dest_path = LIGAND_PREP_DIR_LOCAL / up_file.name
            with open(dest_path, "wb") as f:
                f.write(up_file.getbuffer())
            prepared_ligand_pdbqt_paths.append(str(dest_path))
        st.info(f"Using {len(prepared_ligand_pdbqt_paths)} uploaded PDBQT file(s).")


elif ligand_input_method == "Other Ligand File(s) (e.g., SDF, MOL2)":
    uploaded_other_files = st.file_uploader("Upload other ligand format file(s) (e.g., SDF, MOL2):", accept_multiple_files=True, key="other_lig_uploader")
    if uploaded_other_files and st.button("Convert Uploaded Ligand File(s) to PDBQT", key="convert_other_btn"):
        if not mk_prepare_ligand_py_local:
            st.error("mk_prepare_ligand.py script is missing, cannot convert.")
        else:
            st.info(f"Processing {len(uploaded_other_files)} uploaded file(s) for conversion...")
            progress_bar_other = st.progress(0)
            for i, up_file in enumerate(uploaded_other_files):
                # Save to a temporary spot for conversion
                temp_save_path = LIGAND_UPLOAD_TEMP_DIR / up_file.name
                with open(temp_save_path, "wb") as f:
                    f.write(up_file.getbuffer())

                lig_name_base = Path(up_file.name).stem
                pdbqt_p = convert_ligand_file_to_pdbqt(temp_save_path, lig_name_base, LIGAND_PREP_DIR_LOCAL, mk_prepare_ligand_py_local)
                if pdbqt_p: prepared_ligand_pdbqt_paths.append(pdbqt_p)
                progress_bar_other.progress((i + 1) / len(uploaded_other_files))
            st.success(f"Finished conversion. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")

elif ligand_input_method == "ZIP Archive of Ligands":
    uploaded_zip_file = st.file_uploader("Upload a ZIP archive containing ligand files:", type="zip", key="zip_uploader")
    if uploaded_zip_file and st.button("Process Ligands from ZIP Archive", key="process_zip_btn"):
        if not mk_prepare_ligand_py_local: # Needed if non-PDBQT files are in ZIP
             st.warning("mk_prepare_ligand.py script is missing. Non-PDBQT files in ZIP may not be converted.")

        # Clean and recreate extraction directory
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
                    # Copy to prepared ligands dir
                    dest_path = LIGAND_PREP_DIR_LOCAL / item_path.name
                    shutil.copy(item_path, dest_path)
                    prepared_ligand_pdbqt_paths.append(str(dest_path))
                elif mk_prepare_ligand_py_local: # Attempt conversion for other files
                    pdbqt_p = convert_ligand_file_to_pdbqt(item_path, lig_name_base, LIGAND_PREP_DIR_LOCAL, mk_prepare_ligand_py_local)
                    if pdbqt_p: prepared_ligand_pdbqt_paths.append(pdbqt_p)
                else:
                    st.warning(f"Skipping conversion of {item_path.name} (mk_prepare_ligand.py missing or error).")
                progress_bar_zip.progress((i + 1) / len(files_in_zip))
            st.success(f"Finished processing ZIP. Prepared {len(prepared_ligand_pdbqt_paths)} PDBQT files.")

if prepared_ligand_pdbqt_paths:
    st.success(f"**Total {len(prepared_ligand_pdbqt_paths)} ligand(s) ready for docking.**")
    with st.expander("View Ready Ligands"):
        for p_path in prepared_ligand_pdbqt_paths:
            st.caption(f"- `{Path(p_path).name}` (at `{p_path}`)")
    # Store in session state if needed for persistence across reruns before docking
    st.session_state.prepared_ligand_pdbqt_paths = prepared_ligand_pdbqt_paths


# --- Docking Execution Section ---
st.header("🚀 Docking Execution")

# Retrieve from session state or use current if just prepared
final_ligand_list_for_docking = st.session_state.get('prepared_ligand_pdbqt_paths', prepared_ligand_pdbqt_paths)
current_receptors = st.session_state.get('fetched_receptor_paths', [])
current_configs = st.session_state.get('fetched_config_paths', [])

if not vina_ready:
    st.error("Vina executable is not properly set up. Cannot run docking.")
elif not current_receptors:
    st.warning("No receptors fetched/selected. Please fetch receptors from the sidebar.")
elif not final_ligand_list_for_docking:
    st.warning("No ligands prepared or uploaded. Please prepare/upload ligands first.")
elif not current_configs:
    st.warning("No Vina configuration files fetched/selected. Please fetch config files from the sidebar.")
else:
    st.info(f"Ready to dock {len(final_ligand_list_for_docking)} ligand(s) against {len(current_receptors)} receptor(s) using {len(current_configs)} configuration(s).")

    use_vina_screening_perl = False
    if len(final_ligand_list_for_docking) > 1 and vina_screening_pl_local and os.path.exists(vina_screening_pl_local):
        use_vina_screening_perl = st.checkbox(
            "Use `Vina_screening.pl` for docking multiple ligands? (Recommended for many ligands; processes one receptor at a time)",
            value=True, key="use_perl_script_cb"
        )

    if st.button("Start Docking Run", key="start_docking_main_btn", type="primary"):
        st.markdown("---")
        DOCKING_OUTPUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

        if use_vina_screening_perl:
            st.subheader("Docking via `Vina_screening.pl`")
            if not (vina_screening_pl_local and os.path.exists(vina_screening_pl_local) and os.access(vina_screening_pl_local, os.X_OK)):
                st.error("`Vina_screening.pl` script is not available or not executable.")
            else:
                # Create ligand_list.txt for the Perl script
                ligand_list_file_for_perl = WORKSPACE_PARENT_DIR / "ligands_for_perl.txt"
                with open(ligand_list_file_for_perl, "w") as f:
                    for p_path_str in final_ligand_list_for_docking:
                        f.write(str(Path(p_path_str).resolve()) + "\n") # Absolute paths
                st.info(f"Created ligand list for Perl script: `{ligand_list_file_for_perl}`")

                overall_docking_progress = st.progress(0)
                for i, receptor_path_str in enumerate(current_receptors):
                    receptor_file = Path(receptor_path_str)
                    protein_base = receptor_file.stem
                    st.markdown(f"#### Processing Receptor: `{receptor_file.name}`")

                    # Vina_screening.pl typically expects receptor and config in CWD or specific relative paths.
                    # It also usually expects a config file named like the protein (e.g., protein_base.txt).
                    # We will run it from WORKSPACE_PARENT_DIR and copy necessary files there temporarily.

                    # Copy receptor to CWD for the script
                    temp_receptor_for_perl = WORKSPACE_PARENT_DIR / receptor_file.name
                    shutil.copy(receptor_file, temp_receptor_for_perl)

                    # Find and copy the corresponding config file.
                    # Assumes config file is named like <protein_base>.txt or config_<protein_base>.txt
                    # Or, if only one config is fetched, use that.
                    # This logic might need refinement based on how Vina_screening.pl finds its config.
                    config_for_this_protein = None
                    if len(current_configs) == 1:
                        config_for_this_protein = Path(current_configs[0])
                    else:
                        for cfg_path_str in current_configs:
                            cfg_file = Path(cfg_path_str)
                            if protein_base in cfg_file.name: # Simple matching
                                config_for_this_protein = cfg_file
                                break
                    
                    temp_config_for_perl = None
                    if config_for_this_protein:
                        # Vina_screening.pl might expect a specific config name, e.g., <protein_base>.txt
                        expected_config_name_for_perl = f"{protein_base}.txt" # Or check script's expectation
                        temp_config_for_perl = WORKSPACE_PARENT_DIR / expected_config_name_for_perl
                        shutil.copy(config_for_this_protein, temp_config_for_perl)
                        st.info(f"Using config `{config_for_this_protein.name}` (as `{expected_config_name_for_perl}`) for `{receptor_file.name}`.")
                    else:
                        st.warning(f"Could not determine a specific config for `{receptor_file.name}`. "
                                   f"Perl script might use defaults or fail if config is mandatory.")

                    # Prepare output directory for this protein (as Vina_screening.pl might create it)
                    protein_specific_output_dir = DOCKING_OUTPUT_DIR_LOCAL / protein_base
                    protein_specific_output_dir.mkdir(parents=True, exist_ok=True) # Script might expect it
                    
                    # Command: echo /path/to/ligands_list.txt | perl /path/to/Vina_screening.pl protein_base_name
                    # The script is run with CWD = WORKSPACE_PARENT_DIR
                    # The protein_base_name argument tells the script which receptor/config to use (e.g., protein_base.pdbqt, protein_base.txt)
                    cmd_perl = [
                        "perl",
                        str(Path(vina_screening_pl_local).resolve()),
                        protein_base # Argument to Perl script
                    ]
                    st.code(f"echo {str(ligand_list_file_for_perl.resolve())} | {' '.join(cmd_perl)}")

                    try:
                        process = subprocess.Popen(
                            cmd_perl,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd=WORKSPACE_PARENT_DIR # Run where temp receptor/config are
                        )
                        # Pass the path of the ligand list file to stdin of the Perl script
                        stdout_perl, stderr_perl = process.communicate(input=str(ligand_list_file_for_perl.resolve()) + "\n")

                        if process.returncode == 0:
                            st.success(f"`Vina_screening.pl` completed for `{receptor_file.name}`.")
                        else:
                            st.error(f"`Vina_screening.pl` failed for `{receptor_file.name}` (Return code: {process.returncode}).")
                        with st.expander(f"Perl script STDOUT for {receptor_file.name}", expanded=False):
                            st.text(stdout_perl if stdout_perl else "No standard output.")
                        with st.expander(f"Perl script STDERR for {receptor_file.name}", expanded=True):
                            st.text(stderr_perl if stderr_perl else "No standard error output.")
                        st.info(f"Docking outputs for `{receptor_file.name}` should be in `{DOCKING_OUTPUT_DIR_LOCAL}/{protein_base}` (as created by the Perl script).")

                    except Exception as e_perl:
                        st.error(f"Error executing `Vina_screening.pl` for `{receptor_file.name}`: {e_perl}")
                    finally:
                        # Clean up temporary receptor/config copied for Perl script
                        if temp_receptor_for_perl and temp_receptor_for_perl.exists(): temp_receptor_for_perl.unlink()
                        if temp_config_for_perl and temp_config_for_perl.exists(): temp_config_for_perl.unlink()
                    
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

                        # Define output file names based on inputs
                        output_base = f"{receptor_file.stem}_{ligand_file.stem}_{config_file.stem}"
                        output_pdbqt_docked = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base}_out.pdbqt"
                        output_log_file = DOCKING_OUTPUT_DIR_LOCAL / f"{output_base}_log.txt"

                        cmd_vina_direct = [
                            str(VINA_PATH_LOCAL.resolve()),
                            "--receptor", str(receptor_file.resolve()),
                            "--ligand", str(ligand_file.resolve()),
                            "--config", str(config_file.resolve()),
                            "--out", str(output_pdbqt_docked.resolve()),
                            "--log", str(output_log_file.resolve())
                            # Add other Vina params like --exhaustiveness if not in config
                        ]
                        st.code(" ".join(cmd_vina_direct))

                        try:
                            vina_run_result = subprocess.run(cmd_vina_direct, capture_output=True, text=True, check=True, cwd=WORKSPACE_PARENT_DIR)
                            st.success("Vina docking job completed successfully!")
                            with st.expander("Vina STDOUT", expanded=False):
                                st.text(vina_run_result.stdout if vina_run_result.stdout else "No standard output.")
                            if vina_run_result.stderr:
                                with st.expander("Vina STDERR (check for warnings)", expanded=True):
                                    st.text(vina_run_result.stderr)
                            
                            # Provide download links for the results
                            if output_pdbqt_docked.exists():
                                with open(output_pdbqt_docked, "rb") as fp:
                                    st.download_button(label=f"Download Docked PDBQT ({output_pdbqt_docked.name})", data=fp, file_name=output_pdbqt_docked.name, mime="chemical/x-pdbqt")
                            if output_log_file.exists():
                                with open(output_log_file, "r", encoding='utf-8') as fp:
                                     st.download_button(label=f"Download Log File ({output_log_file.name})", data=fp.read(), file_name=output_log_file.name, mime="text/plain")

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
st.caption("Developed with Streamlit. AutoDock Vina for docking simulations.")
