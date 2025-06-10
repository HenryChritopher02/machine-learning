import streamlit as st
import subprocess
import os
import zipfile
import shutil
from pathlib import Path
import sys
import pandas as pd
from rdkit import Chem

# Import from local utility files
from utils.paths import (
    APP_VERSION, BASE_GITHUB_URL_FOR_DATA, GH_API_BASE_URL, GH_OWNER, GH_REPO, GH_BRANCH,
    GH_ENSEMBLE_DOCKING_ROOT_PATH, RECEPTOR_SUBDIR_GH, CONFIG_SUBDIR_GH,
    APP_ROOT, ENSEMBLE_DOCKING_DIR_LOCAL, LIGAND_PREPROCESSING_SUBDIR_LOCAL,
    SCRUB_PY_LOCAL_PATH, MK_PREPARE_LIGAND_PY_LOCAL_PATH, VINA_SCREENING_PL_LOCAL_PATH,
    VINA_DIR_LOCAL, VINA_EXECUTABLE_NAME, VINA_PATH_LOCAL,
    WORKSPACE_PARENT_DIR, RECEPTOR_DIR_LOCAL, CONFIG_DIR_LOCAL,
    LIGAND_PREP_DIR_LOCAL, LIGAND_UPLOAD_TEMP_DIR, ZIP_EXTRACT_DIR_LOCAL,
    DOCKING_OUTPUT_DIR_LOCAL
)
from utils.app_utils import (
    standardize_smiles_rdkit, initialize_directories, list_files_from_github_repo_dir,
    download_file_from_github, make_file_executable, check_script_exists,
    check_vina_binary, get_smiles_from_pubchem_inchikey, run_ligand_prep_script,
    convert_smiles_to_pdbqt, convert_ligand_file_to_pdbqt,
    find_paired_config_for_protein, convert_df_to_csv, parse_score_from_pdbqt
)
from utils.prediction_utils import (
    calculate_mordred_descriptors, calculate_ecfp4_fingerprints,
    load_and_prepare_train_data_desc, align_input_descriptors, generate_pca_plot,
    run_original_gnn_prediction, run_hybrid_gnn_prediction
)


def display_ensemble_docking_procedure():
    st.header(f"Ensemble Docking AutoDock Vina 1.2.5 (v{APP_VERSION})")
    st.markdown("---")

    if 'prepared_ligand_details_list' not in st.session_state:
        st.session_state.prepared_ligand_details_list = []
    if 'docking_run_outputs' not in st.session_state:
        st.session_state.docking_run_outputs = []
    if 'invalid_smiles_during_standardization' not in st.session_state:
        st.session_state.invalid_smiles_during_standardization = []
    if 'fetched_receptor_paths' not in st.session_state:
        st.session_state.fetched_receptor_paths = []
    if 'fetched_config_paths' not in st.session_state:
        st.session_state.fetched_config_paths = []

    with st.sidebar:
        st.header("⚙️ Docking Setup")
        st.caption("Core Components Status:")
        scrub_py_ok = check_script_exists(SCRUB_PY_LOCAL_PATH, "scrub.py")
        if scrub_py_ok: make_file_executable(str(SCRUB_PY_LOCAL_PATH))
        mk_prepare_ligand_py_ok = check_script_exists(MK_PREPARE_LIGAND_PY_LOCAL_PATH, "mk_prepare_ligand.py")
        if mk_prepare_ligand_py_ok: make_file_executable(str(MK_PREPARE_LIGAND_PY_LOCAL_PATH))
        vina_screening_pl_ok = check_script_exists(VINA_SCREENING_PL_LOCAL_PATH, "Vina_screening.pl", is_critical=True)
        if vina_screening_pl_ok: make_file_executable(str(VINA_SCREENING_PL_LOCAL_PATH))

        vina_ready = check_vina_binary(show_success=True)
        st.markdown("---")

        st.subheader("Receptor(s) and Config(s)")
        
        receptor_fetch_method = st.radio(
            "Receptor/Config Source:",
            ("All from GitHub", "Specify from GitHub", "Upload from Computer"),
            key="receptor_fetch_method_dockpage",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if receptor_fetch_method == "All from GitHub":
            st.caption("Config files (.txt) with the same name as receptors will be fetched automatically.")
            receptor_dir_in_repo = f"{GH_ENSEMBLE_DOCKING_ROOT_PATH}/{RECEPTOR_SUBDIR_GH.strip('/')}"
            if st.button("Fetch All Receptors & Configs", key="fetch_all_receptors_auto_dockpage", help=f"Fetches all .pdbqt from .../{receptor_dir_in_repo} and their corresponding .txt configs."):
                st.session_state.fetched_receptor_paths = []
                st.session_state.fetched_config_paths = []
                with st.spinner(f"Listing .pdbqt files..."):
                    receptor_filenames = list_files_from_github_repo_dir(GH_OWNER, GH_REPO, receptor_dir_in_repo, GH_BRANCH, GH_API_BASE_URL, ".pdbqt")
                if receptor_filenames:
                    st.success(f"Found {len(receptor_filenames)} receptors. Downloading...")
                    temp_receptor_paths, temp_config_paths = [], []
                    with st.spinner(f"Downloading all receptors and matching configs..."):
                        for r_name in receptor_filenames:
                            r_path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"{RECEPTOR_SUBDIR_GH.strip('/')}/{r_name}", r_name, RECEPTOR_DIR_LOCAL)
                            if r_path:
                                temp_receptor_paths.append(r_path)
                                c_name = Path(r_name).stem + ".txt"
                                c_path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"{CONFIG_SUBDIR_GH.strip('/')}/{c_name}", c_name, CONFIG_DIR_LOCAL)
                                if c_path:
                                    temp_config_paths.append(c_path)
                    st.session_state.fetched_receptor_paths = temp_receptor_paths
                    st.session_state.fetched_config_paths = temp_config_paths
                    st.success(f"✅ Fetched {len(st.session_state.fetched_receptor_paths)} receptors and {len(st.session_state.fetched_config_paths)} configs.")
                else:
                    st.warning(f"No .pdbqt files found in GitHub directory.")
        
        elif receptor_fetch_method == "Specify from GitHub":
            st.caption("Config files (.txt) with the same name as receptors will be fetched automatically.")
            receptor_dir_in_repo = f"{GH_ENSEMBLE_DOCKING_ROOT_PATH}/{RECEPTOR_SUBDIR_GH.strip('/')}"
            receptor_names_input = st.text_area(
                "Receptor Filenames (one per line, without extension):",
                key="receptor_filenames_manual_dockpage",
                height=100,
                help=f"Enter names like '3duy', not '3duy.pdbqt'. Fetches from .../{receptor_dir_in_repo}/"
            )
            if st.button("Fetch Specified Receptors & Configs", key="fetch_specified_receptors_dockpage"):
                if receptor_names_input.strip():
                    receptor_base_names = [n.strip() for n in receptor_names_input.splitlines() if n.strip()]
                    st.session_state.fetched_receptor_paths, st.session_state.fetched_config_paths = [], []
                    temp_receptor_paths, temp_config_paths = [], []
                    with st.spinner(f"Downloading {len(receptor_base_names)} specified receptor(s) and config(s)..."):
                        for base_name in receptor_base_names:
                            r_name = base_name + ".pdbqt"
                            r_path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"{RECEPTOR_SUBDIR_GH.strip('/')}/{r_name}", r_name, RECEPTOR_DIR_LOCAL)
                            if r_path:
                                temp_receptor_paths.append(r_path)
                                c_name = base_name + ".txt"
                                c_path = download_file_from_github(BASE_GITHUB_URL_FOR_DATA, f"{CONFIG_SUBDIR_GH.strip('/')}/{c_name}", c_name, CONFIG_DIR_LOCAL)
                                if c_path:
                                    temp_config_paths.append(c_path)
                    if temp_receptor_paths:
                        st.session_state.fetched_receptor_paths = temp_receptor_paths
                        st.session_state.fetched_config_paths = temp_config_paths
                        st.success(f"✅ Fetched {len(st.session_state.fetched_receptor_paths)} receptors and {len(st.session_state.fetched_config_paths)} configs.")
                    else:
                        st.error("No specified receptors downloaded.")
                else:
                    st.warning("Enter receptor filenames.")
        
        elif receptor_fetch_method == "Upload from Computer":
            st.caption("Upload receptor (.pdbqt) and config (.txt) files. Only paired files will be processed.")
            uploaded_receptors = st.file_uploader("Upload Receptor(s) (.pdbqt)", type="pdbqt", accept_multiple_files=True, key="receptor_uploader_local")
            uploaded_configs = st.file_uploader("Upload Config(s) (.txt)", type="txt", accept_multiple_files=True, key="config_uploader_local")

            if st.button("Process Uploaded Files", key="process_local_files_btn"):
                if not uploaded_receptors or not uploaded_configs:
                    st.warning("Please upload both receptor and config files.")
                else:
                    receptor_map = {Path(f.name).stem: f for f in uploaded_receptors}
                    config_map = {Path(f.name).stem: f for f in uploaded_configs}

                    temp_receptor_paths, temp_config_paths, unpaired_receptors = [], [], []

                    with st.spinner("Processing and pairing uploaded files..."):
                        for base_name, receptor_file in receptor_map.items():
                            if base_name in config_map:
                                config_file = config_map[base_name]
                                
                                r_dest_path = RECEPTOR_DIR_LOCAL / receptor_file.name
                                with open(r_dest_path, "wb") as f:
                                    f.write(receptor_file.getbuffer())
                                temp_receptor_paths.append(str(r_dest_path))
                                
                                c_dest_path = CONFIG_DIR_LOCAL / config_file.name
                                with open(c_dest_path, "wb") as f:
                                    f.write(config_file.getbuffer())
                                temp_config_paths.append(str(c_dest_path))
                            else:
                                unpaired_receptors.append(receptor_file.name)

                    st.session_state.fetched_receptor_paths = temp_receptor_paths
                    st.session_state.fetched_config_paths = temp_config_paths

                    if temp_receptor_paths:
                        st.success(f"✅ Processed and paired {len(temp_receptor_paths)} receptor/config file(s).")
                    if unpaired_receptors:
                        st.warning("The following receptors were ignored (missing a matching config file):")
                        for name in unpaired_receptors:
                            st.caption(f"- {name}")
                    if not temp_receptor_paths and uploaded_receptors:
                         st.error("No valid receptor-config pairs were found in the uploaded files.")

        if st.session_state.get('fetched_receptor_paths'):
            exp = st.expander(f"**{len(st.session_state.fetched_receptor_paths)} Receptor(s) Ready**", expanded=False)
            for p_str in st.session_state.fetched_receptor_paths:
                exp.caption(f"✔️ {Path(p_str).name}")
        
        if st.session_state.get('fetched_config_paths'):
            exp = st.expander(f"**{len(st.session_state.fetched_config_paths)} Config(s) Ready**", expanded=False)
            for p_str in st.session_state.fetched_config_paths:
                exp.caption(f"✔️ {Path(p_str).name}")
        
        st.markdown("---")

        if st.button("Clear All Prepared Ligands", key="clear_ligands_btn"):
            st.session_state.prepared_ligand_details_list = []
            st.session_state.invalid_smiles_during_standardization = []
            st.success("All prepared ligands and invalid SMILES records have been cleared.")
        st.markdown("---")


    st.subheader("🔬 Ligand Input & Preparation")
    ligand_input_method = st.radio(
        "Choose ligand input method:",
        ("SMILES String", "SMILES File (.txt)", "PDBQT File(s)", "Other Ligand File(s)", "ZIP Archive"),
        key="ligand_method_radio_mainpage", horizontal=True)

    g_ph_val, g_skip_tautomer, g_skip_acidbase = 7.4, False, False
    if ligand_input_method in ["SMILES String", "SMILES File (.txt)"]:
        with st.expander("SMILES Protonation Options (via scrub.py)", expanded=False):
            g_ph_val = st.number_input("pH for scrub.py", value=7.4, key="g_ph_val_main_ph", format="%.1f")
            g_skip_tautomer = st.checkbox("scrub.py: Skip tautomers", key="g_skip_taut_main_taut")
            g_skip_acidbase = st.checkbox("scrub.py: Skip protomers", key="g_skip_ab_main_ab")

    if ligand_input_method == "SMILES String":
        inchikey_or_smiles_val = st.text_input("InChIKey or SMILES string:", key="smiles_input_main_val_lig")
        use_inchikey = st.checkbox("Input is InChIKey (will fetch SMILES from PubChem)", value=False, key="use_inchikey_main_cb_lig")
        lig_name_base_input = st.text_input("Ligand Base Name:", value="ligand_smiles", key="lig_name_main_name_lig")
        if st.button("Prepare & Add This SMILES Ligand", key="prep_add_smiles_main_btn_lig"):
            _current_batch_processed_details = []
            st.session_state.invalid_smiles_during_standardization = []

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
                            detail['id'] = standardized_s
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
            elif not (not inchikey_or_smiles_val.strip() or not lig_name_base_input.strip() or (not scrub_py_ok or not mk_prepare_ligand_py_ok) or (use_inchikey and not actual_smiles_to_process)) and not _current_batch_processed_details:
                st.warning("No ligands were successfully prepared and added in this step.")


    elif ligand_input_method == "SMILES File (.txt)":
        uploaded_smiles_file = st.file_uploader("Upload SMILES file (.txt, one SMILES per line):", type="txt", key="smiles_uploader_main_file_lig")
        if st.button("Process & Add SMILES File", key="process_add_smiles_file_btn"):
            _current_batch_processed_details = []
            st.session_state.invalid_smiles_during_standardization = []

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
                                    detail['id'] = standardized_s
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
            elif uploaded_smiles_file and not _current_batch_processed_details:
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
                except zipfile.BadZipFile:
                    st.error("Uploaded file is not a valid ZIP archive.")
                except Exception as e:
                    st.error(f"Error processing ZIP archive: {e}")
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
    
    if st.button("Start Docking Run", key="start_docking_main_btn_main_page", type="primary"):
        final_ligand_details_list_for_run = list(st.session_state.get('prepared_ligand_details_list', []))
        current_receptors_for_run = st.session_state.get('fetched_receptor_paths', [])
        current_configs_for_run = st.session_state.get('fetched_config_paths', [])

        # --- MODIFIED: Pre-run checks simplified and made mandatory for Perl script ---
        if not vina_ready:
            st.error("Vina executable not set up. Cannot start docking.")
        elif not vina_screening_pl_ok or not VINA_SCREENING_PL_LOCAL_PATH.exists() or not os.access(VINA_SCREENING_PL_LOCAL_PATH, os.X_OK):
            st.error(f"The required docking script (Vina_screening.pl) is not available or not executable. Cannot start docking.")
        elif not current_receptors_for_run:
            st.warning("No receptors available for docking. Please fetch/upload receptors.")
        elif not final_ligand_details_list_for_run:
            st.warning("No ligands prepared and added for docking. Please prepare ligands.")
        elif not current_configs_for_run:
            st.warning("No Vina configs available for docking. Please fetch/upload configs.")
        else:
            # --- MODIFIED: Logic now ONLY uses the Perl script path ---
            st.info(f"Starting docking for {len(final_ligand_details_list_for_run)} ligand(s) vs {len(current_receptors_for_run)} receptor(s).")
            st.session_state.docking_run_outputs = []
            DOCKING_OUTPUT_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

            ligand_list_file_for_perl = WORKSPACE_PARENT_DIR / "ligands_for_perl.txt"
            with open(ligand_list_file_for_perl, "w") as f_list:
                for lig_detail in final_ligand_details_list_for_run:
                    f_list.write(str(Path(lig_detail['pdbqt_path']).resolve()) + "\n")
            
            overall_docking_progress = st.progress(0)
            receptors_processed_count = 0
            skipped_receptor_count = 0
            
            for i_rec, receptor_path_str in enumerate(current_receptors_for_run):
                receptor_file = Path(receptor_path_str)
                protein_base = receptor_file.stem
                st.markdown(f"--- \n**Receptor: `{receptor_file.name}`**")
                
                config_to_use = find_paired_config_for_protein(protein_base, current_configs_for_run)

                if not config_to_use:
                    st.warning(f"No paired config for `{receptor_file.name}`. Skipping.")
                    skipped_receptor_count +=1
                    overall_docking_progress.progress((i_rec + 1) / len(current_receptors_for_run))
                    continue

                temp_receptor_path_for_perl = WORKSPACE_PARENT_DIR / receptor_file.name
                shutil.copy(receptor_file, temp_receptor_path_for_perl)
                
                cmd_perl = ["perl", str(VINA_SCREENING_PL_LOCAL_PATH.resolve()),
                            str(VINA_PATH_LOCAL.resolve()),
                            str(temp_receptor_path_for_perl.name),
                            str(config_to_use.resolve()),
                            protein_base]
                try:
                    path_to_ligand_list_for_perl_stdin = str(ligand_list_file_for_perl.resolve()) + "\n"
                    proc = subprocess.run(cmd_perl,
                                          input=path_to_ligand_list_for_perl_stdin,
                                          capture_output=True,
                                          text=True,
                                          check=False,
                                          cwd=str(WORKSPACE_PARENT_DIR.resolve()))
                    
                    if proc.stdout.strip():
                        with st.expander(f"Script STDOUT for {protein_base}", expanded=False): st.text(proc.stdout)
                    if proc.stderr.strip():
                        with st.expander(f"Script STDERR for {protein_base}", expanded=True): st.text(proc.stderr)
                    
                    if proc.returncode != 0:
                        st.error(f"Docking script execution failed for `{protein_base}` (RC: {proc.returncode}).")
                    
                    perl_protein_out_dir = WORKSPACE_PARENT_DIR / protein_base
                    if perl_protein_out_dir.is_dir():
                        for lig_detail in final_ligand_details_list_for_run:
                            score = None
                            possible_pdbqt_names = [
                                f"{lig_detail['base_name']}-{protein_base}_out.pdbqt",
                                f"{lig_detail['base_name']}_{protein_base}_out.pdbqt",
                                f"{lig_detail['base_name']}_out.pdbqt",
                                f"{Path(lig_detail['pdbqt_path']).stem}_{protein_base}_out.pdbqt"
                            ]
                            
                            for pdbqt_name_pattern in possible_pdbqt_names:
                                current_expected_pdbqt_file = perl_protein_out_dir / pdbqt_name_pattern
                                if current_expected_pdbqt_file.exists():
                                    score = parse_score_from_pdbqt(str(current_expected_pdbqt_file))
                                    if score is not None:
                                        break
                            
                            if score is not None:
                                st.session_state.docking_run_outputs.append({
                                    "ligand_id": lig_detail["id"],
                                    "ligand_base_name": lig_detail["base_name"],
                                    "protein_stem": protein_base,
                                    "config_stem": config_to_use.stem,
                                    "score": score
                                })
                            else:
                                st.warning(f"Score not obtained for '{lig_detail['base_name']}' with '{protein_base}'.")
                    elif proc.returncode == 0:
                        st.warning(f"Output directory not found for {protein_base}, though script indicated success.")

                except Exception as e_p:
                    st.error(f"Error during script processing for `{protein_base}`: {type(e_p).__name__} - {e_p}")
                finally:
                    if temp_receptor_path_for_perl.exists():
                        temp_receptor_path_for_perl.unlink(missing_ok=True)
                
                receptors_processed_count += 1
                overall_docking_progress.progress((receptors_processed_count + skipped_receptor_count) / len(current_receptors_for_run))
            
            if ligand_list_file_for_perl.exists():
                ligand_list_file_for_perl.unlink(missing_ok=True)
            if skipped_receptor_count > 0:
                st.warning(f"{skipped_receptor_count} receptor(s) were skipped.")
            
            if 'docking_run_outputs' in st.session_state and st.session_state.docking_run_outputs:
                st.markdown("---")
                st.subheader("📊 Docking Results Summary")
                try:
                    df_flat = pd.DataFrame(st.session_state.docking_run_outputs)
                    if not df_flat.empty:
                        df_flat['score'] = pd.to_numeric(df_flat['score'], errors='coerce')
                        df_flat['Protein-Config'] = df_flat['protein_stem'] + '_' + df_flat['config_stem']
                        df_pivot = df_flat.pivot_table(index=['ligand_id', 'ligand_base_name'], columns='Protein-Config', values='score', aggfunc='min')
                        df_summary = df_pivot.reset_index()
                        new_column_names = {'ligand_id': 'SMILES', 'ligand_base_name': 'Ligand Base Name'}
                        
                        for col in df_pivot.columns:
                            new_column_names[col] = f"{col} Score (kcal/mol)"
                        df_summary = df_summary.rename(columns=new_column_names)
                        
                        for col_name in df_summary.columns:
                            if col_name.endswith("Score (kcal/mol)"):
                                df_summary[col_name] = df_summary[col_name].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
                        
                        st.dataframe(df_summary)
                        csv_summary = convert_df_to_csv(df_summary)
                        st.download_button("Download Summary (CSV)", csv_summary, "docking_summary_per_ligand.csv", "text/csv", key="dl_summary_per_ligand_csv")
        
                        score_columns = [col for col in df_summary.columns if col.endswith("Score (kcal/mol)")]
                        scores_for_hybrid = df_summary[['SMILES'] + score_columns].copy()
                        for col in score_columns:
                            scores_for_hybrid[col] = pd.to_numeric(scores_for_hybrid[col], errors='coerce')
                        
                        scores_for_hybrid_cleaned = scores_for_hybrid.dropna(subset=score_columns)
                        num_score_cols_found = len(score_columns)
        
                        if num_score_cols_found == 15:
                            st.session_state.docking_scores_for_hybrid = scores_for_hybrid_cleaned
                            st.success(f"15 docking scores per ligand (for {len(scores_for_hybrid_cleaned)} ligands with complete scores) saved for Hybrid GNN prediction.")
                        elif num_score_cols_found > 0:
                            st.warning(f"Found {num_score_cols_found} docking score columns, but Hybrid GNN model expects 15.")
                            if 'docking_scores_for_hybrid' in st.session_state:
                                del st.session_state.docking_scores_for_hybrid
                        else:
                            st.warning("No docking score columns found. Hybrid GNN input not available.")
                            if 'docking_scores_for_hybrid' in st.session_state:
                                del st.session_state.docking_scores_for_hybrid
                except Exception as e_df:
                    st.error(f"Error processing docking results for summary: {e_df}")
            st.balloons()
            st.header("🏁 Docking Run Finished 🏁")
            st.caption(f"Perl script outputs in `{WORKSPACE_PARENT_DIR.name}/<protein_base_name>/`.")


def display_prediction_model_procedure():
    st.header(f"🧪 Prediction Model (v{APP_VERSION})")
    st.markdown("---")

    model_type = st.radio(
        "Select Prediction Model Type:",
        ("Original GNN Model", "Hybrid GNN Model (Requires 15 Docking Scores)"),
        key="prediction_model_type_selector",
        horizontal=True
    )
    st.markdown("---")

    st.subheader("🧬 Input SMILES for Prediction")
    if 'pred_invalid_smiles' not in st.session_state: st.session_state.pred_invalid_smiles = []
    if 'pred_std_smiles_list' not in st.session_state: st.session_state.pred_std_smiles_list = []
    
    input_smiles_list_for_pred = []
    smiles_input_method_pred = st.radio(
        "Choose SMILES input method:", ("SMILES String", "SMILES File (.txt)"),
        key="smiles_input_method_pred", horizontal=True
    )
    if smiles_input_method_pred == "SMILES String":
        smiles_string_pred = st.text_input("Enter SMILES string(s) (comma-separated for multiple):", key="smiles_string_pred_input")
        if smiles_string_pred.strip():
            input_smiles_list_for_pred.extend([s.strip() for s in smiles_string_pred.split(',') if s.strip()])
    elif smiles_input_method_pred == "SMILES File (.txt)":
        uploaded_smiles_file_pred = st.file_uploader("Upload SMILES file (.txt, one SMILES per line):", type="txt", key="smiles_file_pred_uploader")
        if uploaded_smiles_file_pred:
            try:
                smiles_from_file = uploaded_smiles_file_pred.getvalue().decode("utf-8").splitlines()
                input_smiles_list_for_pred.extend([s.strip() for s in smiles_from_file if s.strip()])
            except Exception as e:
                st.error(f"Error reading SMILES file: {e}")

    if st.button("Process SMILES & Generate Predictions", key="process_all_predictions_btn"):
        if not input_smiles_list_for_pred:
            st.warning("Please provide SMILES input.");
            return

        st.session_state.pred_std_smiles_list = []
        st.session_state.pred_invalid_smiles = []
        with st.spinner("Standardizing SMILES..."):
            for smi in input_smiles_list_for_pred:
                std_smi = standardize_smiles_rdkit(smi, st.session_state.pred_invalid_smiles)
                if std_smi: st.session_state.pred_std_smiles_list.append(std_smi)
        
        if st.session_state.pred_invalid_smiles:
            with st.expander(f"{len(st.session_state.pred_invalid_smiles)} SMILES failed standardization", expanded=True):
                for failed_smi in st.session_state.pred_invalid_smiles: st.caption(f"- {failed_smi}")
        if not st.session_state.pred_std_smiles_list:
            st.error("No valid SMILES available after standardization.");
            return
        st.success(f"Standardized {len(st.session_state.pred_std_smiles_list)} SMILES.")
        
        st.markdown("---");
        st.subheader("🔬 Mordred Descriptors & PCA Analysis")
        mols_for_features = [Chem.MolFromSmiles(s) for s in st.session_state.pred_std_smiles_list]
        mols_for_features = [m for m in mols_for_features if m is not None]

        if not mols_for_features:
            st.warning("Could not generate RDKit molecules for descriptor calculation. Skipping Mordred/PCA.")
        else:
            with st.spinner("Calculating Mordred descriptors..."):
                df_input_mordred = calculate_mordred_descriptors(mols_for_features)
            if df_input_mordred.empty:
                st.warning("Mordred descriptors calculation failed. Skipping PCA.")
            else:
                valid_smiles_for_mordred = st.session_state.pred_std_smiles_list[:len(df_input_mordred)]
                df_input_mordred.index = valid_smiles_for_mordred
                
                with st.spinner("Loading reference data & generating PCA plot..."):
                    df_train_descriptors, train_descriptor_names = load_and_prepare_train_data_desc()
                if df_train_descriptors is not None and train_descriptor_names:
                    df_input_mordred_aligned = align_input_descriptors(df_input_mordred, train_descriptor_names)
                    generate_pca_plot(df_train_descriptors, df_input_mordred_aligned)
                else:
                    st.error("Could not load PCA training data.")
        
        st.markdown("---")
        selected_model_for_prediction = st.session_state.get("prediction_model_type_selector", "Original GNN Model")

        if selected_model_for_prediction == "Original GNN Model":
            st.subheader("📈 Original GNN-based pIC50 Prediction")
            if not st.session_state.pred_std_smiles_list:
                st.warning("No standardized SMILES for Original GNN prediction.")
            else:
                df_gnn_predictions = run_original_gnn_prediction(st.session_state.pred_std_smiles_list)
                if not df_gnn_predictions.empty:
                    st.markdown("#### Original GNN Predicted pIC50 Values:")
                    st.dataframe(df_gnn_predictions)
                else:
                    st.info("Original GNN pIC50 prediction yielded no data or an error occurred.")
        
        elif selected_model_for_prediction == "Hybrid GNN Model (Requires 15 Docking Scores)":
            st.subheader("📈 Hybrid GNN-based pIC50 Prediction")
            if not st.session_state.pred_std_smiles_list:
                st.warning("No standardized SMILES input for Hybrid GNN prediction.")
                return

            docking_scores_data = st.session_state.get('docking_scores_for_hybrid')
            if docking_scores_data is None or docking_scores_data.empty:
                st.error("Docking scores from 'Ensemble Docking' procedure are not available or empty. Cannot run Hybrid GNN. "
                         "Please run Ensemble Docking first for the relevant SMILES and ensure 15 numeric scores are generated and stored.")
                return

            input_smiles_df_for_merge = pd.DataFrame({'SMILES': st.session_state.pred_std_smiles_list})
            merged_data_for_hybrid = pd.merge(input_smiles_df_for_merge, docking_scores_data, on='SMILES', how='inner')

            if merged_data_for_hybrid.empty:
                st.error("None of the current input SMILES have corresponding docking scores from the 'Ensemble Docking' results. "
                         "Please ensure SMILES match and docking was run for them, producing 15 scores.")
                return
            
            aligned_smiles_for_gnn_part = merged_data_for_hybrid['SMILES'].tolist()
            score_cols_for_hybrid_input = [col for col in merged_data_for_hybrid.columns if col.endswith("Score (kcal/mol)")]

            if len(score_cols_for_hybrid_input) != 15:
                st.error(f"Aligned data has {len(score_cols_for_hybrid_input)} docking score columns, but Hybrid GNN expects 15. Columns found: {score_cols_for_hybrid_input}")
                return
            
            aligned_docking_scores_df_for_mlp = merged_data_for_hybrid[score_cols_for_hybrid_input]
            
            for col in score_cols_for_hybrid_input:
                aligned_docking_scores_df_for_mlp[col] = pd.to_numeric(aligned_docking_scores_df_for_mlp[col], errors='coerce')
            
            if aligned_docking_scores_df_for_mlp.isnull().any().any():
                st.error("Some docking scores for the selected SMILES are still not valid numbers (NaN) after alignment. Hybrid GNN cannot proceed.")
                st.dataframe(aligned_docking_scores_df_for_mlp[aligned_docking_scores_df_for_mlp.isnull().any(axis=1)])
                return

            st.info(f"Proceeding with Hybrid GNN for {len(aligned_smiles_for_gnn_part)} SMILES that have corresponding numeric docking scores.")

            df_hybrid_predictions = run_hybrid_gnn_prediction(aligned_smiles_for_gnn_part, aligned_docking_scores_df_for_mlp)
            if not df_hybrid_predictions.empty:
                st.markdown("#### Hybrid GNN Predicted pIC50 Values:")
                if 'SMILES' not in df_hybrid_predictions.columns and len(df_hybrid_predictions) == len(aligned_smiles_for_gnn_part):
                    df_hybrid_predictions.insert(0, 'SMILES', aligned_smiles_for_gnn_part)
                st.dataframe(df_hybrid_predictions)
            else:
                st.info("Hybrid GNN pIC50 prediction yielded no data or an error occurred.")

def display_about_page():
    st.header("About This Application")
    st.markdown(f"**Ensemble AutoDock Vina App - v{APP_VERSION}**")
    st.markdown("""
    This application facilitates molecular docking simulations using AutoDock Vina and provides tools for chemical feature analysis and pIC50 prediction.
    
    **Features:**
    - **Ensemble Docking:**
        - Preparation of ligands from SMILES strings (with RDKit standardization) or various file formats.
        - Docking against one or multiple receptor structures.
        - Utilization of specific or multiple Vina configuration files.
        - Options for using a Perl-based screening script or direct Vina calls.
        - Summarization of best docking scores per ligand. These scores can be used by the Hybrid GNN model.
    - **Prediction Model Insights:**
        - Choice of prediction model: Original GNN or Hybrid GNN.
        - Standardization of input SMILES.
        - Calculation of 2D Mordred descriptors and ECFP4 fingerprints.
        - PCA visualization of input SMILES' Mordred descriptors against a reference BACE dataset.
        - pIC50 prediction using selected GNN model. The Hybrid GNN model utilizes graph features from SMILES and 15 pre-calculated docking scores.

    **File Structure Expectation (Example):**
    - `your_project_root/`
        - `streamlit_app.py` (this file)
        - `utils/`
            - `__init__.py`
            - `paths.py` (stores path constants)
            - `app_utils.py` (utility functions for docking)
            - `prediction_utils.py` (utility functions for prediction/feature analysis)
            - `gnn/`
                - `__init__.py`
                - `gnn_architecture.py` (GIN, GIN_hybrid, MLP1, CombinedMLP classes)
                - `gnn_train.py` (load_model, predict_pic50_gnn, predict_pic50_hybrid functions)
        - `ensemble_docking/`
            - `ligand_preprocessing/scrub.py`
            - `ligand_preprocessing/mk_prepare_ligand.py`
            - `Vina_screening.pl`
        - `vina/vina_1.2.5_linux_x86_64` (Vina executable, `chmod +x`)
        - `autodock_workspace/` (created for temporary files, fetched assets, downloaded models)
        - `autodock_outputs/` (created for PDBQT outputs from direct Vina calls)
        - `requirements.txt`
        - `packages.txt` (for Streamlit Cloud system dependencies)
    """)
    st.markdown(f"**Key Local Paths Used (resolved from `APP_ROOT` = `{APP_ROOT.resolve()}`):**\n"
                  f"- Workspace Parent: `{WORKSPACE_PARENT_DIR.resolve()}`\n"
                  f"- Vina Executable: `{VINA_PATH_LOCAL.resolve()}`\n"
                  f"- Direct Vina Output PDBQTs: `{DOCKING_OUTPUT_DIR_LOCAL.resolve()}`")

def main():
    st.set_page_config(layout="wide", page_title=f"Molecular Modeling Suite v{APP_VERSION}")
    
    initialize_directories()

    st.sidebar.image("https://raw.githubusercontent.com/HenryChritopher02/bace1/main/logo.png", width=300)
    st.sidebar.title("Categories")

    app_mode_options = ("Ensemble Docking", "Prediction Model", "About")
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
    elif app_mode == "Prediction Model":
        display_prediction_model_procedure()
    elif app_mode == "About":
        display_about_page()

if __name__ == "__main__":
    main()
