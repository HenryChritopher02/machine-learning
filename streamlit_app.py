import streamlit as st
import subprocess
import os
import stat # Module for permission constants

# Get absolute path to Vina binary
VINA_PATH = os.path.abspath("./vina/vina_1.2.5_linux_x86_64")

st.write("Vina binary path:", VINA_PATH)
st.write("File exists:", os.path.exists(VINA_PATH))

if os.path.exists(VINA_PATH):
    # Check if executable, if not, try to set it
    if not os.access(VINA_PATH, os.X_OK):
        st.write("File is not executable. Attempting to set execute permission...")
        try:
            # Get current permissions
            current_mode = os.stat(VINA_PATH).st_mode
            # Add execute permission for user, group, and others
            os.chmod(VINA_PATH, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            st.write("Execute permission set (or attempted).")
        except OSError as e:
            st.error(f"Error setting execute permission: {e}. This is likely a permissions issue in the environment.")
            st.warning("It's highly recommended to set execute permissions via 'git add --chmod=+x' instead.")
else:
    st.error("Vina binary not found at the specified path!")


st.write("Is executable (after attempt):", os.access(VINA_PATH, os.X_OK))

# ... rest of your code to run Vina
st.title("AutoDock Vina Docking App")

# Upload pre-prepared PDBQT files
receptor_pdbqt = st.file_uploader("Upload Receptor (PDBQT)", type="pdbqt")
ligand_pdbqt = st.file_uploader("Upload Ligand (PDBQT)", type="pdbqt")

if receptor_pdbqt and ligand_pdbqt:
    # Save uploaded PDBQT files directly
    with open("receptor.pdbqt", "wb") as f:
        f.write(receptor_pdbqt.getbuffer())
    with open("ligand.pdbqt", "wb") as f:
        f.write(ligand_pdbqt.getbuffer())

    if st.button("Run Docking"):
        try:
            cmd = [
                VINA_PATH,
                "--receptor", os.path.abspath("receptor.pdbqt"),
                "--ligand", os.path.abspath("ligand.pdbqt"),
                "--center_x", "0",  # Replace with your box coordinates
                "--center_y", "0",
                "--center_z", "0",
                "--size_x", "20",
                "--size_y", "20",
                "--size_z", "20",
                "--exhaustiveness", "8"  # Add other Vina parameters
            ]
            
            # Run Vina with error handling
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            st.success("Docking completed!")
            st.code(result.stdout)
            
        except subprocess.CalledProcessError as e:
            st.error(f"Docking failed: {e.stderr}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
