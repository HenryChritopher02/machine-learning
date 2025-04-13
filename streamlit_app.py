import streamlit as st
import subprocess
import os

# Get absolute path to Vina binary
VINA_PATH = os.path.abspath("vina/vina_1.2.5_linux_x86_64")
# Debug: Print permissions and path
st.write("Vina binary path:", os.path.abspath("./vina/vina_1.2.5_linux_x86_64"))
st.write("File exists:", os.path.exists("./vina/vina_1.2.5_linux_x86_64"))
st.write("Is executable:", os.access("./vina/vina_1.2.5_linux_x86_64", os.X_OK))
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
