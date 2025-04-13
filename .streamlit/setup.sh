#!/bin/bash
# Locally, make setup.sh executable
chmod +x .streamlit/setup.sh

# Commit and push
git add .streamlit/setup.sh
git commit -m "Make setup.sh executable"
git push
ls -l ./vina/vina_1.2.5_linux_x86_64  # Debug: Check permissions
