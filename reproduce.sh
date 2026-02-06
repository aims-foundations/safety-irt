#!/bin/bash

# ==============================================================================
# REPRODUCE.SH - Safety Tax Project
# ==============================================================================
# Usage: ./reproduce.sh
#
# This script sets up the environment and attempts to run the core analysis pipeline.
# ==============================================================================

set -e

# Always run from the directory where this script lives
cd "$(dirname "$0")"

echo "🚀 Starting reproduction pipeline..."

# 1. ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
VENV_NAME="venv_safety_irt"

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment '$VENV_NAME' detected. Activating..."
else
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv $VENV_NAME
fi

source $VENV_NAME/bin/activate

# Set HF cache to a local writable directory (avoids AFS/NFS permission issues)
export HF_HOME="$(pwd)/.hf_cache"
mkdir -p "$HF_HOME"

# Upgrade pip just in case
pip install --upgrade pip

# 2. INSTALL DEPENDENCIES
# ------------------------------------------------------------------------------
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

cd model

# 3. RUN EFA ANALYSIS
# ------------------------------------------------------------------------------
if [ -f "efa.py" ]; then
    echo "Running EFA Analysis..."
    python efa.py
else
    echo "No EFA script found."
fi

# 4. RUN IRT ANALYSIS
# ------------------------------------------------------------------------------
if [ -f "irt.py" ]; then
    echo "Running Anchored IRT Model..."
    python irt.py
else
    echo "No IRT script found."
fi

# 5. JUPYTER KERNEL SETUP (Optional)
# ------------------------------------------------------------------------------
echo "🔗 Registering Jupyter kernel..."
python -m ipykernel install --user --name=$VENV_NAME --display-name "Python ($VENV_NAME)"

# 6. FINISH
# ------------------------------------------------------------------------------
echo "🎉 Setup complete!"
echo "   - To use the environment: source $VENV_NAME/bin/activate"
echo "   - To run the BatchGrading notebook, launch: jupyter notebook"
