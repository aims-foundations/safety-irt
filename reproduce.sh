#!/bin/bash

# ==============================================================================
# REPRODUCE.SH - Safety Tax Project
# ==============================================================================
# Usage: ./reproduce.sh
#
# This script sets up the environment and attempts to run the core analysis pipeline.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Starting reproduction pipeline..."

# 1. ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
VENV_NAME="venv_safety_tax"

if [ -d "$VENV_NAME" ]; then
    echo "✅ Virtual environment '$VENV_NAME' detected. Activating..."
else
    echo "📦 Creating virtual environment '$VENV_NAME'..."
    python3 -m venv $VENV_NAME
fi

# Activate venv
source $VENV_NAME/bin/activate

# Upgrade pip just in case
pip install --upgrade pip

# 2. INSTALL DEPENDENCIES
# ------------------------------------------------------------------------------
if [ -f "requirements.txt" ]; then
    echo "⬇️  Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "❌ Error: requirements.txt not found!"
    exit 1
fi


# 3. RUN MAIN IRT ANALYSIS
# ------------------------------------------------------------------------------
# Checks for the file we created in previous steps
if [ -f "model/irt_with_new_term.py" ]; then
    echo "🧠 Running Anchored IRT Model..."
    python run_anchored_irt.py
else
    echo "⚠️  No main IRT script found (e.g., run_anchored_irt.py). Skipping model training."
fi

# 4. JUPYTER KERNEL SETUP (Optional)
# ------------------------------------------------------------------------------
# This registers the venv so you can use it inside Jupyter Notebooks
echo "🔗 Registering Jupyter kernel..."
python -m ipykernel install --user --name=$VENV_NAME --display-name "Python ($VENV_NAME)"

# 6. FINISH
# ------------------------------------------------------------------------------
echo "🎉 Setup complete!"
echo "   - To use the environment: source $VENV_NAME/bin/activate"
echo "   - To run the BatchGrading notebook, launch: jupyter notebook"
