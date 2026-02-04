#!/bin/bash

# ==============================================================================
# REPRODUCE.SH - Automate Project Reproduction
# ==============================================================================
# Usage: ./reproduce.sh
#
# This script will:
# 1. Create a Python virtual environment
# 2. Install dependencies
# 3. Run the main experiment/analysis
# 4. Generate the final results
# ==============================================================================

# 1. PRE-FLIGHT CHECKS
# ------------------------------------------------------------------------------
# 'set -e' stops the execution immediately if any command fails (returns non-zero).
# 'set -u' stops execution if an unset variable is used.
set -eu

echo "🚀 Starting reproduction script..."

# 2. ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
VENV_NAME="venv_reproduce"

if [ -d "$VENV_NAME" ]; then
    echo "✅ Virtual environment '$VENV_NAME' already exists. Activating..."
else
    echo "📦 Creating virtual environment '$VENV_NAME'..."
    python3 -m venv $VENV_NAME
fi

# Activate the environment
source $VENV_NAME/bin/activate

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "⬇️ Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "⚠️ Warning: requirements.txt not found. Skipping dependency install."
fi

# 3. DATA PREPARATION
# ------------------------------------------------------------------------------
# Create a data directory if it doesn't exist
mkdir -p data

# Example: Download a file if it's missing (Uncomment to use)
# if [ ! -f "data/dataset.csv" ]; then
#     echo "🌍 Downloading dataset..."
#     curl -o data/dataset.csv https://example.com/data/dataset.csv
# fi

# 4. RUN MAIN EXPERIMENT
# ------------------------------------------------------------------------------
echo "🧠 Running main analysis/experiment..."

# Replace 'main.py' with your actual script name
# We assume the script outputs something to a 'results' folder
if [ -f "main.py" ]; then
    python main.py
else
    echo "❌ Error: main.py not found!"
    exit 1
fi

# 5. CLEANUP / FINISH
# ------------------------------------------------------------------------------
echo "🎉 Reproduction complete!"
echo "   Results can be found in the /results directory."
echo "   To exit the virtual environment, run: deactivate"