#!/bin/bash
# Usage: ./reproduce.sh
# Reproduces the EFA and IRT analysis from the paper.

set -e
cd "$(dirname "$0")"

# 1. Setup virtual environment and install dependencies
python3 -m venv venv_safety_irt
source venv_safety_irt/bin/activate
pip install -q -r requirements.txt

# 2. Run analyses (data is downloaded from HuggingFace automatically)
# python model/efa.py
# python model/irt.py
python model/embedding_analysis_translation_v_DIF.py
python model/embedding_analysis_translation_v_safety.py

echo "Done. Results saved to model/results/"
