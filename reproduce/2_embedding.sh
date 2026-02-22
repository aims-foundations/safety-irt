#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv_safety_irt/bin/activate

pip install "unbabel-comet==2.2.7" "setuptools<81" "numpy>=1.26,<2" --quiet

python huggingface_login.py
python model/embedding_analysis_translation_v_CSG.py
python model/embedding_analysis_translation_v_safety.py