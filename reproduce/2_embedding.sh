#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv_safety_irt/bin/activate

python model/embedding_analysis_translation_v_DIF.py
python model/embedding_analysis_translation_v_safety.py