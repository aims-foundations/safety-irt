#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv_safety_irt/bin/activate

#python irt_validations/A_model-selection.py
#python irt_validations/B_variable-reliability_2PL.py
python irt_validations/D_predictive-validation_2PL.py