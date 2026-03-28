#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv_safety_irt/bin/activate

python model/anchor_validations/anchor_identificability.py
python irt_validations/A_model-selection.py
python irt_validations/anchor_sensitivity_ablation.py
python irt_validations/B_variable-reliability_2PL.py
python irt_validations/D_predictive-validation_2PL.py
python irt_validations/h1_irt_analysis.py
python irt_validations/high_tau_top100-prompts.py
python irt_validations/high_tau_categories.py
python irt_validations/high_tau_prompt-response_inspection.py

python irt_validations/jsr_difficulty.py
python irt_validations/jsr_irt_analysis.py
python irt_validations/jsr_irt_ordering.py
python irt_validations/tau_judge_artifact.py
python irt_validations/temperature_jsr_by_language.py