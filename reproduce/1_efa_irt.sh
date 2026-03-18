#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv_safety_irt/bin/activate

python model/efa.py
python model/irt.py
python model/anchor_validations/anchor_identificability.py