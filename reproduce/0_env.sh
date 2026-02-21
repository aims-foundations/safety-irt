#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv venv_safety_irt
source venv_safety_irt/bin/activate
pip install -q -r requirements.txt