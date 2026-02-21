#!/usr/bin/env bash
chmod +x reproduce/*.sh 2>/dev/null || true
set -euo pipefail
cd "$(dirname "$0")"

task="${1:-all}"

case "$task" in
  all)
    ./reproduce/0_env.sh
    ./reproduce/1_efa_irt.sh
    ./reproduce/2_embedding.sh
    ./reproduce/3_validations.sh
    ;;
  env)          ./reproduce/0_env.sh ;;
  core)         ./reproduce/0_env.sh && ./reproduce/1_efa_irt.sh ;;
  embedding)    ./reproduce/0_env.sh && ./reproduce/2_embedding.sh ;;
  validations)  ./reproduce/0_env.sh && ./reproduce/3_validations.sh ;;
  *)
    echo "Usage: $0 {all|env|core|embedding|validations}"
    exit 1
    ;;
esac

echo "Done. Results saved to model/results/"