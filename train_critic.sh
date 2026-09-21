#!/usr/bin/env bash
# Launch from any directory; relative argument paths use the caller's directory.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: bash train_critic.sh --stage {sft,dpo} --data-dir DIR --output-dir DIR [options]

Train the critic using prepared, registered datasets and LLaMA-Factory.
Activate your training environment before running this script.

Options:
  --stage {sft,dpo}   Training stage (required).
  --model PATH       Base model path or ID (default: Qwen/Qwen3-8B).
  --data-dir DIR     Prepared dataset directory with dataset_info.json (required).
  --output-dir DIR   New or empty checkpoint directory (required).
  --sft-adapter DIR  SFT adapter directory (required for DPO).
  --gpus IDS         Comma-separated GPU indices (default: 0).
  --dry-run          Validate inputs and print configuration without training.
  -h, --help         Show this help.

Set PYTHON_BIN to select a Python executable (default: python).
Relative paths are resolved from your current working directory.
This launcher does not prepare datasets or choose task names.
Keep training, development, and final evaluation tasks disjoint;
exclude development and final evaluation tasks from DPO training.

Examples (run from the repository root; replace paths as needed):
  bash train_critic.sh --stage sft --model /models/Qwen3-8B \
    --data-dir critic_training/data/sft --output-dir critic_training/outputs/sft \
    --gpus 0 --dry-run

  bash train_critic.sh --stage dpo --model /models/Qwen3-8B \
    --data-dir critic_training/data/dpo --output-dir critic_training/outputs/dpo \
    --sft-adapter critic_training/outputs/sft --gpus 0,1,2,3 --dry-run

Remove --dry-run to start training.
HELP
    exit 0
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
    exit 127
fi

exec "$PYTHON_BIN" "$ROOT/critic_training/train.py" "$@"
