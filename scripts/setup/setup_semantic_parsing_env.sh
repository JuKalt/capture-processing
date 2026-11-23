#!/usr/bin/env bash
# Isolated dependencies: never update the converter or fitting environments.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
semantic_env="${VCI_SEMANTIC_ENV:-sapiens2}"
model_dir="${VCI_SAPIENS2_MODEL:-$repo_root/models/sapiens2-seg-1b}"
check_only=0
skip_download=0
while (($#)); do
  case "$1" in
    --env-name|--model-dir)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in --env-name) semantic_env="$2" ;; --model-dir) model_dir="$2" ;; esac
      shift 2 ;;
    --check) check_only=1; shift ;;
    --skip-download) skip_download=1; shift ;;
    -h|--help)
      echo 'Usage: setup_semantic_parsing_env.sh [--env-name sapiens2] [--model-dir DIR] [--skip-download] [--check]'
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
command -v conda >/dev/null || { echo 'Conda is required for the isolated sapiens2 environment.' >&2; exit 1; }
if ((!check_only)); then
  if ! conda run -n "$semantic_env" python -c 'import sys; assert sys.version_info >= (3, 12)' >/dev/null 2>&1; then
    conda create -y -n "$semantic_env" python=3.12 pip
  fi
  if ! conda run -n "$semantic_env" python -c 'import torch, transformers, cv2, safetensors; from transformers import Sapiens2ForSemanticSegmentation; assert transformers.__version__ == "5.20.2"; assert torch.__version__.split("+")[0] == "2.7.1"' >/dev/null 2>&1; then
    conda run --no-capture-output -n "$semantic_env" python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    conda run --no-capture-output -n "$semantic_env" python -m pip install transformers==5.20.2 opencv-python-headless==4.11.0.86 pillow safetensors huggingface_hub
  fi
fi
conda run --no-capture-output -n "$semantic_env" python -c 'import torch, transformers, cv2, PIL, numpy, safetensors; assert torch.__version__.split("+")[0] == "2.7.1"; assert transformers.__version__ == "5.20.2"; from transformers import AutoImageProcessor, Sapiens2ForSemanticSegmentation; print("Sapiens2 runtime imports passed")'
if ((!skip_download)); then
  model_flags=()
  if ((check_only)); then
    model_flags+=(--check)
  elif conda run -n "$semantic_env" python scripts/setup/setup_semantic_model.py --model-dir "$model_dir" --check >/dev/null 2>&1; then
    printf 'Verified existing Sapiens2 assets: %s\n' "$model_dir"
    exit 0
  else
    conda run --no-capture-output -n "$semantic_env" python -m capture_processing.tools.ensure_hf_login
  fi
  conda run --no-capture-output -n "$semantic_env" python scripts/setup/setup_semantic_model.py --model-dir "$model_dir" "${model_flags[@]}"
fi
