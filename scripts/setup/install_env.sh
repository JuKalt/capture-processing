#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-vci-dataset}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_MAMMA_CHECKOUT=1
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
MAMMA_REPO_PATH="${MAMMA_REPO_PATH:-${MAMMA_SUBMODULE_PATH:-external/mamma}}"
MAMMA_REPO_URL="${MAMMA_REPO_URL:-https://github.com/JanSpindler/mamma.git}"
MAMMA_REPO_COMMIT="${MAMMA_REPO_COMMIT:-ed75f08ff4a9ad5991f1767e744c1c853303fe81}"

usage() {
  cat <<'EOF'
Usage: scripts/setup/install_env.sh [--env-name NAME] [--python VERSION] [--no-torch] [--no-mamma-checkout]

Creates a conda environment for the VCI -> THUman4 dataset converter.

Environment variables:
  ENV_NAME          Conda env name. Default: vci-dataset
  PYTHON_VERSION    Python version. Default: 3.10
  INSTALL_TORCH     Install torch/smplx legacy extras. Default: 1
  TORCH_INDEX_URL   Optional pip index URL for torch wheels, e.g.
                    https://download.pytorch.org/whl/cu121
  MAMMA_REPO_PATH   MAMMA checkout path. Default: external/mamma
  MAMMA_REPO_URL    MAMMA Git URL. Default: upstream GitHub repository
  MAMMA_REPO_COMMIT Pinned MAMMA revision used for a new checkout

Notes:
  MAMMA is cloned at the pinned revision when its checkout is absent, unless
  --no-mamma-checkout is used.
  An existing checkout is preserved and used as-is.
  MAMMA inference still runs in the MAMMA environment/checkpoint setup.
  This env is for this converter and for the legacy --smplx-source existing path.
EOF
}

ensure_mamma_checkout() {
  if ! command -v git >/dev/null 2>&1; then
    echo "git was not found on PATH. It is required to clone MAMMA." >&2
    exit 1
  fi

  if [[ ! -e "$MAMMA_REPO_PATH" ]]; then
    echo "Cloning MAMMA into $MAMMA_REPO_PATH"
    mkdir -p "$(dirname "$MAMMA_REPO_PATH")"
    git clone --no-checkout "$MAMMA_REPO_URL" "$MAMMA_REPO_PATH"
    git -C "$MAMMA_REPO_PATH" checkout --detach "$MAMMA_REPO_COMMIT"
  else
    echo "Using existing MAMMA checkout: $MAMMA_REPO_PATH"
  fi

  if [[ -e "$MAMMA_REPO_PATH/.git" ]]; then
    git -C "$MAMMA_REPO_PATH" submodule update --init --recursive
  fi
  if [[ ! -f "$MAMMA_REPO_PATH/inference/cli/run.py" ]]; then
    echo "MAMMA checkout is missing expected inference CLI: $MAMMA_REPO_PATH" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      if [[ $# -lt 2 ]]; then
        echo "--env-name requires a value" >&2
        exit 2
      fi
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      if [[ $# -lt 2 ]]; then
        echo "--python requires a value" >&2
        exit 2
      fi
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --no-torch)
      INSTALL_TORCH=0
      shift
      ;;
    --no-mamma-checkout)
      INSTALL_MAMMA_CHECKOUT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ((INSTALL_MAMMA_CHECKOUT)); then
  ensure_mamma_checkout
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Install Miniconda/Mambaforge first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env already exists: $ENV_NAME"
else
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip

python -m pip install \
  numpy \
  Pillow \
  pillow-avif-plugin \
  python-dotenv \
  PyYAML \
  tqdm \
  textual

if [[ "$INSTALL_TORCH" == "1" ]]; then
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    python -m pip install --index-url "$TORCH_INDEX_URL" torch
  else
    python -m pip install torch
  fi
  python -m pip install scipy smplx
fi

# Compilation alone does not execute imports (many are lazy in the converter).
# Check through conda run, matching the interpreter used by documented runs.
conda run --no-capture-output -n "$ENV_NAME" python -c \
  'import sys, numpy, PIL, pillow_avif, dotenv, yaml, tqdm, textual; print("Converter dependency imports passed:", sys.executable)'
if [[ "$INSTALL_TORCH" == "1" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" python -c \
    'import torch, scipy, smplx; print("Body-parameter dependency imports passed")'
fi
DATA_REPO="${MULTIVIEW_DATA_CHECKOUT:-$REPO_ROOT/../multiview-data}"
packages=(-e "$REPO_ROOT[native]")
if [[ -f "$DATA_REPO/pyproject.toml" ]]; then
  packages+=(-e "$DATA_REPO[native]")
elif [[ -n "${MULTIVIEW_DATA_CHECKOUT:-}" ]]; then
  echo "Invalid MULTIVIEW_DATA_CHECKOUT: $DATA_REPO" >&2
  exit 1
fi
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "${packages[@]}"
conda run --no-capture-output -n "$ENV_NAME" python process_dataset.py --help >/dev/null

python -m py_compile \
  process_dataset.py \
  capture_processing/cli.py \
  capture_processing/core/vci_pipeline.py \
  capture_processing/core/thuman_pipeline.py \
  capture_processing/core/smplx_pipeline.py \
  capture_processing/core/processing_log.py \
  capture_processing/utils/camera.py \
  capture_processing/utils/frames.py \
  capture_processing/utils/mamma.py \
  capture_processing/utils/mhr.py \
  capture_processing/utils/smplx_utils.py \
  capture_processing/utils/paths.py \
  capture_processing/utils/transcode.py

cat <<EOF

Done.

Run converter:
  ./run.sh capture-process <input_parent> <output_parent>

EOF
