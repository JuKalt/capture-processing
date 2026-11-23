#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAMMA_REPO="${MAMMA_REPO_PATH:-$REPO_ROOT/external/mamma}"
MAMMA_REPO_URL="${MAMMA_REPO_URL:-https://github.com/JanSpindler/mamma.git}"
MAMMA_REPO_COMMIT="${MAMMA_REPO_COMMIT:-ed75f08ff4a9ad5991f1767e744c1c853303fe81}"
MAMMA_CUDA_PROFILE="${MAMMA_CUDA_PROFILE:-legacy}"
case "$MAMMA_CUDA_PROFILE" in
  legacy) MAMMA_CUDA_VERSION=12.4; MAMMA_TORCH_VERSION=2.5.1; MAMMA_VISION_VERSION=0.20.1 ;;
  blackwell) MAMMA_CUDA_VERSION=12.8; MAMMA_TORCH_VERSION=2.7.1; MAMMA_VISION_VERSION=0.22.1 ;;
  *) echo 'MAMMA_CUDA_PROFILE must be legacy or blackwell.' >&2; exit 2 ;;
esac
export MAMMA_CUDA_VERSION

usage() {
  cat <<'HELP'
Usage: scripts/setup/setup_mamma_env.sh [--mamma-repo DIR]

Create or update the mamma Conda environment for the converter's default
SAM 2 inference pipeline: Python 3.11, PyTorch 2.5.1, and CUDA 12.4.
Installs the CUDA toolkit and build tools inside the environment.

Options:
  --mamma-repo DIR  Existing or new checkout (default: external/mamma)
  -h, --help        Show this help

Environment variables:
  MAMMA_REPO_PATH    Override the checkout directory
  MAMMA_REPO_URL     Override the clone URL
  MAMMA_REPO_COMMIT  Override the revision for a new checkout
  MAX_JOBS          CUDA extension build parallelism (default: 4)
  MAMMA_CUDA_PROFILE legacy (default: Torch 2.5.1/CUDA 12.4) or blackwell
                    (Torch 2.7.1/CUDA 12.8; target-machine validation required)
  TORCH_CUDA_ARCH_LIST  Optional build architectures for a headless build host

Existing checkouts are preserved. The environment is named mamma because
MAMMA's shipped presets launch their subprocesses in that environment.
Model downloads are separate: scripts/setup/download_mamma_assets.sh.
Optional SAM 3 is not installed; the converter's preset uses SAM 2.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mamma-repo)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "--mamma-repo requires a directory" >&2
        exit 2
      fi
      MAMMA_REPO="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Resolve relative paths from the caller, before changing to the checkout.
if [[ "$MAMMA_REPO" != /* ]]; then
  MAMMA_REPO="$PWD/$MAMMA_REPO"
fi
command -v conda >/dev/null 2>&1 || {
  echo "conda was not found on PATH. Install Miniconda or Miniforge first." >&2
  exit 1
}
if [[ ! -e "$MAMMA_REPO" ]]; then
  command -v git >/dev/null 2>&1 || {
    echo "git was not found on PATH." >&2
    exit 1
  }
  mkdir -p "$(dirname "$MAMMA_REPO")"
  git clone --no-checkout "$MAMMA_REPO_URL" "$MAMMA_REPO"
  git -C "$MAMMA_REPO" checkout --detach "$MAMMA_REPO_COMMIT"
else
  echo "Using existing MAMMA checkout: $MAMMA_REPO"
fi
if [[ -e "$MAMMA_REPO/.git" ]]; then
  git -C "$MAMMA_REPO" submodule update --init --recursive
fi
for required in inference/cli/run.py requirements/mamma_conda.yaml \
    requirements/requirements.txt requirements/requirements_no_build_isolation.txt; do
  [[ -f "$MAMMA_REPO/$required" ]] || {
    echo "MAMMA checkout is invalid or incomplete: missing $MAMMA_REPO/$required" >&2
    exit 1
  }
done
cd "$MAMMA_REPO"

SETUP_TMP="$(mktemp -d)"
trap 'rm -rf "$SETUP_TMP"' EXIT
CONDA_RECIPE=requirements/mamma_conda.yaml
if [[ "$MAMMA_CUDA_PROFILE" == blackwell ]]; then
  sed -e 's|/whl/cu124|/whl/cu128|g' -e 's/torch==2.5.1/torch==2.7.1/g' \
    -e 's/torchvision==0.20.1/torchvision==0.22.1/g' \
    "$CONDA_RECIPE" > "$SETUP_TMP/mamma_conda.yaml"
  CONDA_RECIPE="$SETUP_TMP/mamma_conda.yaml"
fi
if conda env list | awk '{print $1}' | grep -qx mamma; then
  conda env update -n mamma -f "$CONDA_RECIPE"
else
  conda env create -n mamma -f "$CONDA_RECIPE"
fi
# Pin the toolkit channel: current unlabelled NVIDIA packages can mismatch Torch.
if [[ "$MAMMA_CUDA_PROFILE" == blackwell ]]; then
  conda install -y -n mamma -c nvidia/label/cuda-12.8.1 -c conda-forge \
    cuda-toolkit=12.8 gxx_linux-64=12 cmake ninja pkg-config ffmpeg
  conda run --no-capture-output -n mamma python -m pip install \
    torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
else
  conda install -y -n mamma -c nvidia/label/cuda-12.4.1 -c conda-forge \
    cuda-toolkit=12.4 gxx_linux-64=12 cmake ninja pkg-config ffmpeg
fi
MAMMA_ENV_PREFIX="$(conda run -n mamma python -c 'import sys; print(sys.prefix)')"
export CUDA_HOME="$MAMMA_ENV_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
# Conda CUDA headers/libraries can live in the target directory rather than root.
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
conda run --no-capture-output -n mamma nvcc --version

# Keep the upstream files intact. SAM 3 is optional and not used by full.yaml.
sed '\|^git+https://github.com/facebookresearch/sam3|d' \
  requirements/requirements.txt > "$SETUP_TMP/requirements.txt"
# Prevent transitive dependencies from replacing the matched Torch/CUDA stack.
cat > "$SETUP_TMP/constraints.txt" <<CONSTRAINTS
torch==$MAMMA_TORCH_VERSION
torchvision==$MAMMA_VISION_VERSION
numpy>=1.26,<2
# PyArrow 26 requires NumPy 2 at import time without declaring it to pip.
pyarrow==19.0.1
setuptools<81
iopath==0.1.10
CONSTRAINTS
conda run --no-capture-output -n mamma python -m pip install \
  -c "$SETUP_TMP/constraints.txt" -r "$SETUP_TMP/requirements.txt"
# Detectron2's old iopath upper bound conflicts with SAM 2 and MAMMA.
# Build a local wheel with a narrow iopath 0.1.10 requirement; preserve upstream
# checkouts and installed metadata rather than editing site-packages.
DETECTRON_SOURCE="$SETUP_TMP/detectron2"
git clone --no-checkout https://github.com/facebookresearch/detectron2 "$DETECTRON_SOURCE"
git -C "$DETECTRON_SOURCE" checkout --detach 1e3e13bbf607b54f62205c4c33922521822fb298
conda run --no-capture-output -n mamma python - "$DETECTRON_SOURCE" \
  requirements/requirements_no_build_isolation.txt "$SETUP_TMP/detectron_requirements.txt" <<'PYTHON'
from pathlib import Path
import sys

source, requirements, output = map(Path, sys.argv[1:])
setup = source / "setup.py"
text = setup.read_text()
old = '"iopath>=0.1.7,<0.1.10"'
if text.count(old) != 1:
    raise SystemExit("Unexpected Detectron2 iopath requirement; compatibility patch aborted.")
lines = requirements.read_text().splitlines()
expected = "git+https://github.com/facebookresearch/detectron2.git"
if lines.count(expected) != 1:
    raise SystemExit("Unexpected Detectron2 source requirement; compatibility patch aborted.")
setup.write_text(text.replace(old, '"iopath==0.1.10"'))
output.write_text("\n".join(str(source) if line == expected else line for line in lines) + "\n")
print("Prepared Detectron2 dependency compatibility patch for SAM 2.")
PYTHON
BUILD_REQUIREMENTS="$SETUP_TMP/detectron_requirements.txt"
# Reinstall patched wheels even when a previous build has the same version.
BUILD_OPTIONS=(--force-reinstall --no-deps)
if [[ "$MAMMA_CUDA_PROFILE" == blackwell ]]; then
  # The upstream torch2.5-cu124 branch still dispatches on Tensor.type(),
  # which Torch 2.7 no longer accepts. Patch a pinned temporary source tree;
  # never modify the existing MAMMA checkout or downgrade Blackwell's Torch.
  SDF_SOURCE="$SETUP_TMP/pytorch_sdf"
  git clone --no-checkout https://github.com/cuevhv/pytorch_sdf "$SDF_SOURCE"
  git -C "$SDF_SOURCE" checkout --detach 58ff57b542a998e406327b94881bf6f7ce13a390
  git -C "$SDF_SOURCE" submodule update --init --recursive
  conda run --no-capture-output -n mamma python - "$SDF_SOURCE" \
    "$BUILD_REQUIREMENTS" "$SETUP_TMP/build_requirements.txt" <<'PYTHON'
from pathlib import Path
import sys

source, requirements, output = map(Path, sys.argv[1:])
cuda_source = source / "src/bvh_cuda_op.cu"
text = cuda_source.read_text()
old = 'triangles.type(), "bvh_tree_building"'
new = 'triangles.scalar_type(), "bvh_tree_building"'
if text.count(old) != 1:
    raise SystemExit("Unexpected pytorch_sdf dispatch source; compatibility patch aborted.")
lines = requirements.read_text().splitlines()
expected = "git+https://github.com/cuevhv/pytorch_sdf@torch2.5-cu124"
if lines.count(expected) != 1:
    raise SystemExit("Unexpected pytorch_sdf requirement; compatibility patch aborted.")
cuda_source.write_text(text.replace(old, new))
output.write_text("\n".join(str(source) if line == expected else line for line in lines) + "\n")
print("Prepared pytorch_sdf dispatch compatibility patch for Torch 2.7.")
PYTHON
  BUILD_REQUIREMENTS="$SETUP_TMP/build_requirements.txt"
fi
conda run --no-capture-output -n mamma python -m pip install \
  --no-build-isolation "${BUILD_OPTIONS[@]}" -c "$SETUP_TMP/constraints.txt" \
  -r "$BUILD_REQUIREMENTS"
# Rebuild native wheels without reinstalling the entire CUDA stack, then let
# pip resolve their declared runtime dependencies from the installed metadata.
# No --no-deps here: missing fvcore/tensorboard/etc. must be installed.
conda run --no-capture-output -n mamma python -m pip install \
  -c "$SETUP_TMP/constraints.txt" detectron2 pytorch_sdf
# Use a known NumPy 1.x compatible wheel, including when repairing an existing
# environment. pip check alone cannot detect PyArrow's NumPy ABI requirement.
conda run --no-capture-output -n mamma python -m pip install \
  --force-reinstall --no-deps --only-binary=:all: -c "$SETUP_TMP/constraints.txt" \
  pyarrow==19.0.1
conda run --no-capture-output -n mamma python -c \
  'import numpy as np, pyarrow as pa, rerun as rr; rr.Points3D(np.zeros((1, 3), dtype=np.float32)); print("MAMMA visualization imports passed:", np.__version__, pa.__version__)'
conda run --no-capture-output -n mamma python -m pip check
conda run --no-capture-output -n mamma python -c \
  'import os, torch, torchvision, sam2, detectron2, pytorch_sdf, smplx; from detectron2.modeling import build_model; from detectron2.utils.file_io import PathManager; assert torch.version.cuda == os.environ["MAMMA_CUDA_VERSION"], torch.version.cuda; print("MAMMA dependency imports passed; CUDA available:", torch.cuda.is_available())'
conda run --no-capture-output -n mamma python -m inference --help
# --help does not import the lazy segmentation pipeline used by process_seq.
(
  cd "$MAMMA_REPO/segmentation"
  conda run --no-capture-output -n mamma python -c \
    'import sys; from core.pipeline import SegmentMultipleFrames; print("MAMMA segmentation imports passed:", sys.executable)'
)

printf '\nMAMMA environment installed. Download model assets next:\n'
printf '  %q --mamma-repo %q\n' "$REPO_ROOT/scripts/setup/download_mamma_assets.sh" "$MAMMA_REPO"
printf '\nThen check asset paths from the checkout:\n'
printf '  cd %q\n' "$MAMMA_REPO"
printf '  conda run -n mamma python -m inference doctor\n'
