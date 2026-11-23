#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

# Build from a fresh activation context; the caller's shell is unchanged.
unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV

usage() {
  cat <<'EOF'
Usage: ./install.sh [--converter-only | --skip-models | --download-mamma-assets]

Install the converter and default GPU stage runtimes and prepare model assets.
--converter-only          Install just the converter environment and package.
--skip-models             Install runtimes without preparing gated models.
--download-mamma-assets   Use upstream downloads for MAMMA and SMPL-X assets.
                          The default reuses verified assets or downloads missing ones.
Set MULTIVIEW_DATA_CHECKOUT to use an editable dependency checkout and
VCI_MAMMA_ASSET_SOURCE to reuse an authorized local asset directory.
EOF
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "${1:-}" in
  ''|--converter-only|--skip-models|--download-mamma-assets) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ -n "${VCI_CONDA_EXE:-}" ]]; then
  conda_base="$("$VCI_CONDA_EXE" info --base)"
  export PATH="$conda_base/bin:$PATH"
elif [[ -x "$root/../.miniforge/bin/conda" ]]; then
  export PATH="$root/../.miniforge/bin:$PATH"
fi
command -v conda >/dev/null 2>&1 || {
  echo 'Conda is required. Install Miniforge or Miniconda first.' >&2
  exit 1
}

# Also works when distributed as a source archive with no declared submodules.
if [[ -e .git || -f .gitmodules ]]; then
  git submodule update --init --recursive
fi

env_name="${ENV_NAME:-vci-dataset}"
export MAMMA_REPO_PATH="${MAMMA_REPO_PATH:-${MAMMA_SUBMODULE_PATH:-$root/external/mamma}}"
export SAM3D_BODY_RERUN_REPO="${SAM3D_BODY_RERUN_REPO:-$root/external/sam3d-body-rerun}"
bash scripts/setup/install_env.sh --no-mamma-checkout
if ! conda run -n "$env_name" bash -c 'command -v ffmpeg && command -v ffprobe' >/dev/null 2>&1; then
  conda install -y -n "$env_name" -c conda-forge ffmpeg
fi
conda_bin="$(conda info --base)/bin/conda"
prefix="$(conda run -n "$env_name" python -c 'import sys; print(sys.prefix)')"
conda run --no-capture-output -p "$prefix" capture-process --help >/dev/null
printf '%s\n%s\n' "$conda_bin" "$prefix" > .cli-conda
printf '\nInstalled converter environment: %s\n' "$prefix"
if [[ "${1:-}" != --converter-only ]]; then
  if ! command -v pixi >/dev/null 2>&1 && ! conda run -n vci-pixi pixi --version >/dev/null 2>&1; then
    conda create -y -n vci-pixi -c conda-forge pixi
  fi
  bash scripts/setup/setup_mamma_env.sh
  bash scripts/setup/setup_sam3d_body_rerun_env.sh
  bash scripts/setup/setup_smplx_to_mhr_env.sh
  if [[ "${1:-}" == --skip-models ]]; then
    bash scripts/setup/setup_semantic_parsing_env.sh --skip-download
  else
    asset_setup=(conda run --no-capture-output -n "$env_name" python scripts/setup/setup_mamma_assets.py --repo "$MAMMA_REPO_PATH")
    if [[ "${1:-}" == --download-mamma-assets ]]; then
      "${asset_setup[@]}" --download
    elif [[ -n "${VCI_MAMMA_ASSET_SOURCE:-}" || -d models/authorized-mamma-assets ]]; then
      "${asset_setup[@]}"
    else
      "${asset_setup[@]}" --download
    fi
    "$SAM3D_BODY_RERUN_REPO/.pixi/envs/dev/bin/python" -m capture_processing.tools.prepare_mhr_runtime \
      --repo-path "$SAM3D_BODY_RERUN_REPO"
    bash scripts/setup/setup_semantic_parsing_env.sh
  fi
fi
printf '\nTry: ./run.sh capture-process --help\n'
