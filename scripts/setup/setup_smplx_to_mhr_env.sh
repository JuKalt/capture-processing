#!/usr/bin/env bash
# Install the official conversion runtime, preserving existing checkouts.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MHR_REPO="${MHR_REPO:-$REPO_ROOT/external/MHR}"
MHR_COMMIT="${MHR_COMMIT:-d96fafa33bbf018647c70c3525e91f53e79d2a14}"
if [[ ! -e "$MHR_REPO" ]]; then
  git clone --no-checkout https://github.com/facebookresearch/MHR.git "$MHR_REPO"
  git -C "$MHR_REPO" checkout --detach "$MHR_COMMIT"
fi
if [[ -e "$MHR_REPO/.git" ]]; then
  git -C "$MHR_REPO" submodule update --init --recursive
fi
if command -v pixi >/dev/null 2>&1; then
  PIXI_COMMAND=(pixi)
else
  PIXI_COMMAND=(conda run --no-capture-output -n vci-pixi pixi)
fi
cd "$MHR_REPO"
"${PIXI_COMMAND[@]}" add --pypi smplx scikit-learn tqdm pillow opencv-python-headless rerun-sdk
"${PIXI_COMMAND[@]}" install
if [[ ! -f assets/lod1.fbx ]]; then
  "${PIXI_COMMAND[@]}" run python -m mhr.download_assets --release v1.0.1 --dest .
fi
printf '\nOfficial SMPL-X → MHR runtime ready: %s\n' "$MHR_REPO/.pixi/envs/default/bin/python"
