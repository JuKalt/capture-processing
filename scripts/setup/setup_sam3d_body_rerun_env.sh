#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD_REPO="${SAM3D_BODY_RERUN_REPO:-$REPO_ROOT/external/sam3d-body-rerun}"
METHOD_REPO_URL="${SAM3D_BODY_RERUN_REPO_URL:-https://github.com/rerun-io/sam3d-body-rerun.git}"
METHOD_REPO_COMMIT="${SAM3D_BODY_RERUN_REPO_COMMIT:-91f3f725bae353b855c36db894cd07b06baba8ed}"

command -v git >/dev/null 2>&1 || {
  echo "git was not found on PATH. It is required to clone sam3d-body-rerun." >&2
  exit 1
}
# Invoke only Pixi from its optional tool environment. Adding that environment's
# bin directory to PATH can make nested conda runs select its Python instead.
if command -v pixi >/dev/null 2>&1; then
  PIXI_COMMAND=(pixi)
elif command -v conda >/dev/null 2>&1 && conda run -n vci-pixi pixi --version >/dev/null 2>&1; then
  PIXI_COMMAND=(conda run --no-capture-output -n vci-pixi pixi)
else
  echo "Pixi is missing. Install it with: conda create -y -n vci-pixi -c conda-forge pixi" >&2
  exit 1
fi

if [[ ! -e "$METHOD_REPO" ]]; then
  mkdir -p "$(dirname "$METHOD_REPO")"
  git clone --no-checkout "$METHOD_REPO_URL" "$METHOD_REPO"
  git -C "$METHOD_REPO" checkout --detach "$METHOD_REPO_COMMIT"
fi

if [[ ! -f "$METHOD_REPO/pyproject.toml" || ! -d "$METHOD_REPO/src/sam3d_body" ]]; then
  echo "sam3d-body-rerun checkout is invalid or incomplete: $METHOD_REPO" >&2
  echo "Remove or repair that directory, then rerun this script." >&2
  exit 1
fi
if [[ -e "$METHOD_REPO/.git" ]]; then
  git -C "$METHOD_REPO" submodule update --init --recursive
fi

echo "Installing the sam3d-body-rerun Pixi dev environment"
(
  cd "$METHOD_REPO"
  "${PIXI_COMMAND[@]}" install -e dev
)

cat <<EOF

sam3d-body-rerun is ready.
Use its Python environment with:
  --mhr-python "$METHOD_REPO/.pixi/envs/dev/bin/python"
EOF
