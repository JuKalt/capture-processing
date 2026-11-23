#!/usr/bin/env bash
# Reuse an existing licensed model without copying it or replacing user assets.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DESTINATION="${1:?Usage: bash link_smplx_model.sh DESTINATION}"
MODEL_SOURCE="${VCI_SMPLX_MODEL:-$REPO_ROOT/external/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz}"

if [[ -e "$DESTINATION" || -L "$DESTINATION" ]]; then
  echo "Preserving existing SMPL-X model path: $DESTINATION"
  if [[ ! -f "$DESTINATION" ]]; then
    echo "Warning: this path is not a usable model file; repair it before training." >&2
  fi
elif [[ -f "$MODEL_SOURCE" ]]; then
  MODEL_SOURCE="$(realpath "$MODEL_SOURCE")"
  mkdir -p "$(dirname "$DESTINATION")"
  ln -s "$MODEL_SOURCE" "$DESTINATION"
  echo "Linked SMPL-X model: $DESTINATION -> $MODEL_SOURCE"
else
  echo "SMPL-X model was not found at: $MODEL_SOURCE" >&2
  echo "Before training, install MAMMA's model assets and rerun setup, set VCI_SMPLX_MODEL to an existing model, or place SMPLX_NEUTRAL.npz at: $DESTINATION" >&2
fi
