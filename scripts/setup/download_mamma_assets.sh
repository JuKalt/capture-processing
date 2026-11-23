#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Keep the existing entry point and flags for standalone use.
args=()
mode=--download
while (($#)); do
  case "$1" in
    --mamma-repo)
      [[ $# -ge 2 ]] || { echo 'Missing --mamma-repo path' >&2; exit 2; }
      args+=(--repo "$2"); shift ;;
    --public-only|--gated-only)
      [[ "$mode" == --download ]] || { echo 'Choose one download mode.' >&2; exit 2; }
      mode="$1" ;;
    -h|--help) exec "${PYTHON:-python3}" "$root/scripts/setup/setup_mamma_assets.py" --help ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
if (( ! ${#args[@]} )) && [[ -n "${MAMMA_REPO:-}" ]]; then
  args+=(--repo "$MAMMA_REPO")
fi
exec "${PYTHON:-python3}" "$root/scripts/setup/setup_mamma_assets.py" "$mode" "${args[@]}"
