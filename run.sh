#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# == 0 )) || [[ "$1" == --help || "$1" == -h ]]; then
  printf 'Usage: ./run.sh COMMAND [ARGS...]\nRuns capture-process, capture-reprocess, capture-validate or Python in the installed converter.\nSet ENV_NAME to select a different environment or VCI_CONDA_EXE to select Conda.\n'
  exit 0
fi
if [[ -f "$root/.cli-conda" && -z "${VCI_CONDA_EXE:-}${ENV_NAME:-}" ]]; then
  mapfile -t cli < "$root/.cli-conda"
  if [[ ! -x "${cli[0]:-}" || ! -d "${cli[1]:-}/conda-meta" ]]; then
    printf 'The Conda env recorded in %s is unavailable. Use the installed checkout or rerun its installer.\n' "$root/.cli-conda" >&2
    exit 1
  fi
  export PATH="$(dirname -- "${cli[0]}"):$PATH"
  # Start Conda without the caller's activation stack or its deactivation hooks.
  unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV
  exec "${cli[0]}" run --no-capture-output -p "${cli[1]}" "$@"
fi
if [[ -n "${VCI_CONDA_EXE:-}" ]]; then
  conda="$VCI_CONDA_EXE"
elif [[ -x "$root/../.miniforge/bin/conda" ]]; then
  conda="$root/../.miniforge/bin/conda"
else
  conda="$(command -v conda || true)"
fi
if [[ -z "$conda" ]] || ! command -v -- "$conda" >/dev/null 2>&1; then
  printf 'Conda not found. Put your installed Conda on PATH or set VCI_CONDA_EXE.\n' >&2
  exit 1
fi
unset CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV
exec "$conda" run --no-capture-output -n "${ENV_NAME:-vci-dataset}" "$@"
