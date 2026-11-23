"""Launch the official CPU SMPL-X → MHR fitting stage independently of SAM3D."""
from __future__ import annotations

from capture_processing.utils.subject_lock import subject_locked

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAME = "mhr_from_smplx_params.npz"


@dataclass(frozen=True)
class ConversionOptions:
    enabled: bool = True
    repo_path: Path | None = None
    model_path: Path | None = None
    python: str = "python"
    force: bool = False


def add_conversion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--skip-smplx-to-mhr', action='store_true',
                        help='Skip the additional official SMPL-X to MHR conversion (enabled by default).')
    parser.add_argument('--smplx-to-mhr-repo-path', type=Path,
                        help='Official MHR checkout; defaults to MHR_REPO, then external/MHR.')
    parser.add_argument('--smplx-to-mhr-model-path', type=Path,
                        help='SMPL-X neutral NPZ; defaults to VCI_SMPLX_MODEL, then the MAMMA model.')
    parser.add_argument('--smplx-to-mhr-python', default=None,
                        help='Conversion interpreter; defaults to the official MHR Pixi environment if installed, then python.')
    parser.add_argument('--smplx-to-mhr-force', action='store_true', help='Recompute the official conversion.')


def options_from_args(args) -> ConversionOptions | None:
    # Older library callers do not implicitly launch a new external runtime.
    if not hasattr(args, 'skip_smplx_to_mhr'):
        return None
    repo = Path(args.smplx_to_mhr_repo_path or os.environ.get('MHR_REPO', PROJECT_ROOT / 'external/MHR'))
    interpreter = repo / '.pixi/envs/default/bin/python'
    return ConversionOptions(
        enabled=not args.skip_smplx_to_mhr, repo_path=repo,
        model_path=args.smplx_to_mhr_model_path,
        python=args.smplx_to_mhr_python or (str(interpreter) if interpreter.is_file() else 'python'),
        force=args.smplx_to_mhr_force,
    )


def conversion_signature(source: Path, model: Path, repo: Path) -> str:
    from capture_processing.utils.stage_cache import content_digest
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    # Worker-independent: the model is identified by content, not its local path.
    metadata = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'model': model.name, 'model_sha256': content_digest(model), 'revision': revision,
                'backend': 'pymomentum', 'lod': 1, 'single_identity': True,
                'tracking': True, 'seed': 0, 'schema': 2}
    return json.dumps(metadata, sort_keys=True)


def convert_subject_smplx_to_mhr(subject_root: Path, options: ConversionOptions | None) -> Path | None:
    if options is None or not options.enabled:
        return None
    return _convert_subject_smplx_to_mhr(subject_root, options)


@subject_locked
def _convert_subject_smplx_to_mhr(subject_root: Path, options: ConversionOptions) -> Path:
    if options is None or not options.enabled:
        return None
    from capture_processing.utils.mhr import run_mhr_visualization
    source = subject_root / 'smpl_params.npz'
    if not source.is_file():
        raise SystemExit(f'Official SMPL-X → MHR conversion needs {source}')
    repo = Path(options.repo_path or os.environ.get('MHR_REPO', PROJECT_ROOT / 'external/MHR')).expanduser().resolve()
    if not (repo / 'tools/mhr_smpl_conversion/conversion.py').is_file():
        raise SystemExit(f'Official MHR checkout missing at {repo}. Run scripts/setup/setup_smplx_to_mhr_env.sh.')
    model = Path(options.model_path or os.environ.get('VCI_SMPLX_MODEL',
                 PROJECT_ROOT / 'external/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz')).expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f'SMPL-X model missing at {model}; set --smplx-to-mhr-model-path.')
    signature = conversion_signature(source, model, repo)
    output = subject_root / OUTPUT_NAME
    from capture_processing.utils.subject_validation import converted_mhr_complete
    if not options.force and converted_mhr_complete(output, signature):
        print(f'Using existing complete SMPL-X → MHR conversion: {output}', flush=True)
        run_mhr_visualization(subject_root, repo, options.python, 'cpu', kind='mhrfromsmplx')
        return output
    python_cmd = shlex.split(options.python)
    if not python_cmd:
        raise SystemExit('--smplx-to-mhr-python must not be empty')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [*python_cmd, '-m', 'capture_processing.tools.convert_smplx_to_mhr', '--subject-root', str(subject_root.resolve()),
           '--repo-path', str(repo), '--model-path', str(model)]
    print(f'Running official deterministic SMPL-X → MHR conversion: {subject_root.name}', flush=True)
    log_path = subject_root / 'smplx_to_mhr.log'
    try:
        with log_path.open('w') as log:
            subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'Official SMPL-X → MHR conversion failed; see {log_path}. {exc}') from exc
    if not converted_mhr_complete(output, signature):
        raise SystemExit(f"Official conversion did not write a complete matching output: {output}")
    run_mhr_visualization(subject_root, repo, options.python, 'cpu', kind='mhrfromsmplx', force=options.force)
    return output
