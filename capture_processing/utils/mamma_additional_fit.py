"""Independent lifecycle for the additional confidence-filtered MAMMA fit."""
from __future__ import annotations

from capture_processing.utils.subject_lock import subject_locked
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import zipfile

from capture_processing.core.vci_resume import valid_smplx
from capture_processing.utils.stage_cache import content_stamp, file_stamp

KIND = 'smplx_confidence'
PARAMS = 'smpl_params_confidence.npz'
MANIFEST = 'mamma_confidence_fit.json'
PROJECT = Path(__file__).resolve().parents[2]


def requested(args):
    return (getattr(args, 'mamma_confidence_fit', True)
            and getattr(args, 'smplx_source', 'mamma') == 'mamma')


def _assets(repo, original):
    from capture_processing.utils.mamma import _resolve_mamma_asset_path
    def resolve(key, saved, default):
        if os.environ.get(key):
            return _resolve_mamma_asset_path(repo, os.environ, key, default)
        path = Path(original.get(saved) or default)
        if not path.is_absolute():
            path = repo / 'optimization' / path
        return path.resolve() if path.exists() else repo / default
    return (resolve('MAMMA_SMPLX_LOCKHEAD_MODELS', 'smplx_models', 'data/body_models/smplx_locked_head'),
            resolve('MAMMA_DOWNSAMPLED_VERTS_PKL', 'downsampled_verts',
                    'data/body_models/downsampled_verts/verts_512.pkl'))


def inputs(subject, repo, tag):
    from capture_processing.utils.mamma import find_existing_mamma_params
    baseline = find_existing_mamma_params(subject / 'mamma_output', tag, subject, 'images')
    if baseline is None:
        raise ValueError('Confidence fitting requires completed default MAMMA outputs')
    config = baseline.parent / 'run_config.yaml'
    run_args = json.loads((baseline.parent / 'run_args.json').read_text())
    # Input observations remain untouched and shared by both fits.
    cap = subject / 'mamma_output/ma_cap' / tag / subject.name
    landmarks = subject / 'mamma_output/ma_2d' / tag / subject.name
    cameras = run_args['cam_names']
    paths = [subject / 'smpl_params.npz', config, baseline.parent / 'run_args.json']
    paths += [cap / 'images/gt' / f'{camera}.npz' for camera in cameras]
    paths += [landmarks / 'images' / f'{camera}.npz' for camera in cameras]
    if not cameras or any(not path.is_file() for path in paths):
        raise ValueError('Missing default MAMMA observations, calibration, or run configuration')
    dependencies = [PROJECT / 'capture_processing/utils/mamma_confidence.py', PROJECT / 'capture_processing/tools/run_mamma_confidence.py',
                    repo / 'optimization/run_ma_3d.py',
                    repo / 'optimization/utils/optimization.py', repo / 'optimization/utils/fitting.py',
                    repo / 'optimization/losses/losses.py', repo / 'optimization/utils/triangulation_functions.py']
    model_root, matrix = _assets(repo, run_args)
    models = [model_root / 'smplx' / f'SMPLX_{gender}.npz' for gender in ('NEUTRAL', 'MALE', 'FEMALE')] + [matrix]
    if any(not path.is_file() for path in models):
        raise ValueError('Missing MAMMA SMPL-X body models or joint matrix')
    # Subject inputs live on shared storage; models live in each worker's own
    # checkout, so they are identified by content to stay valid across workers.
    payload = {'version': 2, 'threshold': .5, 'max_reprojection_px': 50.,
               'inputs': [file_stamp(path) for path in paths],
               'models': [content_stamp(path) for path in models],
               'dependencies': [hashlib.sha256(path.read_bytes()).hexdigest() for path in dependencies]}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return baseline, config, run_args, cap, landmarks, signature


def _read_manifest(subject):
    try:
        value = json.loads((subject / MANIFEST).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def confidence_fit_complete(subject, repo=None, tag='vci'):
    import numpy as np
    try:
        manifest = _read_manifest(subject)
        with np.load(subject / 'smpl_params.npz') as default:
            count = len(default['transl'])
        if not manifest.get('complete') or not valid_smplx(subject / PARAMS, count):
            return False
        if repo is not None and inputs(subject, repo, tag)[-1] != manifest['signature']:
            return False
        artifacts = manifest.get('artifacts', {})
        root = Path('mamma_confidence_output')
        folder = root / 'ma_3d' / tag / subject.name / 'images'
        viz = root / 'ma_vis' / tag / subject.name / 'images'
        required = {PARAMS, 'smplx_confidence_fit_diagnostics.json',
                    str(folder / 'smplx_params_body_id-00.npz'), str(folder / 'verts_joints_body_id-00.npz'),
                    str(viz / 'scene.rrd'), str(viz / 'preview.mp4')}
        required.update(str(viz / 'overlay' / name) for name in manifest.get('overlay_files', []))
        return bool(manifest.get('overlay_files')) and required <= artifacts.keys() and all(stamp is not None and stamp[1] > 0 and
                    file_stamp(subject / name) == stamp for name, stamp in artifacts.items())
    except (OSError, ValueError, KeyError, TypeError, IndexError, zipfile.BadZipFile):
        return False


def _valid_upstream(folder, count):
    import numpy as np
    try:
        with np.load(folder / 'smplx_params_body_id-00.npz') as p, np.load(folder / 'verts_joints_body_id-00.npz') as m:
            return (p['smplx_pose'].shape == (count, 165)
                    and p['smplx_translation'].shape == (count, 3)
                    and np.isfinite(p['smplx_pose']).all() and np.isfinite(p['smplx_translation']).all()
                    and p['smplx_betas'].size > 0 and np.isfinite(p['smplx_betas']).all()
                    and p['triangulated_3d_pts'].shape == (count, 512, 3)
                    and np.isfinite(p['triangulated_3d_pts']).all(-1).any(-1).all()
                    and all(m[key].shape[0] == count and np.isfinite(m[key]).all()
                            for key in ('pred_vertices', 'pred_joints')))
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile):
        return False


def _avoid_hugepage_stalls():
    # Large model arrays can spend minutes in kernel compaction on busy hosts.
    # Linux inherits this process-local setting across fork/exec; no host setting
    # is changed. Unsupported platforms retain their normal allocation policy.
    import sys
    if sys.platform == 'linux':
        import ctypes
        ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE


def ensure_confidence_fit(subject, args):
    if not requested(args):
        return
    return _ensure_confidence_fit(subject, args)


@subject_locked
def _ensure_confidence_fit(subject, args):
    if not requested(args):
        return
    _avoid_hugepage_stalls()
    import numpy as np
    from capture_processing.utils.mamma import resolve_mamma_repo, _mamma_subprocess_env, _apply_mamma_device_env
    from capture_processing.utils.smplx_utils import save_mamma_smpl_params
    from capture_processing.utils.fit_metrics import prepare_smplx_fit_diagnostics
    subject = Path(subject).resolve()
    repo = resolve_mamma_repo(getattr(args, 'mamma_repo_path', None))
    tag = getattr(args, 'mamma_out_tag', 'vci')
    baseline, config, original, cap, landmarks, signature = inputs(subject, repo, tag)
    force = getattr(args, 'mamma_force', False)
    if not force and confidence_fit_complete(subject, repo, tag):
        print(f'Using complete confidence-filtered MAMMA fit: {subject / PARAMS}', flush=True)
        return
    root = subject / 'mamma_confidence_output'
    previous = _read_manifest(subject)
    if force or previous.get('signature') != signature:
        if root.exists():
            shutil.rmtree(root)
        (subject / PARAMS).unlink(missing_ok=True)
        (subject / 'smplx_confidence_fit_diagnostics.json').unlink(missing_ok=True)
        previous = {}
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'complete': False, 'signature': signature, 'visibility_threshold': .5,
                'max_reprojection_px': 50., 'baseline': str(baseline),
                'observations': str(landmarks / 'images'), 'artifacts': previous.get('artifacts', {})}
    def persist():
        path = subject / MANIFEST
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=2) + '\n')
        tmp.replace(path)
    def changed(path):
        old = manifest['artifacts'].get(str(path.relative_to(subject)))
        return old is not None and old != file_stamp(path)
    def record(paths):
        for path in paths:
            manifest['artifacts'][str(path.relative_to(subject))] = file_stamp(path)
        persist()
    persist()
    folder = root / 'ma_3d' / tag / subject.name / 'images'
    destination = folder.parent
    visualization = root / 'ma_vis' / tag / subject.name
    python = shlex.split(getattr(args, 'mamma_python', 'conda run -n mamma python'))
    env = _mamma_subprocess_env()
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('NPY_MADVISE_HUGEPAGE', '0')
    env['PYTHONPATH'] = os.pathsep.join([str(PROJECT), str(repo), env.get('PYTHONPATH', '')])
    _apply_mamma_device_env(env, getattr(args, 'device', None))
    with np.load(subject / 'smpl_params.npz') as data:
        count = len(data['transl'])
    def run(command, cwd, log):
        print(f'Confidence-filtered MAMMA: {log.name}', flush=True)
        with log.open('w') as output:
            try:
                subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f'Confidence-filtered MAMMA failed; see {log}') from exc
    upstream_files = [folder / 'smplx_params_body_id-00.npz', folder / 'verts_joints_body_id-00.npz']
    refit = not _valid_upstream(folder, count) or any(changed(path) for path in upstream_files)
    if refit:
        # Upstream reuses any params/mesh pair, including damaged files. Only
        # remove the extra fit's stage; the baseline and observations are shared.
        if destination.exists():
            shutil.rmtree(destination)
        model_root, matrix = _assets(repo, original)
        command = [*python, '-m', 'capture_processing.tools.run_mamma_confidence',
                   '--config_file', str(config), '--seq_name', 'images',
                   '--ma_2d_dir', str(landmarks), '--ma_cap_dir', str(cap),
                   '--out_path', str(destination), '--smplx-models', str(model_root),
                   '--downsampled-verts', str(matrix), '--skip_detection_analysis',
                   '--cam_names', *original['cam_names']]
        for key in ('start_frame', 'end_frame', 'ignore_start_frames'):
            if original.get(key):
                command += ['--' + key, str(original[key])]
        run(command, repo / 'optimization', root / 'fit.log')
        if not _valid_upstream(folder, count):
            raise ValueError(f'Confidence fit produced incomplete/nonfinite geometry: {folder}')
    record(upstream_files)
    if refit:
        (subject / PARAMS).unlink(missing_ok=True)
        (subject / 'smplx_confidence_fit_diagnostics.json').unlink(missing_ok=True)
        if visualization.exists():
            shutil.rmtree(visualization)
    params = folder / 'smplx_params_body_id-00.npz'
    if changed(subject / PARAMS) or not valid_smplx(subject / PARAMS, count):
        save_mamma_smpl_params(subject, params, filename=PARAMS)
    if not valid_smplx(subject / PARAMS, count):
        raise ValueError('Invalid confidence-filtered SMPL-X export')
    record([subject / PARAMS])
    if changed(subject / 'smplx_confidence_fit_diagnostics.json'):
        (subject / 'smplx_confidence_fit_diagnostics.json').unlink(missing_ok=True)
    prepare_smplx_fit_diagnostics(subject, params, getattr(args, 'mamma_python', 'conda run -n mamma python'), kind=KIND)
    diagnostics = json.loads((subject / 'smplx_confidence_fit_diagnostics.json').read_text())
    if not diagnostics.get('metrics', {}).get('reprojection_error', {}).get('summary', {}).get('count'):
        raise ValueError('Confidence fit reprojection validation failed: ' + str(diagnostics.get('unavailable')))
    record([subject / 'smplx_confidence_fit_diagnostics.json'])
    viz_folder = visualization / 'images'
    required = [viz_folder / 'scene.rrd', viz_folder / 'preview.mp4']
    required += [viz_folder / 'overlay' / f'{camera}.mp4' for camera in sorted(original['cam_names'])[:4]]
    if any(changed(path) for path in required) or not all(path.is_file() and path.stat().st_size > 0 for path in required):
        run([*python, str(repo / 'visualization/run_ma_vis.py'), '--ma_2d_dir', str(landmarks),
             '--ma_3d_dir', str(destination), '--seq_name', 'images', '--out_path', str(visualization),
             '--ma_cap_dir', str(cap)], repo / 'visualization', root / 'visualization.log')
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise ValueError('Confidence fit visualization is incomplete')
    artifacts = [subject / PARAMS, subject / 'smplx_confidence_fit_diagnostics.json', params,
                 folder / 'verts_joints_body_id-00.npz', *required, *sorted((viz_folder / 'overlay').glob('*.mp4'))]
    manifest['artifacts'] = {str(path.relative_to(subject)): file_stamp(path) for path in artifacts}
    manifest['overlay_files'] = [path.name for path in required[2:]]
    manifest['complete'] = True
    persist()
    print(f'Confidence-filtered MAMMA fit ready: {subject / PARAMS}', flush=True)


def validate_confidence_metrics(subject, report_path):
    """A successful pipeline exit must include the new fit's requested metrics."""
    from capture_processing.utils.stage_cache import fit_artifacts
    report = json.loads(Path(report_path).read_text())
    fit = report.get('fits', {}).get(KIND, {})
    if (not report.get('complete') or fit.get('status') != 'complete'
            or report.get('fit_artifacts') != fit_artifacts(subject)
            or 'reprojection_error' not in fit.get('metrics', {})
            or 'silhouette_iou' not in fit.get('metrics', {})):
        raise RuntimeError(f'Confidence-filtered fit metrics are incomplete; rerun to compute only missing metrics: {report_path}')
