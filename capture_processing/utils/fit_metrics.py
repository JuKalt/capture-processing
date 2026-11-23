"""Per-subject fit diagnostics without treating fitted observations as ground truth."""
from __future__ import annotations

from capture_processing.utils.subject_lock import subject_locked

import hashlib
import json
from pathlib import Path

import numpy as np

FIT_FILES = {'smplx': 'smpl_params.npz', 'smplx_confidence': 'smpl_params_confidence.npz', 'mhr': 'mhr_params.npz',
             'mhrfromsmplx': 'mhr_from_smplx_params.npz'}
FIELDS = {
    'smplx': {'global_orient': 3, 'transl': 3, 'body_pose': 63, 'betas': None},
    'mhr': {'global_rot': 3, 'body_pose': 133, 'translation': 3, 'shape': 45,
            'scale': 28, 'hand_pose': 108, 'expression': 72},
    'mhrfromsmplx': {'lbs_model_params': 204, 'identity_coeffs': 45, 'face_expr_coeffs': 72},
}


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    return {'count': int(len(finite)), 'missing_count': int(len(values) - len(finite)),
            **{key: (float(func(finite)) if len(finite) else None) for key, func in {
                'mean': np.mean, 'median': np.median, 'p95': lambda x: np.percentile(x, 95),
                'max': np.max}.items()}}


def metric(values, units, description):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {'units': units, 'description': description, 'summary': summary(values),
            'per_frame': [float(value) if np.isfinite(value) else None for value in values]}


def reprojection_errors(points_world, observations, K, world_to_camera, visibility=None):
    """Mean Euclidean pixel residual per frame; ignore invalid/behind-camera points."""
    points = np.asarray(points_world)
    observations = np.asarray(observations)[..., :2]
    if points.ndim != 3 or points.shape[-1] != 3 or observations.shape != (*points.shape[:2], 2):
        raise ValueError('Landmark and fitted point shapes do not match')
    camera = points @ np.asarray(world_to_camera)[:3, :3].T + np.asarray(world_to_camera)[:3, 3]
    homogeneous = camera @ np.asarray(K).T
    with np.errstate(divide='ignore', invalid='ignore'):
        projected = homogeneous[..., :2] / homogeneous[..., 2:]
    valid = np.isfinite(projected).all(-1) & np.isfinite(observations).all(-1)
    valid &= (camera[..., 2] > 0) & (observations.sum(-1) > 0)
    if visibility is not None:
        valid &= np.isfinite(visibility) & (visibility >= .5)
    errors = np.linalg.norm(projected - observations, axis=-1)
    totals = np.where(valid, errors, 0).sum(-1)
    counts = valid.sum(-1)
    means = np.full(len(points), np.nan)
    np.divide(totals, counts, out=means, where=counts > 0)
    return means, counts, totals


def root_motion(positions, ids, valid):
    """Displacement and second difference on adjacent local frame IDs only."""
    positions = np.asarray(positions, dtype=np.float64)
    step = np.full(len(ids), np.nan)
    acceleration = step.copy()
    pairs = valid[1:] & valid[:-1] & (np.diff(ids) == 1)
    step[1:] = np.where(pairs, np.linalg.norm(np.diff(positions, axis=0), axis=1), np.nan)
    if len(ids) > 2:
        triples = pairs[1:] & pairs[:-1]
        acceleration[2:] = np.where(triples, np.linalg.norm(np.diff(positions, n=2, axis=0), axis=1), np.nan)
    return {'root_step_m': metric(step, 'm/local_frame', 'Root translation displacement between consecutive local frames; motion diagnostic, not accuracy.'),
            'root_second_difference_m': metric(acceleration, 'm/local_frame^2', 'Root translation second difference; motion diagnostic, not physical acceleration without FPS.')}


FIELDS['smplx_confidence'] = FIELDS['smplx']


def _expected_ids(subject):
    with_source = subject / FIT_FILES['smplx']
    if with_source.is_file():
        try:
            with np.load(with_source, allow_pickle=False) as data:
                return np.asarray(data['frame_ids'] if 'frame_ids' in data else np.arange(len(data['global_orient'])), dtype=np.int64)
        except (ValueError, OSError, KeyError):
            pass
    images = subject / 'images'
    cameras = sorted(p for p in images.iterdir() if p.is_dir()) if images.is_dir() else []
    ids = set()
    if cameras:
        for path in cameras[0].iterdir():
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'}:
                continue
            try:
                ids.add(int(path.stem.split('_')[-1]))
            except ValueError:
                pass
    return np.asarray(sorted(ids), dtype=np.int64)


def _fit(subject, kind, expected):
    path = subject / FIT_FILES[kind]
    entry = {'file': path.name, 'status': 'missing', 'metrics': {}, 'unavailable': {}}
    if not path.is_file():
        return entry
    try:
        with np.load(path, allow_pickle=False) as data:
            anchor = next(iter(FIELDS[kind]))
            count = len(data[anchor])
            ids = np.asarray(data['frame_ids'] if 'frame_ids' in data else np.arange(count), dtype=np.int64)
            if ids.shape != (count,) or len(np.unique(ids)) != count:
                raise ValueError('Invalid frame IDs')
            valid = np.ones(count, dtype=bool)
            fields = dict(FIELDS[kind])
            if kind in ('smplx', 'smplx_confidence'):
                fields.update({name: width for name, width in {'jaw_pose': 3, 'left_hand_pose': 45,
                               'right_hand_pose': 45, 'expression': None, 'leye_pose': 3, 'reye_pose': 3}.items() if name in data})
            for name, width in fields.items():
                value = np.asarray(data[name])
                if name == 'betas' and value.ndim == 1:
                    value = value[None]
                if value.ndim != 2 or value.shape[0] not in (1, count) or not value.shape[1] or (width and value.shape[1] != width):
                    raise ValueError(f'Invalid {name} shape {value.shape}')
                valid &= np.broadcast_to(np.isfinite(value).all(axis=1), (count,))
            if 'checkpointed_frames' in data:
                valid[int(data['checkpointed_frames'].item()):] = False
            entry.update(status='complete' if bool(data.get('complete', np.asarray(True)).item()) else 'incomplete',
                         frame_ids=ids.tolist(), valid_frame_ids=ids[valid].tolist(), finite_frames=int(valid.sum()), total_frames=count,
                         expected_frames=len(expected), valid_frame_fraction=float(valid.mean()) if count else None,
                         coverage_fraction=float(np.isin(expected, ids[valid]).mean()) if len(expected) else None)
            metrics = entry['metrics']
            if kind in ('smplx', 'smplx_confidence'):
                positions = data['transl']
            elif kind == 'mhrfromsmplx':
                positions = data['lbs_model_params'][:, :3] / 100.
            else:
                positions = None
                calibration_path = subject / 'calibration.json'
                if 'reference_cam_names' in data and calibration_path.is_file():
                    calibration = json.loads(calibration_path.read_text())
                    positions = np.full((count, 3), np.nan)
                    for i, reference in enumerate(data['reference_cam_names']):
                        if str(reference) in calibration:
                            camera = calibration[str(reference)]
                            R = np.asarray(camera['R']).reshape(3, 3)
                            T = np.asarray(camera['T']).reshape(3)
                            positions[i] = (data['translation'][i] - T) @ R
                else:
                    entry['unavailable']['root_motion'] = 'SAM3D translations require recorded reference cameras and calibration for world-space motion.'
            if positions is not None:
                metrics.update(root_motion(positions, ids, valid & np.isfinite(positions).all(-1)))
            native = {'mhr': [('loss_total', 'optimizer_total', 'weighted_loss', 'Last recorded SAM3D optimizer objective; backend-specific, not comparable to other fits.'),
                              ('reprojection_error_px', 'reprojection_error', 'px', 'Mean Euclidean residual to SAM3D detected keypoints after the final update.'),
                              ('triangulated_keypoint_error_mm', 'triangulated_keypoint_error', 'mm', 'Mean Euclidean residual to valid triangulated SAM3D keypoints; observation consistency, not ground truth.')],
                      'mhrfromsmplx': [('fitting_errors_cm', 'conversion_vertex_error', 'cm', 'Official mean vertex distance to the barycentrically mapped SMPL-X surface.')], 'smplx': [], 'smplx_confidence': []}
            for field, name, units, description in native[kind]:
                if field in data:
                    value = np.asarray(data[field], dtype=float)
                    if value.shape != (count,):
                        raise ValueError(f'Invalid {field} shape')
                    metrics[name] = metric(np.where(valid, value, np.nan), units, description)
                else:
                    entry['unavailable'][name] = f'{field} was not saved in this fit archive.'
            if kind == 'mhr' and 'num_views' in data:
                entry['view_counts'] = {'multi_view_frames': int(((data['num_views'] >= 2) & valid).sum()),
                                        'single_view_frames': int(((data['num_views'] == 1) & valid).sum())}
            if kind in ('smplx', 'smplx_confidence'):
                sidecar = subject / f'{kind}_fit_diagnostics.json'
                diagnostics = json.loads(sidecar.read_text()) if sidecar.is_file() else None
                if diagnostics and diagnostics.get('source_sha256') == hashlib.sha256(path.read_bytes()).hexdigest():
                    metrics.update(diagnostics['metrics'])
                    entry['unavailable'].update(diagnostics.get('unavailable', {}))
                else:
                    entry['unavailable']['reprojection_error'] = 'No matching MAMMA landmark/mesh diagnostics; unavailable for imported fits without observations.'
            entry['unavailable']['ground_truth_accuracy'] = 'No independent ground-truth joints or surface supplied; MPJPE/PVE are not computed.'
    except (OSError, ValueError, KeyError, TypeError) as exc:
        entry.update(status='invalid', error=str(exc))
    return entry


@subject_locked
def write_fit_metrics(subject_root: Path, silhouette_options=None) -> Path:
    subject = Path(subject_root)
    from capture_processing.utils.stage_cache import metrics_signature, metric_artifacts, fit_artifacts
    from capture_processing.utils.silhouette_metrics import SilhouetteOptions
    silhouette_options = silhouette_options or SilhouetteOptions()
    evaluation_options = {'cameras': list(silhouette_options.cameras) if silhouette_options.cameras else None,
                          'max_frames': silhouette_options.max_frames}
    signature = metrics_signature(subject, silhouette_options)
    output = subject / 'fit_metrics.json'
    if output.is_file() and not silhouette_options.force:
        try:
            cached = json.loads(output.read_text())
            if (isinstance(cached, dict) and cached.get('complete') is True and cached.get('input_signature') == signature
                    and cached.get('cache_artifacts') == metric_artifacts(subject)
                    and cached.get('evaluation_options') == evaluation_options):
                print(f'Using existing complete fit metrics: {output}', flush=True)
                return output
        except (OSError, ValueError, TypeError):
            pass
    expected = _expected_ids(subject)
    report = {'schema_version': 2, 'subject': subject.name,
              'notes': ['Silhouette metrics share the same masks/views across fits. Other fit-specific residuals use different observations.',
                        'Root translations are model origins; temporal diagnostics are not joint or surface accuracy.'],
              'fits': {kind: _fit(subject, kind, expected) for kind in FIT_FILES
                       if kind != 'smplx_confidence' or (subject / FIT_FILES[kind]).is_file()
                       or (subject / 'mamma_confidence_fit.json').is_file()}}
    from capture_processing.utils.silhouette_metrics import evaluate_shared_silhouettes
    report['silhouette_evaluation'] = evaluate_shared_silhouettes(subject, report['fits'], silhouette_options)
    report['evaluation_options'] = evaluation_options
    report['input_signature'] = signature
    report['cache_artifacts'] = metric_artifacts(subject)
    report['fit_artifacts'] = fit_artifacts(subject)
    report['complete'] = (report['silhouette_evaluation'].get('complete_comparison', False)
                          and all(fit['status'] in ('complete', 'missing') for fit in report['fits'].values()))
    temporary = output.with_name(output.name + '.tmp')
    contents = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if not output.is_file() or output.read_text() != contents:
        temporary.write_text(contents)
        temporary.replace(output)
    return output



def validate_fit_metrics(report_path, required_fits):
    """Never report successful processing with failed/missing requested metrics."""
    try:
        report = json.loads(Path(report_path).read_text())
        fits = report.get('fits', {})
        if (report.get('complete') is True
                and report.get('silhouette_evaluation', {}).get('complete_comparison') is True
                and all(fits.get(kind, {}).get('status') == 'complete'
                        and 'silhouette_iou' in fits[kind].get('metrics', {})
                        and 'silhouette_error' in fits[kind].get('metrics', {})
                        for kind in required_fits)):
            return
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    raise RuntimeError(f'Required fit metrics are incomplete: {report_path}. '
                       'See silhouette_metrics.*.log; rerun to retry missing metrics using cached fits.')

def prepare_smplx_fit_diagnostics(subject_root: Path, params_path: Path, python_executable: str, kind="smplx") -> None:
    """Use the MAMMA environment to read its tensor correspondence asset."""
    import os
    import shlex
    import subprocess
    source = subject_root / FIT_FILES[kind]
    diagnostics_path = subject_root / f'{kind}_fit_diagnostics.json'
    if diagnostics_path.is_file():
        try:
            diagnostics = json.loads(diagnostics_path.read_text())
            if diagnostics.get('source_sha256') == hashlib.sha256(source.read_bytes()).hexdigest() and diagnostics.get('metrics'):
                return
        except (OSError, ValueError, AttributeError):
            pass
    project = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project) + os.pathsep + env.get('PYTHONPATH', '')
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['OPENBLAS_NUM_THREADS'] = '1'
    subprocess.run([*shlex.split(python_executable), '-m', 'capture_processing.tools.evaluate_smplx_fit',
                    '--subject-root', str(subject_root.resolve()), '--mamma-params', str(params_path.resolve()), '--kind', kind],
                   cwd=project, env=env, check=True)
