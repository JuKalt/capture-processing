"""Shared-mask silhouette evaluation and caches for all fitted body models."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
VERSION = 1


@dataclass(frozen=True)
class SilhouetteOptions:
    smplx_python: str | None = None
    official_python: str | None = None
    sam3d_python: str | None = None
    smplx_model: Path | None = None
    official_repo: Path | None = None
    sam3d_repo: Path | None = None
    cameras: tuple[str, ...] | None = None
    max_frames: int | None = None
    force: bool = False


def options_from_pipeline(args) -> SilhouetteOptions:
    conversion = getattr(args, 'smplx_mhr_options', None)
    return SilhouetteOptions(
        smplx_python=getattr(args, 'mamma_python', None),
        official_python=conversion.python if conversion else None,
        sam3d_python=getattr(args, 'mhr_python', None),
        smplx_model=conversion.model_path if conversion else None,
        official_repo=conversion.repo_path if conversion else None,
        sam3d_repo=getattr(args, 'mhr_repo_path', None))


def resolve_paths(options):
    official = Path(options.official_repo or os.environ.get('MHR_REPO', PROJECT / 'external/MHR')).expanduser().resolve()
    sam3d = Path(options.sam3d_repo or os.environ.get('SAM3D_BODY_RERUN_REPO', PROJECT / 'external/sam3d-body-rerun')).expanduser().resolve()
    model = Path(options.smplx_model or os.environ.get('VCI_SMPLX_MODEL', PROJECT / 'external/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz')).expanduser().resolve()
    return official, sam3d, model


def media_path(root, camera, frame_id, prefix):
    """Resolve both prefixed VCI masks and numeric PNG/JPEG dataset frames."""
    for stem in (f'{prefix}_{frame_id:08d}', f'{frame_id:08d}', str(frame_id)):
        for suffix in ('.png', '.jpg', '.jpeg', '.webp'):
            path = root / camera / (stem + suffix)
            if path.is_file():
                return path
    return None


def binary_mask(pixels):
    """Preserve 0/1 masks; threshold 8-bit masks above 127 to suppress JPEG noise."""
    pixels = np.asarray(pixels)
    return pixels > (0 if pixels.max(initial=0) <= 1 else 127)


def silhouette_counts(prediction, target):
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError('Rendered silhouette and target mask dimensions must match')
    intersection = int(np.count_nonzero(prediction & target))
    union = int(np.count_nonzero(prediction | target))
    return {'intersection': intersection, 'union': union, 'xor': union - intersection,
            'pixels': int(target.size), 'iou': intersection / union if union else 1.,
            'silhouette_error': (union - intersection) / union if union else 0.}


def _clip_near(polygon, near):
    output = []
    previous = polygon[-1]
    for current in polygon:
        prev_in, curr_in = previous[2] >= near, current[2] >= near
        if prev_in != curr_in:
            ratio = (near - previous[2]) / (current[2] - previous[2])
            output.append(previous + ratio * (current - previous))
        if curr_in:
            output.append(current)
        previous = current
    return np.asarray(output)


def render_silhouette(vertices_world, faces, K, world_to_camera, shape):
    """Filled triangle union, with near-plane clipping and no backface culling.

    Visibility/depth ordering is unnecessary for a single opaque body's silhouette.
    All fits use this same non-antialiased CPU rasterizer at target-mask resolution.
    """
    from PIL import Image, ImageDraw
    height, width = shape
    vertices = np.asarray(vertices_world, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Mesh vertices must have finite shape [vertices, 3]')
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or (faces < 0).any() or (faces >= len(vertices)).any():
        raise ValueError('Invalid mesh triangle indices')
    extrinsics = np.asarray(world_to_camera)
    points = vertices @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    triangles = points[faces]
    near = 1e-5
    candidates = triangles[(triangles[..., 2] >= near).any(axis=1)]
    canvas = Image.new('1', (width, height), 0)
    drawing = ImageDraw.Draw(canvas)
    # Project all fully visible triangles in one batch, then union their fills.
    front = candidates[(candidates[..., 2] >= near).all(axis=1)]
    projected = front @ np.asarray(K).T
    polygons = projected[..., :2] / projected[..., 2:]
    if len(polygons):
        visible = (polygons[..., 0].max(axis=1) >= 0) & (polygons[..., 0].min(axis=1) < width)
        visible &= (polygons[..., 1].max(axis=1) >= 0) & (polygons[..., 1].min(axis=1) < height)
        for polygon in polygons[visible]:
            drawing.polygon([tuple(point) for point in np.rint(polygon).clip(-1e8, 1e8).astype(int)], fill=1)
    for triangle in candidates[~(candidates[..., 2] >= near).all(axis=1)]:
        clipped = _clip_near(triangle, near)
        projected = clipped @ np.asarray(K).T
        polygon = projected[:, :2] / projected[:, 2:]
        drawing.polygon([tuple(point) for point in np.rint(polygon).clip(-1e8, 1e8).astype(int)], fill=1)
    return np.asarray(canvas, dtype=bool)


def _file_stamp(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def shared_selection(subject, fits, options):
    """Intersection of valid frame IDs across all available fits; one mask list."""
    available = {kind: fit for kind, fit in fits.items() if fit['status'] in ('complete', 'incomplete')}
    if not available:
        return [], {}, 'No valid fits are available.'
    calibration_file = subject / 'calibration.json'
    if not calibration_file.is_file() or not (subject / 'masks').is_dir():
        return [], {}, 'Silhouette evaluation requires subject masks/ and calibration.json.'
    calibration = json.loads(calibration_file.read_text())
    cameras = sorted(set(options.cameras or calibration))
    if any(name not in calibration for name in cameras):
        raise ValueError('Requested silhouette camera is not in calibration.json')
    common = set.intersection(*(set(fit['valid_frame_ids']) for fit in available.values()))
    frame_ids = sorted(common)
    if options.max_frames is not None:
        if options.max_frames < 1:
            raise ValueError('Silhouette max_frames must be positive')
        frame_ids = frame_ids[:options.max_frames]
    views = []
    for frame_id in frame_ids:
        for camera in cameras:
            mask = media_path(subject / 'masks', camera, frame_id, 'mask')
            image = media_path(subject / 'images', camera, frame_id, 'color')
            if mask is not None and image is not None:
                views.append({'frame_id': frame_id, 'camera': camera, 'mask': str(mask.resolve()),
                              'image': str(image.resolve()), 'mask_stamp': _file_stamp(mask),
                              'image_stamp': _file_stamp(image)})
    return views, calibration, None if views else 'No common valid camera/frame pairs with both subject masks and source images.'


def _cache_matches(candidate, signature, views):
    if not isinstance(candidate, dict) or candidate.get('signature') != signature:
        return False
    metrics = candidate.get('metrics')
    if not isinstance(metrics, dict):
        return False
    expected = [(view['camera'], view['frame_id']) for view in views]
    for key in ('silhouette_iou', 'silhouette_error'):
        measurement = metrics.get(key)
        if not isinstance(measurement, dict):
            return False
        records = measurement.get('per_view', [])
        if not isinstance(records, list) or len(records) != len(expected):
            return False
        for record, pair in zip(records, expected):
            if not isinstance(record, dict) or (record.get('camera'), record.get('frame_id')) != pair:
                return False
            value = record.get('value')
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                return False
    return True


def _repo_stamp(path):
    """Checkout identity independent of where a worker keeps the checkout."""
    from capture_processing.utils.stage_cache import content_stamp
    try:
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    assets = {str(asset.relative_to(path)): content_stamp(asset) for asset in
              [path / 'pixi.lock', path / 'assets/lod1.fbx', path / 'assets/compact_v6_1.model',
               path / 'assets/corrective_blendshapes_lod1.npz'] if asset.is_file()}
    return {'revision': revision, 'assets': assets}


def _sam_asset_stamps():
    from capture_processing.utils.stage_cache import content_stamp
    cache = Path(os.environ.get('HF_HUB_CACHE', Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface')) / 'hub'))
    repository = cache / 'models--facebook--sam-3d-body-dinov3'
    reference = repository / 'refs/main'
    if not reference.is_file():
        return {}
    snapshot = repository / 'snapshots' / reference.read_text().strip()
    return {str(path.relative_to(snapshot)): content_stamp(path) for path in
            [snapshot / 'model.ckpt', snapshot / 'model_config.yaml', snapshot / 'assets/mhr_model.pt'] if path.is_file()}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def silhouette_signatures(subject, fits, options):
    """Common view selection plus per-fit cache signatures.

    Returns (views, calibration, reason, selection_signature, {kind: signature}).
    Everything signed is identical on every worker sharing the subject."""
    from capture_processing.utils.stage_cache import content_stamp, fit_digest
    views, calibration, reason = shared_selection(subject, fits, options)
    if reason:
        return views, calibration, reason, None, {}
    official, sam3d, model = resolve_paths(options)
    available = [kind for kind, fit in fits.items() if fit['status'] in ('complete', 'incomplete')]
    shared = {'version': VERSION, 'views': views, 'calibration': calibration,
              'model': content_stamp(model),
              'official_repo': _repo_stamp(official), 'sam3d_repo': _repo_stamp(sam3d),
              'sam_assets': _sam_asset_stamps() if 'mhr' in available else {}}
    signatures = {kind: _digest({'selection': shared, 'kind': kind,
                                 'source_sha256': fit_digest(subject / fits[kind]['file'])})
                  for kind in available}
    return views, calibration, None, _digest(shared), signatures


def evaluate_shared_silhouettes(subject, fits, options=None):
    options = options or SilhouetteOptions()
    views, calibration, reason, selection_signature, signatures = silhouette_signatures(subject, fits, options)
    selection = {'policy': 'intersection of valid frames across available fits, all calibrated mask/image views',
                 'frame_ids': sorted({view['frame_id'] for view in views}),
                 'cameras': sorted({view['camera'] for view in views}), 'view_count': len(views),
                 'mask_root': str((subject / 'masks').resolve()),
                 'available_fits': [kind for kind, fit in fits.items() if fit['status'] in ('complete', 'incomplete')],
                 'mask_threshold': '0/1 masks > 0; other masks > 127',
                 'rasterizer': 'filled triangle union, near-plane clipping, native mask resolution, no antialiasing'}
    if reason:
        for fit in fits.values():
            if fit['status'] in ('complete', 'incomplete'):
                fit['unavailable']['silhouette'] = reason
        selection['complete_comparison'] = False
        return selection
    official, sam3d, model = resolve_paths(options)
    selection['signature'] = selection_signature
    for kind, fit in fits.items():
        if fit['status'] not in ('complete', 'incomplete'):
            continue
        signature = signatures[kind]
        cache = subject / f'silhouette_metrics.{kind}.json'
        cached = None
        if cache.is_file() and not options.force:
            try:
                candidate = json.loads(cache.read_text())
                if _cache_matches(candidate, signature, views):
                    cached = candidate
            except (OSError, ValueError):
                pass
        if cached is None:
            official_python = official / '.pixi/envs/default/bin/python'
            sam_python = sam3d / '.pixi/envs/dev/bin/python'
            if kind == 'mhr':
                interpreter = options.sam3d_python or (str(sam_python) if sam_python.is_file() else sys.executable)
            else:
                interpreter = (options.smplx_python if kind in ('smplx', 'smplx_confidence') else options.official_python) or (str(official_python) if official_python.is_file() else sys.executable)
            request = subject / f'.silhouette_request.{kind}.json'
            request.write_text(json.dumps({'signature': signature, 'views': views, 'calibration': calibration,
                                           'kind': kind, 'file': fit['file'], 'official_repo': str(official),
                                           'sam3d_repo': str(sam3d), 'model': str(model)}, indent=2))
            env = os.environ.copy()
            env['PYTHONPATH'] = str(PROJECT) + os.pathsep + env.get('PYTHONPATH', '')
            env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            log = subject / f'silhouette_metrics.{kind}.log'
            try:
                with log.open('w') as output:
                    subprocess.run([*shlex.split(interpreter), '-m',
                                    ('capture_processing.tools.evaluate_confidence_silhouette' if kind == 'smplx_confidence'
                                     else 'capture_processing.tools.evaluate_silhouette_fit'),
                                    '--subject-root', str(subject.resolve()), '--request', str(request.resolve())],
                                   cwd=PROJECT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
                cached = json.loads(cache.read_text())
                if not _cache_matches(cached, signature, views):
                    raise ValueError('Silhouette worker returned incomplete or stale metrics')
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                fit['unavailable']['silhouette'] = f'{exc}; details: {log.name}'
                print(f'Silhouette metrics unavailable for {kind}; see {log}', flush=True)
                continue
            finally:
                request.unlink(missing_ok=True)
        fit['metrics'].update(cached['metrics'])
        fit['silhouette_selection_signature'] = selection_signature
    # A strict common subset prevents missing views from silently changing a comparison.
    selection['complete_comparison'] = all('silhouette_iou' in fit['metrics'] for fit in fits.values()
                                           if fit['status'] in ('complete', 'incomplete')) and bool(views)
    return selection
