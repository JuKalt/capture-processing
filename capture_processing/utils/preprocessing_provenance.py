"""Content identities for baseline fitting dependencies, independent of output exports."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=128)
def _digest_at_stamp(path, size, mtime, ctime):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def content_identity(path):
    path = Path(path)
    if not path.is_file():
        return None
    stamp = path.stat()
    return _digest_at_stamp(str(path.resolve()), stamp.st_size, stamp.st_mtime_ns, stamp.st_ctime_ns)


def baseline_dependencies(args, subject=None):
    from capture_processing.utils.mamma import (default_mamma_repo, _resolve_mamma_asset_path,
                                 _preset_ma2d_weights, _preset_step_enabled, _preset_ma_masks_uses_sam2)
    repo = Path(getattr(args, 'mamma_repo_path', None) or os.environ.get('MAMMA_REPO_PATH')
                or default_mamma_repo()).expanduser().resolve()
    preset = Path(getattr(args, 'mamma_preset', None) or 'configs/examples/presets/full.yaml').expanduser()
    preset = preset if preset.is_absolute() else repo / preset
    try:
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    files = {name: content_identity(repo / name) for name in (
        'optimization/run_ma_3d.py', 'optimization/utils/optimization.py',
        'optimization/utils/fitting.py', 'optimization/losses/losses.py',
        'optimization/utils/triangulation_functions.py', 'capture/run_ma_cap.py',
        'landmarks/run_ma_2d.py', 'segmentation/run_ma_masks.py', 'segmentation/core/pipeline.py')}
    preset_text = preset.read_text() if preset.is_file() else ''
    weights = _preset_ma2d_weights(preset_text)
    assets = {
        'landmarks': _resolve_mamma_asset_path(repo, os.environ, 'MAMMA_MA2D_CHECKPOINT',
                         weights or 'data/weights/ma_2d/mamma_mask_full_cvpr.ckpt'),
        'correspondence': _resolve_mamma_asset_path(repo, os.environ, 'MAMMA_DOWNSAMPLED_VERTS_PKL',
                         'data/body_models/downsampled_verts/verts_512.pkl'),
    }
    if _preset_step_enabled(preset_text, 'ma_masks'):
        assets['detector'] = _resolve_mamma_asset_path(repo, os.environ, 'MAMMA_YOLO_CHECKPOINT',
                                                       'data/weights/yolo/yolo12x.pt')
        if _preset_ma_masks_uses_sam2(preset_text):
            assets['segmentation'] = _resolve_mamma_asset_path(repo, os.environ, 'MAMMA_SAM2_CHECKPOINT',
                                                                'data/weights/sam2/sam2.1_hiera_large.pt')
    models = _resolve_mamma_asset_path(repo, os.environ, 'MAMMA_SMPLX_LOCKHEAD_MODELS',
                                      'data/body_models/smplx_locked_head')
    assets.update({f'smplx_{gender}': models / 'smplx' / f'SMPLX_{gender}.npz'
                   for gender in ('NEUTRAL', 'MALE', 'FEMALE')})
    adapters = {name: content_identity(PROJECT / 'capture_processing' / name) for name in (
        'core/smplx_pipeline.py', 'utils/smplx_utils.py', 'utils/mamma.py',
        'utils/mamma_landmarks.py', 'utils/preprocessing_provenance.py')}
    masks = Path(subject) / 'mamma_input_masks' if subject is not None else None
    supplied = [[str(path.relative_to(masks)), content_identity(path)]
                for path in sorted(masks.rglob('*')) if path.is_file()] if masks and masks.is_dir() else None
    payload = {'version': 1, 'supplied_masks': supplied, 'revision': revision, 'fitting_code': files, 'adapters': adapters,
               'preset_sha256': content_identity(preset),
               'preset': str(preset) if not preset.is_file() else None,
               'assets': {key: content_identity(value) for key, value in assets.items()},
               'output_tag': getattr(args, 'mamma_out_tag', 'vci'),
               'source': getattr(args, 'smplx_source', 'mamma')}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return signature, payload
