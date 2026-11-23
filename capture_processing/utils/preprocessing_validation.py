"""Fail-fast input contracts shared by preprocessing and read-only validation."""
from pathlib import Path


def validate_options(args):
    for name in ('mamma_workers', 'mamma_batch_size', 'mhr_workers', 'mhr_iterations',
                 'gpu_workers', 'mhr_max_frames', 'max_subjects', 'resize_long_edge'):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if getattr(args, 'workers', 0) < 0:
        raise ValueError('--workers must be >= 0')
    if not 1 <= getattr(args, 'jpeg_quality', 95) <= 100:
        raise ValueError('--jpeg-quality must be between 1 and 100')


def validate_cameras(data):
    import numpy as np
    cameras = data.get('cameras')
    if not isinstance(cameras, list) or not cameras:
        raise ValueError('Calibration requires a nonempty cameras list')
    seen = set()
    for camera in cameras:
        try:
            cid = camera['camera_id'].upper()
            if not cid or Path(cid).name != cid or cid in ('.', '..') or cid in seen:
                raise ValueError(f'Invalid or duplicate camera ID: {cid!r}')
            seen.add(cid)
            k = np.asarray(camera['intrinsics']['camera_matrix'], dtype=float).reshape(3, 3)
            view = np.asarray(camera['extrinsics']['view_matrix'], dtype=float).reshape(4, 4)
            size = np.asarray(camera['intrinsics']['resolution'], dtype=float)
            if (size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any()
                    or not np.equal(size, np.floor(size)).all()):
                raise ValueError(f'Invalid resolution for {cid}')
            if (not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                    or not np.allclose(k[2], [0, 0, 1])):
                raise ValueError(f'Invalid intrinsics for {cid}')
            r = view[:3, :3]
            if (not np.isfinite(view).all() or not np.allclose(view[3], [0, 0, 0, 1])
                    or not np.allclose(r @ r.T, np.eye(3), atol=1e-3)
                    or not np.isclose(np.linalg.det(r), 1, atol=1e-3)):
                raise ValueError(f'Invalid rigid world-to-camera transform for {cid}')
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f'Malformed camera calibration: {camera!r}') from exc


def validate_source_media(jobs, cameras, mapping):
    from PIL import Image
    for image, output, mask, _, _, _ in jobs:
        if any(path.suffix.lower() == ".avif" for path in (image, mask)):
            import pillow_avif  # noqa: F401  (register AVIF support)
        cid = mapping[output.parent.name]
        camera = cameras[cid]
        expected = (camera['W'], camera['H'])
        with Image.open(image) as decoded:
            if decoded.size != expected:
                raise ValueError(f'Calibration/image resolution mismatch for {cid}: '
                                 f'{image}: {decoded.size}, calibration: {expected}')
        with Image.open(mask) as decoded:
            if decoded.size != expected:
                raise ValueError(f'Image/mask resolution mismatch for {cid}: '
                                 f'{mask}: {decoded.size}, expected: {expected}')
