"""Export MAMMA predictions as canonical sequence-level SMPL-X parameters."""
from __future__ import annotations
from pathlib import Path


def save_mamma_smpl_params(out_root: Path, mamma_params_path: Path, filename="smpl_params.npz") -> None:
    """Convert MAMMA ma_3d output to this dataset's smpl_params.npz format."""
    import numpy as np

    with np.load(mamma_params_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("smplx_pose", "smplx_betas", "smplx_translation", "frame_ids") if key in archive}
    pose = np.asarray(data["smplx_pose"], dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 165:
        raise SystemExit(
            f"Unexpected MAMMA smplx_pose shape in {mamma_params_path}: {pose.shape}"
        )

    betas = np.asarray(data["smplx_betas"], dtype=np.float32)
    if betas.ndim == 1:
        betas = betas.reshape(1, -1)
    elif betas.ndim == 2 and betas.shape[0] != 1:
        betas = np.median(betas, axis=0, keepdims=True).astype(np.float32)

    transl = np.asarray(data["smplx_translation"], dtype=np.float32)
    n_frames = pose.shape[0]
    if transl.shape != (n_frames, 3):
        raise SystemExit(
            f"Unexpected MAMMA smplx_translation shape in {mamma_params_path}: "
            f"{transl.shape}; expected {(n_frames, 3)}"
        )

    ids = np.asarray(data['frame_ids'] if 'frame_ids' in data else np.arange(n_frames))
    if (n_frames <= 0 or ids.shape != (n_frames,) or ids.dtype.kind not in 'iu'
            or (ids < 0).any() or len(np.unique(ids)) != n_frames
            or betas.ndim != 2 or betas.shape[0] != 1 or betas.shape[1] == 0
            or not all(np.isfinite(value).all() for value in (pose, transl, betas))):
        raise ValueError(f'Invalid MAMMA parameters or frame IDs: {mamma_params_path}')
    temporary = out_root / (filename + ".tmp.npz")
    np.savez(
        temporary,
        frame_ids=ids.astype(np.int64),
        global_orient=pose[:, 0:3],
        transl=transl,
        body_pose=pose[:, 3:66],
        jaw_pose=pose[:, 66:69],
        betas=betas,
        expression=np.zeros((n_frames, 10), dtype=np.float32),
        left_hand_pose=pose[:, 75:120],
        right_hand_pose=pose[:, 120:165],
    )

    temporary.replace(out_root / filename)
