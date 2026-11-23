"""Evaluate MAMMA's final fitted surface against its saved dense 2D detections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from capture_processing.utils.fit_metrics import FIT_FILES, metric, reprojection_errors
from capture_processing.utils.mamma import resolve_mamma_repo
from capture_processing.utils.smplx_hands import MARKER, mamma_hands_mean


def evaluate(subject: Path, params_path: Path, kind="smplx"):
    source = subject / FIT_FILES[kind]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {'source_sha256': digest, 'metrics': {}, 'unavailable': {}}
    try:
        with np.load(source) as canonical, np.load(params_path) as upstream:
            # Never attach residuals from another/stale MAMMA fit to this archive.
            pose = upstream['smplx_pose']
            betas = np.asarray(upstream['smplx_betas'], dtype=np.float32)
            if betas.ndim == 1:
                betas = betas[None]
            elif betas.shape[0] != 1:
                betas = np.median(betas, axis=0, keepdims=True).astype(np.float32)
            left, right = pose[:, 75:120], pose[:, 120:165]
            if MARKER in canonical.files and bool(canonical[MARKER]):
                # Exported in the flat-hand convention (see smplx_hands).
                run_args = json.loads((params_path.parent / 'run_args.json').read_text())
                models = Path(run_args.get('smplx_models') or '')
                mean_left, mean_right = (mamma_hands_mean(models=models) if (models / 'smplx/SMPLX_NEUTRAL.npz').is_file()
                                         else mamma_hands_mean(resolve_mamma_repo(None)))
                left = (left.astype(np.float32) + mean_left).astype(np.float32)
                right = (right.astype(np.float32) + mean_right).astype(np.float32)
            if not (np.array_equal(canonical['global_orient'], pose[:, :3]) and
                    np.array_equal(canonical['body_pose'], pose[:, 3:66]) and
                    np.array_equal(canonical['jaw_pose'], pose[:, 66:69]) and
                    np.array_equal(canonical['left_hand_pose'], left) and
                    np.array_equal(canonical['right_hand_pose'], right) and
                    np.array_equal(canonical['betas'], betas) and
                    np.all(canonical['expression'] == 0) and
                    np.array_equal(canonical['transl'], upstream['smplx_translation'])):
                raise ValueError('MAMMA parameters do not match canonical SMPL-X fit')
        folder = params_path.parent
        config = json.loads((folder / 'run_args.json').read_text())
        matrix_path = Path(config['downsampled_verts'])
        if not matrix_path.is_absolute():
            matrix_path = Path(__file__).resolve().parents[2] / 'external/mamma/optimization' / matrix_path
        # MAMMA's own local model asset stores a tensor in pickle format.
        matrix = np.load(matrix_path, allow_pickle=True)
        if hasattr(matrix, 'detach'):
            matrix = matrix.detach().cpu().numpy()
        matrix = np.asarray(matrix)
        with np.load(folder / 'verts_joints_body_id-00.npz') as meshes:
            vertices = meshes['pred_vertices']
        if matrix.ndim != 2 or matrix.shape[1] != vertices.shape[1]:
            raise ValueError('Dense surface correspondence does not match mesh')
        points = np.empty((len(vertices), len(matrix), 3), dtype=np.float32)
        for start in range(0, len(vertices), 32):
            points[start:start + 32] = np.einsum('kv,fvc->fkc', matrix, vertices[start:start + 32], optimize=True)
        totals = np.zeros(len(points))
        counts = np.zeros(len(points), dtype=np.int64)
        by_camera = {}
        first = config.get('start_frame') or 0
        for camera_name in config['cam_names']:
            landmarks_path = Path(config['ma_2d_dir']) / config['seq_name'] / f'{camera_name}.npz'
            camera_path = Path(config['ma_cap_dir']) / config['seq_name'] / 'gt' / f'{camera_name}.npz'
            with np.load(landmarks_path) as detections, np.load(camera_path) as calibration:
                observations = detections['landmarks'][first:first + len(points), 0]
                visibility = detections['visibilities'][first:first + len(points), 0] if 'visibilities' in detections else None
                per_frame, camera_counts, camera_totals = reprojection_errors(
                    points, observations, calibration['cam_int'], calibration['cam_ext'], visibility)
            by_camera[camera_name] = metric(per_frame, 'px', 'Mean Euclidean dense landmark residual; visibility >= 0.5 and positive depth.')
            totals += camera_totals
            counts += camera_counts
        errors = np.full(len(points), np.nan)
        np.divide(totals, counts, out=errors, where=counts > 0)
        reprojection = metric(errors, 'px', 'Mean Euclidean dense surface landmark residual across all retained cameras/points; visibility >= 0.5, positive depth. Different landmarks from SAM3D.')
        reprojection['per_camera'] = by_camera
        reprojection['observations_per_frame'] = counts.tolist()
        reprojection['observation_weighted_mean'] = float(totals.sum() / counts.sum()) if counts.sum() else None
        result['metrics']['reprojection_error'] = reprojection
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        result['unavailable']['reprojection_error'] = str(exc)
    output = subject / f'{kind}_fit_diagnostics.json'
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, required=True)
    parser.add_argument('--mamma-params', type=Path, required=True)
    parser.add_argument('--kind', choices=['smplx', 'smplx_confidence'], default='smplx')
    args = parser.parse_args()
    evaluate(args.subject_root.resolve(), args.mamma_params.resolve(), args.kind)


if __name__ == '__main__':
    main()
