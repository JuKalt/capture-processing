"""Convert canonical world-space SMPL-X parameters using Meta's official fitter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import sys

import numpy as np

from capture_processing.utils.smplx_mhr import OUTPUT_NAME, conversion_signature

POSE_FIELDS = {'global_orient': 3, 'body_pose': 63, 'jaw_pose': 3,
               'left_hand_pose': 45, 'right_hand_pose': 45, 'transl': 3,
               'leye_pose': 3, 'reye_pose': 3}


def load_parameters(path: Path):
    """Broadcast constant shape/expression; reject missing or invalid poses."""
    with np.load(path, allow_pickle=False) as data:
        root = np.asarray(data['global_orient'], dtype=np.float32)
        if root.ndim != 2 or root.shape[1] != 3 or not len(root):
            raise ValueError('global_orient must have nonempty shape [frames, 3]')
        count = len(root)
        fields = dict(POSE_FIELDS, betas=None, expression=None)
        parameters = {}
        for name, width in fields.items():
            if name not in data:
                if name not in {'jaw_pose', 'left_hand_pose', 'right_hand_pose', 'leye_pose', 'reye_pose', 'expression'}:
                    raise ValueError(f'Missing SMPL-X parameter: {name}')
                value = np.zeros((count, width or 10), dtype=np.float32)
            else:
                value = np.asarray(data[name], dtype=np.float32)
                if value.ndim == 1:
                    value = value[None, :]
                if value.ndim != 2 or value.shape[0] not in (1, count) or (width and value.shape[1] != width) or not value.shape[1]:
                    raise ValueError(f'Invalid SMPL-X {name} shape: {value.shape}')
                value = np.broadcast_to(value, (count, value.shape[1])).copy()
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite SMPL-X parameter: {name}')
            parameters[name] = value
        ids = np.asarray(data['frame_ids'] if 'frame_ids' in data else np.arange(count))
        if ids.shape != (count,) or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != count:
            raise ValueError('frame_ids must contain one unique integer per pose')
    return parameters, ids


def save_conversion(output, result, frame_ids, faces, signature):
    parameters = {name: value.detach().cpu().numpy() for name, value in result.result_parameters.items()}
    count = len(frame_ids)
    for name, width in [('lbs_model_params', 204), ('identity_coeffs', 45), ('face_expr_coeffs', 72)]:
        value = parameters[name]
        if value.shape == (1, width):
            value = np.repeat(value, count, axis=0)
        if value.shape != (count, width) or not np.isfinite(value).all():
            raise ValueError(f'Invalid converted {name}: {value.shape}')
        parameters[name] = value
    errors = np.asarray(result.result_errors)
    if errors.shape != (count,) or not np.isfinite(errors).all():
        raise ValueError(f'Invalid fitting errors: {errors.shape}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp.npz')
    np.savez_compressed(temporary, **parameters, frame_ids=frame_ids, faces=faces,
                        fitting_errors_cm=errors, coordinate_system=np.asarray('world'),
                        units=np.asarray('centimeters'), source=np.asarray('official-smplx-to-mhr'),
                        backend=np.asarray('pymomentum'), lod=np.asarray(1),
                        conversion_signature=np.asarray(signature), complete=np.asarray(True))
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, required=True)
    parser.add_argument('--repo-path', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    args = parser.parse_args()
    source = args.subject_root.resolve() / 'smpl_params.npz'
    repo, model_path = args.repo_path.resolve(), args.model_path.resolve()
    arrays, frame_ids = load_parameters(source)
    signature = conversion_signature(source, model_path, repo)
    sys.path[:0] = [str(repo / 'tools/mhr_smpl_conversion'), str(repo)]
    # The official converter uses relative paths for its correspondence assets.
    os.chdir(repo / 'tools/mhr_smpl_conversion')
    random.seed(0)
    np.random.seed(0)
    import torch
    import smplx
    from mhr.mhr import MHR
    from conversion import Conversion

    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = smplx.SMPLX(model_path=str(model_path), gender='neutral', use_pca=False,
                        flat_hand_mean=True, num_betas=arrays['betas'].shape[1],
                        num_expression_coeffs=arrays['expression'].shape[1], batch_size=1)
    mhr = MHR.from_files(folder=repo / 'assets', device=torch.device('cpu'), lod=1)
    converter = Conversion(mhr_model=mhr, smpl_model=model, method='pymomentum')
    result = converter.convert_smpl2mhr(
        smpl_parameters={name: torch.from_numpy(value) for name, value in arrays.items()},
        single_identity=True, is_tracking=True, return_mhr_parameters=True, return_fitting_errors=True)
    output = source.parent / OUTPUT_NAME
    save_conversion(output, result, frame_ids, np.asarray(mhr.character.mesh.faces), signature)
    print(f'Official MHR conversion ready: {output}', flush=True)


if __name__ == '__main__':
    main()
