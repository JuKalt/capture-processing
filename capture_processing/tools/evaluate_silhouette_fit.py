"""Reconstruct a cached fitted mesh and score it against shared subject masks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from capture_processing.utils.fit_metrics import metric, summary
from capture_processing.utils.silhouette_metrics import binary_mask, render_silhouette, silhouette_counts


def load_sam3d_head(repo):
    """Load just the rig head and its checkpoint buffers, without the image encoder."""
    os.environ['MOMENTUM_ENABLED'] = '0'  # Match SAM3D's TorchScript rig convention.
    sys.path.insert(0, str(repo / 'src'))
    from capture_processing.tools.run_mhr import _install_simplecv_triangulate_compat
    _install_simplecv_triangulate_compat()
    import torch
    from huggingface_hub import snapshot_download
    from sam3d_body.utils.config import get_config
    from sam3d_body.models.heads import build_head
    from sam3d_body.utils.checkpoint import load_state_dict

    # Estimation already populated this cache; metrics never download a new model.
    snapshot = Path(snapshot_download('facebook/sam-3d-body-dinov3', local_files_only=True))
    config = get_config(str(snapshot / 'model_config.yaml'))
    config.defrost()
    config.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(snapshot / 'assets/mhr_model.pt')
    config.freeze()
    head = build_head(config, 'mhr')
    head.hand_pose_comps_ori = torch.nn.Parameter(head.hand_pose_comps.clone(), requires_grad=False)
    head.hand_pose_comps.data = torch.eye(54).float()
    try:
        checkpoint = torch.load(snapshot / 'model.ckpt', map_location='cpu', weights_only=False, mmap=True)
    except RuntimeError:
        checkpoint = torch.load(snapshot / 'model.ckpt', map_location='cpu', weights_only=False)
    state = checkpoint.get('state_dict', checkpoint)
    buffers = {key.removeprefix('head_pose.'): value for key, value in state.items() if key.startswith('head_pose.')}
    required = {'scale_mean', 'scale_comps', 'hand_pose_mean', 'hand_pose_comps',
                'hand_joint_idxs_left', 'hand_joint_idxs_right', 'joint_rotation'}
    if not required.issubset(buffers):
        raise ValueError(f'SAM3D checkpoint is missing rig buffers: {sorted(required - buffers.keys())}')
    load_state_dict(head, buffers, strict=False)
    return head.cpu().eval()


def mesh_provider(subject, request):
    """Return a world-space mesh accessor for one fit, in meters."""
    import torch
    from capture_processing.tools.visualize_mhr import camera_mesh_to_world, reference_camera
    kind = request['kind']
    data = np.load(subject / request['file'], allow_pickle=False)
    anchor = 'global_orient' if kind == 'smplx' else ('global_rot' if kind == 'mhr' else 'lbs_model_params')
    ids = data['frame_ids'] if 'frame_ids' in data else np.arange(len(data[anchor]))
    indices = {int(frame): i for i, frame in enumerate(ids)}
    torch.set_num_threads(1)

    def tensor(name, index, width=None):
        if name not in data:
            return torch.zeros(1, width)
        value = np.asarray(data[name], dtype=np.float32)
        if value.ndim == 1:
            value = value[None]
        value = value[0:1] if len(value) == 1 else value[index:index + 1]
        return torch.from_numpy(value.copy())

    if kind == 'smplx':
        import smplx
        shape = np.asarray(data['betas'])
        num_betas = shape.shape[-1]
        expression_width = data['expression'].shape[-1] if 'expression' in data else 10
        model = smplx.SMPLX(model_path=request['model'], gender='neutral', use_pca=False,
                            flat_hand_mean=True, num_betas=num_betas,
                            num_expression_coeffs=expression_width, batch_size=1).eval()
        faces = np.asarray(model.faces)
        fields = {'global_orient': 3, 'transl': 3, 'body_pose': 63, 'betas': num_betas,
                  'jaw_pose': 3, 'leye_pose': 3, 'reye_pose': 3, 'expression': expression_width,
                  'left_hand_pose': 45, 'right_hand_pose': 45}
        def evaluate(index):
            with torch.inference_mode():
                return model(**{name: tensor(name, index, width) for name, width in fields.items()}).vertices[0].numpy()
    elif kind == 'mhrfromsmplx':
        sys.path.insert(0, request['official_repo'])
        from mhr.mhr import MHR
        lod = int(data['lod'].item()) if 'lod' in data else 1
        model = MHR.from_files(folder=Path(request['official_repo']) / 'assets', device=torch.device('cpu'), lod=lod).eval()
        faces = np.asarray(model.character.mesh.faces)
        def evaluate(index):
            with torch.inference_mode():
                vertices, _ = model(identity_coeffs=tensor('identity_coeffs', index),
                                    model_parameters=tensor('lbs_model_params', index),
                                    face_expr_coeffs=tensor('face_expr_coeffs', index), apply_correctives=True)
            return vertices[0].numpy() / 100.
    else:
        from capture_processing.tools.run_mhr import _load_calibration
        calibration = _load_calibration(subject / 'calibration.json')
        head = load_sam3d_head(Path(request['sam3d_repo']))
        faces = np.asarray(data['faces'])
        def evaluate(index):
            camera = reference_camera(data, index)
            with torch.inference_mode():
                vertices = head.mhr_forward(
                    global_trans=torch.zeros(1, 3), global_rot=tensor('global_rot', index),
                    body_pose_params=tensor('body_pose', index), hand_pose_params=tensor('hand_pose', index),
                    scale_params=tensor('scale', index), shape_params=tensor('shape', index),
                    expr_params=tensor('expression', index), return_keypoints=False)
            # return_keypoints=False yields the vertices tensor directly.
            return camera_mesh_to_world(vertices[0].numpy(), data['translation'][index], calibration[camera])
    previous = [None, None]
    def get_mesh(frame):
        if previous[0] != frame:
            previous[:] = [frame, evaluate(indices[frame])]
        return previous[1], faces
    return get_mesh, data.close


def evaluate(subject, request, provider=None):
    """Every worker reads exactly the same view list; unreadable views fail explicitly."""
    from PIL import Image
    get_mesh, close = provider(subject, request) if provider else mesh_provider(subject, request)
    rows = []
    try:
        for view in request['views']:
            vertices, faces = get_mesh(view['frame_id'])
            with Image.open(view['mask']) as image:
                target = binary_mask(np.asarray(image.convert('L')))
            with Image.open(view['image']) as image:
                image_width, image_height = image.size
            camera = request['calibration'][view['camera']]
            K = np.asarray(camera['K'], dtype=float).reshape(3, 3).copy()
            height, width = target.shape
            K[0] *= width / image_width
            K[1] *= height / image_height
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = np.asarray(camera['R']).reshape(3, 3)
            extrinsics[:3, 3] = np.asarray(camera['T']).reshape(3)
            prediction = render_silhouette(vertices, faces, K, extrinsics, target.shape)
            rows.append({'camera': view['camera'], 'frame_id': view['frame_id'],
                         **silhouette_counts(prediction, target)})
            if len(rows) == 1 or len(rows) % 50 == 0:
                print(f"{request['kind']}: {len(rows)}/{len(request['views'])} silhouette views", flush=True)
    finally:
        close()
    frames = sorted({row['frame_id'] for row in rows})
    metrics = {}
    for field, name in [('iou', 'silhouette_iou'), ('silhouette_error', 'silhouette_error')]:
        values = [np.mean([row[field] for row in rows if row['frame_id'] == frame]) for frame in frames]
        result = metric(values, 'fraction', 'Mask intersection/union (higher is better).' if field == 'iou' else '1 - mask IoU (lower is better).')
        result['frame_ids'] = frames
        result['per_view'] = [{'camera': row['camera'], 'frame_id': row['frame_id'], 'value': row[field]} for row in rows]
        result['view_summary'] = summary([row[field] for row in rows])
        result['per_camera'] = {camera: summary([row[field] for row in rows if row['camera'] == camera])
                                for camera in sorted({row['camera'] for row in rows})}
        union = sum(row['union'] for row in rows)
        intersection = sum(row['intersection'] for row in rows)
        pooled_iou = intersection / union if union else 1.
        result['pooled_value'] = pooled_iou if field == 'iou' else 1 - pooled_iou
        metrics[name] = result
    return {'signature': request['signature'], 'metrics': metrics,
            'pixel_counts_per_view': [{key: row[key] for key in ('camera', 'frame_id', 'intersection', 'union', 'xor', 'pixels')} for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    subject = args.subject_root.resolve()
    request = json.loads(args.request.read_text())
    result = evaluate(subject, request)
    output = subject / f"silhouette_metrics.{request['kind']}.json"
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(f'Silhouette metrics ready: {output}', flush=True)


if __name__ == '__main__':
    main()
