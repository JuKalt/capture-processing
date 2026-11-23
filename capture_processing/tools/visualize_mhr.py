"""Visualize cached MHR estimates as calibrated overlays and a Rerun recording."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def reference_camera(data, index, override=None):
    """Resolve recorded references; legacy full-view frames have an unambiguous first camera."""
    cameras = [str(value) for value in data['cam_names']]
    recorded = str(data['reference_cam_names'][index]) if 'reference_cam_names' in data else ''
    camera = override or recorded
    if not camera and int(data['num_views'][index]) == len(cameras) and cameras:
        camera = sorted(cameras)[0]
    if not camera:
        raise ValueError(f"Frame {int(data['frame_ids'][index])} has no recorded reference camera. "
                         "For legacy partial-view estimates, pass --reference-camera with the actual first successful camera.")
    if camera not in cameras:
        raise ValueError(f"Unknown reference camera: {camera}")
    return camera


def camera_mesh_to_world(vertices, translation, camera):
    """Mirror SAM3D: flip FK Y/Z, add camera translation, then invert calibration."""
    import numpy as np
    points = np.array(vertices, dtype=np.float32, copy=True)
    points[:, 1:] *= -1
    points += translation
    extrinsics = camera['world_to_cam']
    return (points - extrinsics[:3, 3]) @ extrinsics[:3, :3]


def mesh_overlay(image, vertices_world, faces, camera, alpha=.55):
    """Draw shaded, depth-sorted triangles using calibrated pinhole projection."""
    import cv2
    import numpy as np
    extrinsics = camera['world_to_cam']
    points = vertices_world @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    projected = points @ camera['K'].T
    uv = projected[:, :2] / np.maximum(projected[:, 2:], 1e-6)
    triangles = points[faces]
    polygons = uv[faces]
    height, width = image.shape[:2]
    valid = (triangles[:, :, 2] > 1e-5).all(axis=1)
    valid &= np.isfinite(polygons).all(axis=(1, 2))
    valid &= (polygons[:, :, 0].max(axis=1) >= 0) & (polygons[:, :, 0].min(axis=1) < width)
    valid &= (polygons[:, :, 1].max(axis=1) >= 0) & (polygons[:, :, 1].min(axis=1) < height)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    shade = .45 + .55 * np.abs(normals @ np.array([.3, -.4, -.866]))
    colors = (shade[:, None] * np.array([225, 190, 35])).clip(0, 255).astype(np.uint8)
    result = image.copy()
    visible = np.flatnonzero(valid)
    for face in visible[np.argsort(triangles[visible, :, 2].mean(axis=1))[::-1]]:
        polygon = np.rint(polygons[face]).clip(-100000, 100000).astype(np.int32)
        cv2.fillConvexPoly(result, polygon, tuple(int(value) for value in colors[face]))
    return cv2.addWeighted(result, alpha, image, 1 - alpha, 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, required=True)
    parser.add_argument('--kind', choices=['mhr', 'mhrfromsmplx'], default='mhr')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--repo-path', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cameras', nargs='+')
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--long-edge', type=int, default=1024)
    parser.add_argument('--reference-camera', help='Actual first successful camera for ambiguous legacy estimates')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.max_frames, args.frame_step, args.long_edge, args.fps) <= 0:
        parser.error('Frame count, step, image size, and FPS must be positive')
    return args


def main():
    args = parse_args()
    from capture_processing.utils.stage_cache import rendering_signature, rendering_complete, file_stamp
    subject = args.subject_root.expanduser().resolve()
    folder = 'mhr_visualization' if args.kind == 'mhr' else 'mhr_from_smplx_visualization'
    output = (args.output_dir or subject / folder).expanduser().resolve()
    repo = (args.repo_path or Path(__file__).resolve().parents[2] / ('external/sam3d-body-rerun' if args.kind == 'mhr' else 'external/MHR')).expanduser().resolve()
    settings = {key: getattr(args, key) for key in ('cameras', 'max_frames', 'frame_step', 'fps', 'long_edge', 'reference_camera')}
    signature = rendering_signature(subject, repo, args.kind, settings)
    if not args.force and rendering_complete(output, signature):
        print(f'Using existing complete visualization: {output}', flush=True)
        return
    # Remove the completion marker before overwriting any artifacts.
    (output / 'manifest.json').unlink(missing_ok=True)
    from capture_processing.utils.mhr import resolve_sam3d_body_rerun_repo
    from capture_processing.tools.run_mhr import _load_calibration, _install_simplecv_triangulate_compat, _run_with_log
    from capture_processing.utils.silhouette_metrics import media_path
    import cv2
    import numpy as np
    import torch
    import rerun as rr

    subject = args.subject_root.expanduser().resolve()
    calibration = _load_calibration(subject / 'calibration.json')
    data = np.load(subject / ('mhr_params.npz' if args.kind == 'mhr' else 'mhr_from_smplx_params.npz'), allow_pickle=False)
    cameras = args.cameras or sorted(calibration)[:2]
    if any(camera not in calibration for camera in cameras):
        raise SystemExit('Selected cameras must exist in calibration.json')
    numeric = (['global_rot', 'body_pose', 'translation', 'shape', 'scale', 'hand_pose', 'expression']
               if args.kind == 'mhr' else ['lbs_model_params', 'identity_coeffs', 'face_expr_coeffs'])
    valid = np.ones(len(data['frame_ids']), dtype=bool)
    for key in numeric:
        valid &= np.isfinite(data[key]).all(axis=1)
    if 'checkpointed_frames' in data:
        valid[int(data['checkpointed_frames']):] = False
    indices = np.flatnonzero(valid)[::args.frame_step][:args.max_frames]
    if not len(indices):
        raise SystemExit('No fitted finite MHR frames are available to visualize')
    try:
        references = ({int(index): reference_camera(data, index, args.reference_camera) for index in indices}
                      if args.kind == 'mhr' else {})
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if any(camera not in calibration for camera in references.values()):
        raise SystemExit("Reference camera is missing from calibration.json")
    selected_frame_ids = [int(data["frame_ids"][index]) for index in indices]
    if args.kind == 'mhr':
        repo = resolve_sam3d_body_rerun_repo(repo)
    if args.kind == 'mhr':
        sys.path.insert(0, str(repo / 'src'))
        _install_simplecv_triangulate_compat()
    torch.backends.cudnn.enabled = False
    output.mkdir(parents=True, exist_ok=True)
    close_mesh = lambda: None
    if args.kind == 'mhr':
        from sam3d_body.build_models import load_sam_3d_body_hf
        print('Loading cached MHR model for visualization.', flush=True)
        model = _run_with_log(output / 'model_loading.log',
                             lambda: load_sam_3d_body_hf(repo_id='facebook/sam-3d-body-dinov3')[0])
        head = model.head_pose.to(args.device).eval()
        faces = np.asarray(data['faces'], dtype=np.int32)
    else:
        from capture_processing.tools.evaluate_silhouette_fit import mesh_provider
        get_mesh, close_mesh = mesh_provider(subject, {'kind': args.kind, 'file': 'mhr_from_smplx_params.npz',
                                                      'official_repo': str(repo)})
    label = 'MHR' if args.kind == 'mhr' else 'MHR from SMPL-X'
    rr.init(f'{label}: {subject.name}', spawn=False)
    rr.save(str(output / 'scene.rrd'))
    rr.log('world', rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    writers, sizes, missing_images = {}, {}, []
    try:
        for index in indices:
            frame_id = int(data['frame_ids'][index])
            if args.kind == 'mhr':
                tensor = lambda key: torch.as_tensor(data[key][index:index + 1], device=args.device).float()
                with torch.inference_mode():
                    vertices = head.mhr_forward(global_trans=torch.zeros(1, 3, device=args.device),
                                                global_rot=tensor('global_rot'), body_pose_params=tensor('body_pose'),
                                                hand_pose_params=tensor('hand_pose'), scale_params=tensor('scale'),
                                                shape_params=tensor('shape'), expr_params=tensor('expression'),
                                                return_keypoints=False)[0].cpu().numpy()
                world = camera_mesh_to_world(vertices, data['translation'][index], calibration[references[int(index)]])
            else:
                world, faces = get_mesh(frame_id)
            rr.set_time('frame', sequence=frame_id)
            rr.log('world/mhr', rr.Mesh3D(vertex_positions=world, triangle_indices=faces,
                                        vertex_colors=[35, 190, 225]))
            if 'triangulated_keypoints_3d' in data:
                points = data['triangulated_keypoints_3d'][index]
                good = np.isfinite(points).all(axis=1) & (points[:, 3] > 0)
                rr.log('world/triangulated_keypoints', rr.Points3D(points[good, :3], colors=[255, 100, 70], radii=.003))
            for name in cameras:
                path = media_path(subject / 'images', name, frame_id, 'color')
                image = cv2.imread(str(path)) if path else None
                if image is None:
                    missing_images.append({'camera': name, 'frame': frame_id})
                    continue
                height, width = image.shape[:2]
                scale = min(1., args.long_edge / max(height, width))
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = cv2.resize(image, size)
                camera = dict(calibration[name])
                camera['K'] = camera['K'].copy()
                camera['K'][0, :] *= size[0] / width
                camera['K'][1, :] *= size[1] / height
                overlay = mesh_overlay(image, world, faces, camera)
                side_by_side = np.concatenate([image, overlay], axis=1)
                cv2.putText(side_by_side, f'{name} | frame {frame_id} | source / {label}', (12, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
                if name not in writers:
                    writer = cv2.VideoWriter(str(output / f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                                             (size[0] * 2, size[1]))
                    if not writer.isOpened():
                        raise RuntimeError(f'Could not open MP4 writer for {name}')
                    writers[name], sizes[name] = writer, size
                    cv2.imwrite(str(output / f'{name}_preview.png'), side_by_side)
                    extrinsics = camera['world_to_cam']
                    rr.log(f'world/cameras/{name}', rr.Transform3D(mat3x3=extrinsics[:3, :3].T,
                           translation=-extrinsics[:3, :3].T @ extrinsics[:3, 3]), static=True)
                    rr.log(f'world/cameras/{name}/image', rr.Pinhole(image_from_camera=camera['K'],
                           resolution=size, camera_xyz=rr.ViewCoordinates.RDF), static=True)
                if sizes[name] != size:
                    raise ValueError(f'Image dimensions changed within camera {name}')
                writers[name].write(side_by_side)
                rr.log(f'world/cameras/{name}/image', rr.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=85))
    finally:
        for writer in writers.values():
            writer.release()
        rr.disconnect()
        data.close()
        close_mesh()
    manifest = {'subject': str(subject), 'frame_ids': selected_frame_ids,
                'reference_cameras': {str(frame_id): references[int(index)] for index, frame_id in zip(indices, selected_frame_ids) if int(index) in references},
                'cameras': cameras, 'fps': args.fps, 'frame_step': args.frame_step, 'missing_images': missing_images,
                'rendering': 'Depth-sorted shaded triangles; overlays use a painter approximation, not a per-pixel depth buffer.'}
    artifacts = ['scene.rrd'] + [name + suffix for name in cameras for suffix in ('.mp4', '_preview.png')]
    if missing_images or any(not (output / name).is_file() or (output / name).stat().st_size == 0 for name in artifacts):
        raise RuntimeError(f'Visualization incomplete: missing images or artifacts in {output}')
    manifest.update(complete=True, signature=signature, kind=args.kind,
                    artifacts={name: file_stamp(output / name) for name in artifacts})
    temporary = output / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(output / 'manifest.json')
    print(f'Wrote MHR visualization to {output}', flush=True)


if __name__ == '__main__':
    main()
