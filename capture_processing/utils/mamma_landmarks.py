"""Batched masked MAMMA landmarks, launched from the upstream landmarks cwd.

Reuse the upstream crop implementation and model; CPU threads prepare frames
while a single model performs GPU inference. No external checkout edits needed.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runpy
import sys


def ordered_prefetch(function, indices, workers, capacity):
    """Bound prepared frames in memory and propagate worker exceptions in order."""
    iterator = iter(indices)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(capacity):
            index = next(iterator, None)
            if index is None:
                break
            pending.append(pool.submit(function, index))
        while pending:
            result = pending.popleft().result()
            index = next(iterator, None)
            if index is not None:
                pending.append(pool.submit(function, index))
            yield result


def camera_has_no_detections(mask_dir):
    """Recognize an empty cache or an explicitly skipped, completed SAM view."""
    import numpy as np

    camera_dir = mask_dir.parent
    cache = camera_dir / 'masks.npy'
    if cache.is_file():
        # Do not load a large, nonempty segmentation just to inspect its IDs.
        if cache.stat().st_size >= 1024 * 1024:
            return False
        segmentation = np.load(cache, allow_pickle=True).item()
        return (isinstance(segmentation, dict) and bool(segmentation)
                and all(isinstance(body, dict) and 'mask' in body and len(body['mask']) == 0
                        for body in segmentation.values()))

    # SAM's no-matched-detection branch returns without writing masks.npy or
    # creating masks/. Require its exact diagnostic and overall completion;
    # missing files alone must never turn a broken export into an empty view.
    sequence_dir = camera_dir.parent
    log = sequence_dir / 'run.log'
    if not camera_dir.is_dir() or not (sequence_dir / 'DONE').is_file() or not log.is_file():
        return False
    diagnostic = (f"No valid detections found for any ID in '{camera_dir.name}'. "
                  'Skipping propagation. Check matching.start_frame/samples and detection thresholds.')
    with log.open(errors='replace') as stream:
        return any(line.rstrip().endswith(diagnostic) for line in stream)


def process_masked_camera(source, device, model, cfg, out_folder, masks_path,
                          dataset_type, workers=4, batch_size=8):
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from tqdm import tqdm

    camera = str(source.cam_name)
    destination = Path(out_folder) / f"{camera}.npz"
    if destination.is_file():
        return
    mask_dir = Path(masks_path) / camera / "masks"
    ids = [int(path.stem.split('_')[2]) for path in mask_dir.glob('*.png')]
    if not ids:
        # A completed segmentation can legitimately detect nobody in a view
        # (e.g. a ceiling camera). Distinguish this from lost mask PNGs: only
        # accept an explicit empty segmentation result, with bodies observed
        # in other cameras to establish the export's body dimension.
        empty = camera_has_no_detections(mask_dir)
        sibling_ids = ([int(path.stem.split('_')[2])
                        for path in Path(masks_path).glob('*/masks/mask_*_*.png')]
                       if empty else [])
        if not sibling_ids:
            raise ValueError(f"No mask files found in {mask_dir}")
        ids = sibling_ids
        print(f'{camera}: segmentation contains no detections; exporting zero-visibility '
              'observations for this camera.', flush=True)
    n_frames = len(source)
    if not n_frames:
        raise ValueError(f"No frames found for camera {camera}")
    vertices, visibility, contacts, floor_contacts = [], [], [], []
    # Video readers contain mutable decoder state; use one preprocessing thread.
    camera_workers = 1 if hasattr(source, 'video_path') else workers
    for body_id in range(1, max(ids) + 1):
        body_verts = np.zeros((n_frames, cfg.num_joints, 3))
        body_vis = np.zeros((n_frames, cfg.num_joints))
        body_contact = np.zeros_like(body_vis)
        body_floor = np.zeros_like(body_vis)

        def prepare(index):
            mask_path = mask_dir / f'mask_{index:04d}_{body_id:02d}.png'
            if not mask_path.is_file():
                return index, None
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or np.sum(mask) / mask.size < 0.01:
                return index, None
            ys, xs = np.where(mask)
            boxes = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]])
            image = cv2.cvtColor(source.read_rgb(index), cv2.COLOR_RGB2BGR)
            dataset = dataset_type(cfg, image, mask, boxes=boxes)
            return index, dataset[0]

        def infer(samples):
            indices, items = zip(*samples)
            batch = default_collate(items)
            # Geometry stays on the CPU; only model inputs need GPU transfer.
            images = batch['img'].to(device)
            masks = batch['mask'].to(device)
            with torch.inference_mode():
                out = model(images, masks)
                joints = out['joints2d'].cpu().numpy().copy()
                if cfg.data['train']['normalize_plus_min_one']:
                    joints[:, :, :2] = (joints[:, :, :2] + 1) / 2
                centers = batch['box_center'].numpy()
                sizes = batch['box_size'].numpy()
                joints[:, :, :2] = (joints[:, :, :2] * sizes[:, None, :]
                                    - sizes[:, None, :] / 2 + centers[:, None, :])
                body_verts[list(indices)] = joints
                for key, target in [('visibility', body_vis), ('contact', body_contact),
                                    ('floor_contact', body_floor)]:
                    target[list(indices)] = torch.sigmoid(out[key].squeeze(-1)).cpu().numpy()

        samples = []
        prepared = ordered_prefetch(prepare, range(n_frames), camera_workers,
                                    max(batch_size * 2, camera_workers * 2))
        for index, item in tqdm(prepared, total=n_frames, desc=f'{camera} body {body_id}'):
            if item is not None:
                samples.append((index, item))
            if len(samples) == batch_size:
                infer(samples)
                samples.clear()
        if samples:
            infer(samples)
        vertices.append(body_verts)
        visibility.append(body_vis)
        contacts.append(body_contact)
        floor_contacts.append(body_floor)

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents interrupted runs being mistaken for cached outputs.
    temporary = destination.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, landmarks=np.stack(vertices, axis=1),
                 visibilities=np.stack(visibility, axis=1),
                 contacts=np.stack(contacts, axis=1),
                 floor_contacts=np.stack(floor_contacts, axis=1),
                 contacts_gt=None, floor_contacts_gt=None)
    temporary.replace(destination)


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--workers', type=int, default=4)
    extra.add_argument('--batch-size', type=int, default=8)
    options, remaining = extra.parse_known_args()
    if options.workers < 1 or options.batch_size < 1:
        raise SystemExit('MAMMA workers and batch size must be positive')
    sys.argv = [sys.argv[0], *remaining]
    landmarks_dir = Path.cwd()
    sys.path.insert(0, str(landmarks_dir))
    upstream = runpy.run_path(str(landmarks_dir / 'run_ma_2d.py'), run_name='_mamma_upstream')
    # Some worker checkouts rename the upstream inference entry point while
    # adding their own camera-worker pool. Keep both interfaces supported.
    predict = upstream.get('main') or upstream.get('run_predictions')
    if not callable(predict):
        raise RuntimeError('MAMMA landmarks entry point missing: expected main or run_predictions '
                           f'in {landmarks_dir / "run_ma_2d.py"}')
    args = upstream['parser']()
    masks = str(Path(args.mask_path) / args.seq_name) if args.mask_path else None
    output = str(Path(args.out_folder) / args.seq_name)
    images = str(Path(args.img_folder) / args.seq_name / 'gt') if args.img_folder else None
    accelerated = masks is not None and not args.save_cam_output
    if accelerated:
        # This wrapper owns CPU prefetching and runs one GPU model. Disable
        # upstream camera subprocesses so they cannot lose our patched globals.
        if hasattr(args, 'workers'):
            args.workers = 1
        namespace = predict.__globals__
        # MAMMA masks already provide boxes; detector weights are never used.
        namespace['DefaultPredictor_Lazy'] = lambda config: None
        def process(source, detector, device, model, cfg, out_folder,
                    save_cam_output, masks_path=None, **kwargs):
            return process_masked_camera(source, device, model, cfg, out_folder,
                                         masks_path, upstream['ViTDetDataset'],
                                         options.workers, options.batch_size)
        namespace['process_data'] = process
        print(f'MAMMA landmarks: {options.workers} CPU workers, batch size {options.batch_size}, one model', flush=True)
    predict(args, out_folder=output, masks_folder=masks, img_folder=images)
    if args.save_cam_output and args.seq_name:
        upstream['make_videos_from_args'](args)


if __name__ == '__main__':
    main()
