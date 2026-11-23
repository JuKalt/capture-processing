"""Prepare undistorted DNA images and estimate new SMPL-X parameters."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path

from capture_processing.core.smplx_pipeline import ensure_subject_smplx
from capture_processing.utils.fit_metrics import write_fit_metrics
from capture_processing.utils.silhouette_metrics import options_from_pipeline
from capture_processing.utils.smplx_mhr import convert_subject_smplx_to_mhr, options_from_args
from multiview_data.dna import DNADataset
from multiview_data.dna_reader import close_reader as _close_reader, load_reader as _load_dna_reader
from capture_processing.utils.paths import paths_overlap


def prepare_subject(
    subject: dict[str, Any],
    destination: Path,
    camera_ids: list[int],
    frame_ids: list[int],
    resolution: int,
    reader_class=None,
) -> Path:
    """Export a selection with contiguous local frame IDs and explicit provenance.

    Completed exports are reusable only with identical inputs. Interrupted exports
    resume by overwriting media; fitting starts only after the completion marker.
    """
    import cv2
    import numpy as np

    destination = Path(destination)
    sources = {}
    for key in ("images", "smplx"):
        path = Path(subject[key]).resolve()
        stat = path.stat()
        sources[key] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    signature = {"version": 1, "sources": sources, "camera_ids": camera_ids,
                 "frame_ids": frame_ids, "resolution": resolution, "undistorted": True}
    marker = destination / ".dna_preparation.json"
    if destination.exists() and any(destination.iterdir()):
        if not marker.is_file():
            raise SystemExit(f"Refusing to reuse an unmanaged output directory: {destination}")
        previous = json.loads(marker.read_text())
        if previous["selection"] != signature:
            raise SystemExit(f"DNA selection or source changed; choose a new output directory: {destination}")
        if previous.get("complete"):
            expected = [destination / kind / f"cam{camera:02d}" / f"{frame:08d}.{suffix}"
                        for camera in camera_ids for frame in range(len(frame_ids))
                        for kind, suffix in (("images", "jpg"), ("masks", "png"))]
            expected.extend(destination / "mamma_input_masks" / "images" / f"cam{camera:02d}" / "masks" / f"mask_{frame:04d}_01.png"
                            for camera in camera_ids for frame in range(len(frame_ids)))
            expected.extend(destination / name for name in ("calibration.json", "camera_id_mapping.json", "frame_id_mapping.json"))
            if all(path.is_file() for path in expected):
                print(f"Reusing prepared DNA sequence: {destination}", flush=True)
                return destination
            raise SystemExit(f"Completed DNA export has missing files; choose a new output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"selection": signature, "complete": False}, indent=2))
    reader_type = reader_class or _load_dna_reader(subject)
    readers = []
    try:
        rgb = reader_type(str(subject["images"]))
        readers.append(rgb)
        annotations = reader_type(str(subject["smplx"]))
        readers.append(annotations)
        calibration = annotations.get_Calibration_all()
        capacity = int(rgb.get_Camera_5mp_info()["num_frame"])
        if not frame_ids or min(frame_ids) < 0 or max(frame_ids) >= capacity:
            raise SystemExit(f"Selected frames exceed DNA RGB capacity ({capacity})")
        missing = [c for c in camera_ids if f"{c:02d}" not in calibration or not 0 <= c < 48]
        if missing:
            raise SystemExit(f"Unavailable DNA 5MP cameras: {missing}")
        output_calibration = {}
        for camera_id in camera_ids:
            name = f"cam{camera_id:02d}"
            camera = calibration[f"{camera_id:02d}"]
            K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
            D = np.asarray(camera["D"], dtype=np.float64).reshape(-1)
            extrinsics = np.linalg.inv(np.asarray(camera["RT"], dtype=np.float64).reshape(4, 4))
            image_dir = destination / "images" / name
            mask_dir = destination / "masks" / name
            supplied_masks = destination / "mamma_input_masks" / "images" / name / "masks"
            for directory in (image_dir, mask_dir, supplied_masks):
                directory.mkdir(parents=True, exist_ok=True)
            image_size = None
            print(f"Exporting DNA camera {camera_id}: {len(frame_ids)} frames", flush=True)
            for local_id, source_id in enumerate(frame_ids):
                image = rgb.get_img("Camera_5mp", camera_id, "color", Frame_id=source_id, disable_tqdm=True)
                mask = annotations.get_mask(camera_id, Frame_id=source_id, disable_tqdm=True)
                if image is None or mask is None:
                    raise SystemExit(f"Missing DNA image/mask: camera {camera_id}, frame {source_id}")
                if mask.ndim == 3:
                    mask = mask.max(axis=2)
                height, width = image.shape[:2]
                if mask.shape != (height, width):
                    raise SystemExit(f"DNA image/mask dimensions differ: camera {camera_id}, frame {source_id}")
                if image_size is None:
                    image_size = (width, height)
                    scale = min(resolution / max(width, height), 1.0) if resolution else 1.0
                    size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    new_K = K.copy()
                    new_K[0, :] *= size[0] / width
                    new_K[1, :] *= size[1] / height
                    maps = cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_32FC1)
                elif image_size != (width, height):
                    raise SystemExit(f"DNA image dimensions changed in camera {camera_id}")
                image = cv2.remap(image, *maps, cv2.INTER_LINEAR)
                mask = cv2.remap(mask, *maps, cv2.INTER_NEAREST)
                mask = ((mask > 0) * 255).astype(np.uint8)
                image_path = image_dir / f"{local_id:08d}.jpg"
                mask_path = mask_dir / f"{local_id:08d}.png"
                if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Could not write {image_path}")
                if not cv2.imwrite(str(mask_path), mask):
                    raise OSError(f"Could not write {mask_path}")
                link = supplied_masks / f"mask_{local_id:04d}_01.png"
                if not link.is_symlink() and link.exists():
                    raise SystemExit(f"Unexpected supplied mask file: {link}")
                link.unlink(missing_ok=True)
                link.symlink_to(mask_path.resolve())
            output_calibration[name] = {"K": new_K.reshape(-1).tolist(),
                                        "R": extrinsics[:3, :3].reshape(-1).tolist(),
                                        "T": extrinsics[:3, 3].tolist(), "distCoeff": [0.0] * 5}
        (destination / "calibration.json").write_text(json.dumps(output_calibration, indent=2))
        (destination / "camera_id_mapping.json").write_text(json.dumps({f"cam{c:02d}": c for c in camera_ids}, indent=2))
        (destination / "frame_id_mapping.json").write_text(json.dumps({str(i): f for i, f in enumerate(frame_ids)}, indent=2))
        marker.write_text(json.dumps({"selection": signature, "complete": True}, indent=2))
    finally:
        for reader in readers:
            _close_reader(reader)
    return destination


def process_dna(args):
    dataset = DNADataset(args.dataset_root, required_keys=("images", "masks", "calibration"))
    output = args.output_root.expanduser().resolve()
    if paths_overlap(dataset.root, output):
        raise SystemExit("DNA input and output roots must not overlap")
    subjects = {subject["subject_id"]: subject for subject in dataset.subjects}
    missing = [s for s in args.subjects if s not in subjects or subjects[s]["missing_keys"]]
    if missing:
        raise SystemExit(f"DNA sequences missing RGB or annotations: {missing}")
    for subject_id in args.subjects:
        subject = subjects[subject_id]
        # RGB frame capacity does not depend on the supplied SMPL-X fits.
        import h5py
        with h5py.File(subject["images"], "r") as container:
            capacity = int(container["Camera_5mp"].attrs["num_frame"])
        stop = args.frame_stop if args.frame_stop is not None else capacity
        if stop > capacity or stop <= args.frame_start:
            raise SystemExit(f"Invalid frame interval for {subject_id}; RGB capacity is {capacity}")
        frames = list(range(args.frame_start, stop, args.frame_step))
        if not args.prepare_only and args.mamma_preset is None and len(frames) < 3:
            raise SystemExit("MAMMA's default temporal fitting preset needs at least three selected frames; use --prepare-only for shorter exports")
        destination = prepare_subject(subject, output / subject_id, args.cameras, frames, args.resolution)
        if not args.prepare_only:
            ensure_subject_smplx(destination, smplx_source="mamma",
                                 mamma_repo_path=args.mamma_repo_path, mamma_preset=args.mamma_preset,
                                 mamma_out_tag=args.mamma_out_tag, mamma_python=args.mamma_python,
                                 mamma_force=args.mamma_force, mamma_workers=args.mamma_workers,
                                 mamma_batch_size=getattr(args, "mamma_batch_size", 8),
                                 device=args.device)
            import numpy as np
            with np.load(destination / "smpl_params.npz") as params:
                for name, dimensions in (("global_orient", 3), ("transl", 3), ("body_pose", 63)):
                    if params[name].shape != (len(frames), dimensions) or not np.isfinite(params[name]).all():
                        raise SystemExit(f"Invalid fitted {name} for {subject_id}; expected {len(frames)} finite poses")
            convert_subject_smplx_to_mhr(destination, options_from_args(args))
            write_fit_metrics(destination, options_from_pipeline(args))
        print(f"DNA output: {destination}", flush=True)
