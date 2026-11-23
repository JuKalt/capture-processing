"""DanceNet3D extraction, canonicalization, and SMPL-X generation."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from capture_processing.core.smplx_pipeline import ensure_subject_smplx
from capture_processing.utils.fit_metrics import write_fit_metrics
from capture_processing.utils.silhouette_metrics import options_from_pipeline
from capture_processing.utils.smplx_mhr import convert_subject_smplx_to_mhr, options_from_args


def _data_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"COLMAP file not found: {path}")
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_colmap_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in _data_lines(path):
        values = line.split()
        if len(values) < 8:
            raise SystemExit(f"Malformed COLMAP camera line in {path}: {line}")
        camera_id = int(values[0])
        model = values[1]
        width, height = int(values[2]), int(values[3])
        params = [float(value) for value in values[4:]]
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            focal, cx, cy = params[:3]
            fx = fy = focal
        elif model in {
            "PINHOLE",
            "OPENCV",
            "OPENCV_FISHEYE",
            "FULL_OPENCV",
        }:
            fx, fy, cx, cy = params[:4]
        else:
            raise SystemExit(
                f"Unsupported DanceNet3D COLMAP camera model {model!r} in {path}"
            )
        cameras[camera_id] = {
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "width": width,
            "height": height,
        }
    return cameras


def _quaternion_matrix(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise SystemExit("COLMAP camera quaternion has zero length")
    w, x, y, z = (value / norm for value in (qw, qx, qy, qz))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def read_colmap_images(
    path: Path, source_camera_ids: list[str]
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"COLMAP file not found: {path}")
    known = set(source_camera_ids)
    images: dict[str, dict[str, Any]] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 10:
            # The second line of a COLMAP image record contains optional
            # POINTS2D triples and is not a camera-pose record.
            continue
        stem = Path(values[9]).stem
        tokens = stem.replace("-", "_").split("_")
        candidates = [token for token in tokens if token in known]
        if stem in known:
            source_camera_id = stem
        elif candidates:
            source_camera_id = candidates[-1]
        else:
            suffix_matches = [
                camera_id
                for camera_id in source_camera_ids
                if stem.endswith(camera_id)
            ]
            if len(suffix_matches) != 1:
                continue
            source_camera_id = suffix_matches[0]
        try:
            qw, qx, qy, qz = (float(value) for value in values[1:5])
            translation = [float(value) for value in values[5:8]]
            colmap_camera_id = int(values[8])
        except ValueError as exc:
            raise SystemExit(f"Malformed COLMAP image line in {path}: {line}") from exc
        images[source_camera_id] = {
            "camera_id": colmap_camera_id,
            "R": _quaternion_matrix(qw, qx, qy, qz),
            "T": translation,
        }
    return images


def _link_file(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise SystemExit(f"Existing symlink points at different data: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def _clear_managed_links(root: Path) -> None:
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if path.is_symlink():
            path.unlink()
    for path in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        try:
            path.rmdir()
        except OSError:
            pass


def prepare_dancenet3d_subject(
    *,
    dataset_root: Path,
    extracted_session_root: Path,
    output_root: Path,
    session: str,
    sequence: str,
    sequence_data: dict[str, Any],
    max_frames: int | None = None,
    selected_cameras: list[str] | None = None,
) -> Path:
    """Build a canonical, zero-copy subject from extracted DanceNet3D frames."""
    source_sequence = dataset_root / session / sequence
    extracted = extracted_session_root / sequence / "images_and_masks"
    if not extracted.is_dir():
        raise SystemExit(
            f"Extracted DanceNet3D sequence not found: {extracted}\n"
            "Run without --skip-extraction first."
        )

    cameras = list(sequence_data["cameras"])
    if selected_cameras:
        selected = set(selected_cameras)
        cameras = [camera for camera in cameras if camera in selected]
    if not cameras:
        raise SystemExit(f"No selected cameras are available for {session}/{sequence}")

    colmap = source_sequence / "colmap"
    camera_models = read_colmap_cameras(colmap / "cameras.txt")
    poses = read_colmap_images(colmap / "images.txt", cameras)
    missing_calibration = [camera for camera in cameras if camera not in poses]
    if missing_calibration:
        raise SystemExit(
            f"COLMAP calibration is missing DanceNet cameras: {missing_calibration}"
        )

    subject_id = f"{session}_{sequence}"
    subject_root = output_root / subject_id
    subject_root.mkdir(parents=True, exist_ok=True)
    _clear_managed_links(subject_root / "images")
    _clear_managed_links(subject_root / "masks")
    mapping = {
        f"cam{index:02d}": source_camera
        for index, source_camera in enumerate(sorted(cameras))
    }

    source_frame_ids = list(sequence_data["frame_ids"])
    if max_frames is not None:
        if max_frames <= 0:
            raise SystemExit("--max-frames must be positive")
        source_frame_ids = source_frame_ids[:max_frames]
    extension = sequence_data.get("image_extension", ".png")
    content_type = sequence_data.get("content_type", "images_no_lut")
    missing_media = [
        f"{frame_id}/{camera}"
        for frame_id in source_frame_ids
        for camera in cameras
        if not (extracted / frame_id / content_type / f"{camera}{extension}").is_file()
        or not (extracted / frame_id / "masks" / f"{camera}.png").is_file()
    ]
    # MAMMA emits one SMPL-X row per usable input frame. Give those rows the
    # same dense, zero-based IDs in the canonical image layout and retain the
    # native DanceNet frame names in a sidecar.
    usable_source_frames = [
        frame_id
        for frame_id in source_frame_ids
        if all(
            (extracted / frame_id / content_type / f"{camera}{extension}").is_file()
            and (extracted / frame_id / "masks" / f"{camera}.png").is_file()
            for camera in cameras
        )
    ]
    if not usable_source_frames:
        raise SystemExit(f"No complete multiview frames found for {session}/{sequence}")
    frame_mapping = {
        f"{index:08d}": source_frame
        for index, source_frame in enumerate(usable_source_frames)
    }
    linked_by_camera = {camera: 0 for camera in mapping}

    for canonical_camera, source_camera in mapping.items():
        for canonical_frame, source_frame in frame_mapping.items():
            image = extracted / source_frame / content_type / f"{source_camera}{extension}"
            mask = extracted / source_frame / "masks" / f"{source_camera}.png"
            _link_file(
                image,
                subject_root / "images" / canonical_camera / f"{canonical_frame}{extension}",
            )
            _link_file(
                mask,
                subject_root / "masks" / canonical_camera / f"{canonical_frame}.png",
            )
            linked_by_camera[canonical_camera] += 1

    empty_cameras = [
        mapping[camera]
        for camera, linked_count in linked_by_camera.items()
        if linked_count == 0
    ]
    if empty_cameras:
        raise SystemExit(
            "No paired extracted images and masks were found for cameras: "
            + ", ".join(empty_cameras)
        )

    # MAMMA ma_2d accepts existing masks at
    # <mask-root>/<sequence>/<camera>/masks/mask_<index>_01.png. DanceNet3D
    # already provides reviewed single-person masks, so expose them directly
    # and let the runtime preset skip the expensive SAM ma_masks step.
    mamma_masks = subject_root / "mamma_input_masks"
    _clear_managed_links(mamma_masks)
    for canonical_camera in mapping:
        camera_masks = sorted((subject_root / "masks" / canonical_camera).glob("*.png"))
        for frame_index, mask in enumerate(camera_masks):
            _link_file(
                mask,
                mamma_masks
                / "images"
                / canonical_camera
                / "masks"
                / f"mask_{frame_index:04d}_01.png",
            )

    calibration: dict[str, dict[str, Any]] = {}
    for canonical_camera, source_camera in mapping.items():
        pose = poses[source_camera]
        try:
            model = camera_models[pose["camera_id"]]
        except KeyError as exc:
            raise SystemExit(
                f"COLMAP camera {pose['camera_id']} referenced by {source_camera} "
                "is absent from cameras.txt"
            ) from exc
        calibration[canonical_camera] = {
            "K": [value for row in model["K"] for value in row],
            "R": [value for row in pose["R"] for value in row],
            "T": pose["T"],
        }

    (subject_root / "calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n"
    )
    (subject_root / "camera_id_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n"
    )
    (subject_root / "frame_id_mapping.json").write_text(
        json.dumps(frame_mapping, indent=2) + "\n"
    )
    (subject_root / "missing_img_files.txt").write_text(
        "\n".join(missing_media) + ("\n" if missing_media else "")
    )
    return subject_root


def incomplete_extraction_cameras(
    *,
    extracted_session_root: Path,
    sequence: str,
    sequence_data: dict[str, Any],
    selected_cameras: list[str] | None,
) -> tuple[list[str], bool]:
    """Return cameras needing video decoding and whether all masks exist."""
    cameras = list(sequence_data["cameras"])
    if selected_cameras:
        selected = set(selected_cameras)
        cameras = [camera for camera in cameras if camera in selected]
    if not cameras:
        raise SystemExit(f"No selected cameras are available for {sequence}")

    extracted = extracted_session_root / sequence / "images_and_masks"
    frame_ids = list(sequence_data["frame_ids"])
    extension = sequence_data.get("image_extension", ".png")
    content_type = sequence_data.get("content_type", "images_no_lut")
    incomplete_cameras = []
    for camera in cameras:
        if any(
            not (extracted / frame_id / content_type / f"{camera}{extension}").is_file()
            for frame_id in frame_ids
        ):
            incomplete_cameras.append(camera)

    masks_complete = not sequence_data.get("has_masks", False) or all(
        (extracted / frame_id / "masks" / f"{camera}.png").is_file()
        for frame_id in frame_ids
        for camera in cameras
    )
    return incomplete_cameras, masks_complete


def extract_dancenet3d_session(
    *,
    dataset_root: Path,
    extracted_session_root: Path,
    session: str,
    sequences: list[str],
    cameras: list[str] | None,
    parallel: int,
) -> None:
    extractor = dataset_root / "video_to_images.py"
    if not extractor.is_file():
        raise SystemExit(f"DanceNet3D extractor not found: {extractor}")
    if not (dataset_root / session / "manifest.json").is_file():
        raise SystemExit(f"DanceNet3D session not downloaded: {dataset_root / session}")
    missing = []
    if importlib.util.find_spec("numpy") is None:
        missing.append(
            f"numpy (install with `{sys.executable} -m pip install numpy`)"
        )
    if shutil.which("ffmpeg") is None:
        missing.append(
            "ffmpeg (install with `conda install -c conda-forge ffmpeg` "
            "or your system package manager)"
        )
    if missing:
        raise SystemExit(
            "DanceNet3D extraction dependencies are missing:\n  - "
            + "\n  - ".join(missing)
        )
    cmd = [
        sys.executable,
        str(extractor),
        "--input",
        str(dataset_root / session),
        "--output",
        str(extracted_session_root),
        "--parallel",
        str(parallel),
        "--sequences",
        *sequences,
    ]
    if cameras:
        cmd.extend(["--cameras", *cameras])

    manifest = json.loads((dataset_root / session / "manifest.json").read_text())
    available = manifest.get("sequences", {})
    content_type = manifest.get("content_type", "images_no_lut")
    expected_images = 0
    for sequence in sequences:
        sequence_data = available.get(sequence, {})
        sequence_cameras = cameras or sequence_data.get("cameras", [])
        expected_images += len(sequence_data.get("frame_ids", [])) * len(
            sequence_cameras
        )

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(cmd, env=env)
    started = time.monotonic()
    print(
        f"Monitoring DanceNet3D extraction for {session}; "
        "progress updates every 10 seconds.",
        flush=True,
    )
    while True:
        try:
            return_code = process.wait(timeout=10)
            break
        except subprocess.TimeoutExpired:
            decoded_images = 0
            for sequence in sequences:
                image_root = (
                    extracted_session_root / sequence / "images_and_masks"
                )
                for image in image_root.glob(f"*/{content_type}/*"):
                    if cameras is None or image.stem in cameras:
                        decoded_images += 1
            elapsed = int(time.monotonic() - started)
            progress = (
                f"{100 * decoded_images / expected_images:.1f}%"
                if expected_images
                else "unknown"
            )
            print(
                f"[{session}] extraction: {decoded_images:,}/"
                f"{expected_images:,} images ({progress}), "
                f"elapsed {elapsed // 60}m {elapsed % 60:02d}s",
                flush=True,
            )
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def process_dancenet3d(
    args,
    *,
    smplx_runner=ensure_subject_smplx,
) -> list[Path]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = dataset_root / args.session / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"DanceNet3D manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    available = manifest.get("sequences", {})
    sequences = list(args.sequences or available)
    unknown = [sequence for sequence in sequences if sequence not in available]
    if unknown:
        raise SystemExit(
            f"Unknown DanceNet3D sequences for {args.session}: {', '.join(unknown)}"
        )
    extracted_root = (
        args.extracted_root.expanduser().resolve()
        if args.extracted_root
        else dataset_root / "extracted"
    )
    extracted_session_root = extracted_root / args.session
    output_root.mkdir(parents=True, exist_ok=True)
    subjects: list[Path] = []
    for sequence in sequences:
        sequence_data = dict(available[sequence])
        sequence_data.setdefault(
            "content_type", manifest.get("content_type", "images_no_lut")
        )
        if not args.skip_extraction:
            incomplete_cameras, masks_complete = incomplete_extraction_cameras(
                extracted_session_root=extracted_session_root,
                sequence=sequence,
                sequence_data=sequence_data,
                selected_cameras=args.cameras,
            )
            if not incomplete_cameras and masks_complete:
                print(
                    f"Reusing complete DanceNet3D extraction for "
                    f"{args.session}/{sequence}."
                )
            else:
                repair_cameras = incomplete_cameras
                if not repair_cameras:
                    available_cameras = list(sequence_data["cameras"])
                    if args.cameras:
                        selected = set(args.cameras)
                        available_cameras = [
                            camera
                            for camera in available_cameras
                            if camera in selected
                        ]
                    # The official extractor always decodes at least one video
                    # before unpacking the shared mask archive.
                    repair_cameras = available_cameras[:1]
                reason = (
                    f"{len(incomplete_cameras)} incomplete camera(s)"
                    if incomplete_cameras
                    else "incomplete masks"
                )
                print(
                    f"Extracting DanceNet3D {args.session}/{sequence}: {reason}; "
                    "existing complete cameras will be reused."
                )
                extract_dancenet3d_session(
                    dataset_root=dataset_root,
                    extracted_session_root=extracted_session_root,
                    session=args.session,
                    sequences=[sequence],
                    cameras=repair_cameras,
                    parallel=args.parallel,
                )
        print(f"Preparing DanceNet3D {args.session}/{sequence}...")
        subject = prepare_dancenet3d_subject(
            dataset_root=dataset_root,
            extracted_session_root=extracted_session_root,
            output_root=output_root,
            session=args.session,
            sequence=sequence,
            sequence_data=sequence_data,
            max_frames=args.max_frames,
            selected_cameras=args.cameras,
        )
        subjects.append(subject)
        if args.skip_smplx:
            continue
        print(f"Generating SMPL-X for {subject.name} with MAMMA...")
        smplx_runner(
            subject,
            smplx_source="mamma",
            mamma_repo_path=args.mamma_repo_path,
            mamma_preset=args.mamma_preset,
            mamma_out_tag=args.mamma_out_tag,
            mamma_python=args.mamma_python,
            mamma_force=args.mamma_force,
            mamma_workers=args.mamma_workers,
            mamma_batch_size=getattr(args, "mamma_batch_size", 8),
            device=args.device,
        )
        convert_subject_smplx_to_mhr(subject, options_from_args(args))
        write_fit_metrics(subject, options_from_pipeline(args))
    return subjects
