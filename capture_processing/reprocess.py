"""Reprocess an external sequence through the same stages as a VCI recording."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

from multiview_data import AVAILABLE_DATASETS, open_dataset
from capture_processing.cli import main as process
from capture_processing.utils.paths import paths_overlap
from capture_processing.utils.semantic_parsing import atomic_json
from capture_processing.utils.subject_lock import SUBJECT_BUSY_EXIT, SubjectBusyError, subject_lock


def prepare(sequence, stage, cameras, frames):
    """Stage lossless, undistorted image/mask pairs and record original IDs."""
    records = []
    for index, camera_id in enumerate(cameras):
        name = f"C{index:04d}"
        calibration = sequence.read_camera(camera_id)
        if calibration is None:
            raise ValueError(f"No calibration for {sequence.id}/{camera_id}")
        size = None
        maps = None
        for local_frame, source_frame in enumerate(frames):
            view = sequence.read_observation(camera_id, source_frame)
            if view.image is None or view.mask is None:
                continue
            image, mask = view.image, view.mask.astype(np.uint8) * 255
            dimensions = image.shape[1], image.shape[0]
            if size is None:
                size = dimensions
                if np.any(calibration.distortion):
                    import cv2

                    maps = cv2.initUndistortRectifyMap(
                        calibration.K, calibration.distortion, None, calibration.K, size, cv2.CV_32FC1
                    )
            if size != dimensions:
                raise ValueError(f"Image dimensions changed within camera {camera_id}")
            if maps is not None:
                import cv2

                image = cv2.remap(image, *maps, cv2.INTER_LINEAR)
                mask = cv2.remap(mask, *maps, cv2.INTER_NEAREST)
            for folder, stem, pixels in (("image", "color", image), ("mask", "mask", mask)):
                target = stage / f"frame_{local_frame:08d}" / folder / f"{stem}_{name}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(target)
        if size is None:
            raise ValueError(f"No selected image/mask pairs for camera {camera_id}")
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = calibration.R
        extrinsics[:3, 3] = calibration.T
        records.append({"camera_id": name, "intrinsics": {"camera_matrix": calibration.K.tolist(),
                        "resolution": list(size)}, "extrinsics": {"view_matrix": extrinsics.tolist()}})
    staged_frames = {int(path.name.removeprefix("frame_")) for path in stage.glob("frame_*")}
    if staged_frames != set(range(len(frames))):
        missing = [frame for index, frame in enumerate(frames) if index not in staged_frames]
        raise ValueError(f"No image/mask pairs in selected source frames: {missing}")
    provenance = {"dataset": sequence.kind, "subject_id": sequence.id, "frame_ids": frames,
                  "camera_ids": {record["camera_id"]: camera for record, camera in zip(records, cameras)}}
    atomic_json(stage / "calibration_dome.json", {"cameras": records, "meta": {"imported": provenance}})


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        index = argv.index("--")
        argv, options = argv[:index], argv[index + 1:]
    else:
        options = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(AVAILABLE_DATASETS))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--frames", nargs="+", type=int, required=True)
    parser.add_argument("--trusted-pickles", action="store_true")
    parser.epilog = "Pass normal VCI processing options after --."
    args = parser.parse_args(argv)
    if (not args.subject or Path(args.subject).name != args.subject or args.subject in (".", "..")
            or len(set(args.cameras)) != len(args.cameras)
            or any(frame < 0 for frame in args.frames) or sorted(set(args.frames)) != args.frames):
        parser.error("Use one subject ID, unique cameras and increasing nonnegative frame IDs")
    if paths_overlap(args.dataset.resolve(), args.output.resolve()):
        parser.error("External dataset and output roots must be disjoint")
    if any(flag.partition("=")[0] in {"--dataset-kind", "--calibration", "--skip-images", "--resume-validation", "--devices", "--gpu-workers"}
           for flag in options):
        parser.error("Reprocessing fixes the input layout and uses full media validation")
    sequences = open_dataset(args.kind, args.dataset, trusted_pickles=args.trusted_pickles)
    sequence = next((item for item in sequences if item.id == args.subject), None)
    if sequence is None:
        parser.error(f"Unknown subject: {args.subject}")
    try:
        output = args.output.expanduser().resolve() / args.subject
        # Prevent two different selections from writing the same subject concurrently.
        if output.is_symlink() or paths_overlap(sequence.root, output):
            parser.error("Output subject must be a separate directory")
        with subject_lock(output, "external reprocessing"), tempfile.TemporaryDirectory(prefix="capture-import-") as staging:
            with sequence:
                prepare(sequence, Path(staging) / args.subject, args.cameras, args.frames)
            process([staging, str(args.output), *options])
    except SubjectBusyError as exc:
        print(f"[busy] {exc}", file=sys.stderr)
        return SUBJECT_BUSY_EXIT
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
