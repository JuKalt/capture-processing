"""Shared SMPL-X parameter creation helpers."""

from __future__ import annotations

from capture_processing.utils.subject_lock import subject_locked

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from ..utils.mamma import (
    find_existing_mamma_params,
    mamma_steps_complete,
    prepare_mamma_runtime_preset,
    resolve_mamma_preset,
    resolve_mamma_repo,
    run_mamma,
)
from ..utils.smplx_utils import save_mamma_smpl_params
from ..utils.fit_metrics import prepare_smplx_fit_diagnostics


@subject_locked
def ensure_subject_smplx(
    subject_root: Path,
    *,
    smplx_source: str,
    mamma_repo_path: Path | None,
    mamma_preset: Path | None,
    mamma_out_tag: str,
    mamma_python: str,
    mamma_force: bool,
    mamma_workers: int,
    mamma_batch_size: int = 8,
    mamma_confidence_fit: bool = False,
    device: str | None,
) -> None:
    params_path = subject_root / "smpl_params.npz"
    if smplx_source == "existing":
        if params_path.is_file() and not mamma_force:
            print(f"Using existing SMPL-X parameters: {params_path}")
            return
        raise SystemExit(
            f"No existing smpl_params.npz found for {subject_root.name}; "
            "use --smplx-source mamma."
        )
    if params_path.is_file() and not mamma_force:
        print(f"Using existing SMPL-X parameters: {params_path}")
        existing = find_existing_mamma_params(subject_root / "mamma_output", mamma_out_tag, subject_root, "images")
        if existing is not None:
            prepare_smplx_fit_diagnostics(subject_root, existing, mamma_python)
        from ..utils.mamma_additional_fit import ensure_confidence_fit
        ensure_confidence_fit(subject_root, SimpleNamespace(
            smplx_source=smplx_source, mamma_repo_path=mamma_repo_path,
            mamma_out_tag=mamma_out_tag, mamma_python=mamma_python,
            mamma_force=mamma_force, mamma_confidence_fit=mamma_confidence_fit, device=device))
        return

    args = SimpleNamespace(
        smplx_source="mamma",
        mamma_repo_path=mamma_repo_path,
        mamma_preset=mamma_preset,
        mamma_out_tag=mamma_out_tag,
        mamma_python=mamma_python,
        mamma_force=mamma_force,
        mamma_workers=mamma_workers,
        mamma_batch_size=mamma_batch_size,
        mamma_confidence_fit=mamma_confidence_fit,
        device=device,
    )
    create_smpl_params(args, subject_root, [])
    from ..utils.mamma_additional_fit import ensure_confidence_fit
    ensure_confidence_fit(subject_root, args)


@subject_locked(argument=1)
def create_smpl_params(
    args: argparse.Namespace,
    out_root: Path,
    frames: list[tuple[int, Path]],
) -> None:
    source = args.smplx_source
    if source == "mamma":
        repo = resolve_mamma_repo(args.mamma_repo_path)
        create_smpl_params_with_mamma(args, out_root, repo)
        return

    if source == "existing":
        raise SystemExit(
            "--smplx-source existing is only supported for copied THuman subjects "
            "that already contain smpl_params.npz. For VCI conversion, use "
            "--smplx-source mamma."
        )

    raise SystemExit(f"Unsupported SMPL-X source: {source}")


def create_smpl_params_with_mamma(
    args: argparse.Namespace,
    out_root: Path,
    repo: Path,
) -> None:
    images_root = out_root / "images"
    if not images_root.is_dir():
        raise SystemExit(
            f"MAMMA needs transcoded images at {images_root}. "
            "Run without --skip-images first."
        )

    preset = resolve_mamma_preset(repo, args.mamma_preset)
    supplied_masks = out_root / "mamma_input_masks"
    preset = prepare_mamma_runtime_preset(
        preset,
        out_root,
        getattr(args, "mamma_workers", 4),
        masks_root=supplied_masks if supplied_masks.is_dir() else None,
        batch_size=getattr(args, "mamma_batch_size", 8),
    )
    mamma_calibration = out_root / "mamma_calibration.json"
    write_mamma_calibration_from_thuman(
        out_root / "calibration.json", mamma_calibration
    )

    output_root = out_root / "mamma_output"
    mamma_footage_root = out_root
    mamma_seq_name = images_root.name
    params_path = find_existing_mamma_params(
        output_root, args.mamma_out_tag, mamma_footage_root, mamma_seq_name
    )

    if params_path is None or args.mamma_force or not mamma_steps_complete(
        output_root, args.mamma_out_tag, mamma_footage_root.name, mamma_seq_name, preset
    ):
        print("Running MAMMA SMPL-X estimation from multi-view images...")
        params_path = run_mamma(
            repo=repo,
            preset=preset,
            footage_root=mamma_footage_root,
            seq_name=mamma_seq_name,
            calibration_path=mamma_calibration,
            output_root=output_root,
            tag=args.mamma_out_tag,
            python_executable=args.mamma_python,
            force=args.mamma_force,
            device=args.device,
        )
    else:
        print(f"Using existing MAMMA SMPL-X output: {params_path}")

    save_mamma_smpl_params(out_root, params_path)
    prepare_smplx_fit_diagnostics(out_root, params_path, args.mamma_python)
    print(f"  wrote smpl_params.npz from MAMMA ({params_path.name})")


def write_mamma_calibration_from_thuman(src_path: Path, out_path: Path) -> None:
    """Convert the already-written THUman calibration to MAMMA JSON."""
    import numpy as np
    from PIL import Image

    thuman_cal = json.loads(src_path.read_text())
    out: dict[str, dict] = {}
    for cam_id, camera in thuman_cal.items():
        image_dir = src_path.parent / "images" / cam_id
        first_image = next(
            (
                path
                for path in sorted(image_dir.iterdir())
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ),
            None,
        )
        if first_image is None:
            raise SystemExit(f"No images found for MAMMA camera {cam_id}: {image_dir}")
        with Image.open(first_image) as image:
            width, height = image.size

        K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
        R = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
        T = np.asarray(camera["T"], dtype=np.float64).reshape(3)
        out[cam_id] = {
            "intrinsic_matrix": K.tolist(),
            "extrinsics_matrix": np.hstack([R, T.reshape(3, 1)]).tolist(),
            "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [width, height],
        }
    out_path.write_text(json.dumps(out, indent=2))
