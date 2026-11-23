"""VCI dataset creation pipeline."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .processing_log import (
    CONVERSION_STEPS,
    load_processing_log,
    save_processing_log,
    mark_step_complete,
    get_missing_conversion_steps,
)
from ..utils.camera import (
    load_source_calibration,
    build_camera_id_mapping,
    read_camera_list_arg,
    resolve_drop_tokens,
    apply_camera_filter,
    write_thuman4_calibration,
    build_resize_sizes,
    thuman4_calibration,
)
from ..utils.frames import discover_frames
from ..utils.mhr import maybe_create_mhr_params
from ..utils.fit_metrics import write_fit_metrics, validate_fit_metrics
from ..utils.silhouette_metrics import options_from_pipeline
from ..utils.smplx_mhr import convert_subject_smplx_to_mhr
from ..utils.paths import paths_overlap
from .smplx_pipeline import create_smpl_params
from .vci_resume import (
    media_signature, valid_media_pair, valid_json, valid_smplx,
    quick_conversion_complete, quick_media_complete, quick_settings,
    media_pair_signature, read_media_receipt, record_media_pair, fitted_media_signature,
)
from ..utils.transcode import transcode_pair, build_transcoding_jobs

MAX_TRANSCODE_WORKERS = 64


def run_dataset_creation(args: argparse.Namespace) -> Path:
    from ..utils.subject_lock import subject_lock
    source = args.input_path.expanduser().resolve()
    output = args.out_dir.expanduser().resolve() / source.name
    if paths_overlap(source, output):
        raise SystemExit(
            "Output subject directory must not overlap the input object directory.\n"
            f"  input object: {source}\n"
            f"  output subject: {output}\n"
            "Choose a separate output parent, for example: "
            f"{source.parent / 'processed'}"
        )
    with subject_lock(output, 'VCI preprocessing'):
        from ..utils.subject_manifest import write_subject_manifest
        try:
            result = _run_dataset_creation(args)
        except (Exception, SystemExit, KeyboardInterrupt) as exc:
            try:
                write_subject_manifest(output, args, result='failed', error=str(exc))
            except Exception as manifest_error:
                import sys
                print(f'Could not record failed-run manifest: {manifest_error}', file=sys.stderr)
            raise
        write_subject_manifest(output, args)
        return result


def _run_dataset_creation(args: argparse.Namespace) -> Path:
    """Run the full dataset conversion pipeline and return the output path."""
    from ..utils.preprocessing_validation import validate_options
    validate_options(args)
    anchor = args.input_path.expanduser().resolve()
    if not anchor.is_dir():
        raise SystemExit(f"Not a directory: {anchor}")

    # run_dataset_creation has already rejected overlapping input/output paths.
    out_root = args.out_dir.expanduser().resolve() / anchor.name
    out_root.mkdir(parents=True, exist_ok=True)

    log = load_processing_log(out_root)
    from ..utils.preprocessing_provenance import baseline_dependencies
    dependency_signature, dependencies = baseline_dependencies(args, out_root)
    previous_dependencies = log.get('smplx_dependency_signature')
    if (previous_dependencies is not None and previous_dependencies != dependency_signature):
        log['smplx_force_pending'] = True
    elif previous_dependencies is None and (getattr(args, 'mamma_preset', None) is not None
                                             or getattr(args, 'mamma_repo_path', None) is not None):
        # Unknown legacy provenance cannot establish a match for custom fitting inputs.
        log['smplx_force_pending'] = True
    missing_steps = get_missing_conversion_steps(log)

    print(f"Output subject dir: {out_root}")
    print(f"Conversion status: {missing_steps[0] if missing_steps else 'complete; checking postprocessing'}")

    if quick_conversion_complete(args, out_root, log):
        print("Quick resume: reusing completed conversion without scanning media.", flush=True)
        _create_mhr_params_from_args(args, out_root)
        print(f"Done: {out_root}", flush=True)
        return out_root

    # Load calibration (only if not already done)
    if "calibration_loaded" in missing_steps:
        src_cams = load_source_calibration(anchor / args.calibration)
        print(f"Loaded {len(src_cams)} cameras from source calibration.")
        mark_step_complete(log, "calibration_loaded")
        save_processing_log(out_root, log)
    else:
        src_cams = load_source_calibration(anchor / args.calibration)

    full_id_map = build_camera_id_mapping(list(src_cams.keys()))
    drop_tokens = read_camera_list_arg(args.drop_cameras)
    drop_set = (
        resolve_drop_tokens(drop_tokens, set(src_cams.keys())) if drop_tokens else set()
    )
    id_map = apply_camera_filter(full_id_map, drop_set) if drop_set else full_id_map

    if "cameras_mapped" in missing_steps:
        if drop_set:
            print(
                f"Camera filter: {len(id_map)} kept, {len(full_id_map) - len(id_map)} excluded."
            )
            print(f"  dropped source IDs: {sorted(drop_set)}")

        (out_root / "camera_id_mapping.json").write_text(json.dumps(id_map, indent=2))
        mark_step_complete(log, "cameras_mapped")
        save_processing_log(out_root, log)
    elif drop_set:
        print(
            f"Camera filter: {len(id_map)} kept, {len(full_id_map) - len(id_map)} excluded."
        )
        print(f"  dropped source IDs: {sorted(drop_set)}")

    frames = discover_frames(anchor)
    if not frames:
        raise SystemExit(f"No frame_* dirs found in {anchor}")

    frame_ids = [fid for fid, _ in frames]
    frame_id_set = set(frame_ids)
    n_frames_total = max(frame_ids) + 1
    missing_frames = [fid for fid in range(n_frames_total) if fid not in frame_id_set]

    if "frames_discovered" in missing_steps:
        print(
            f"Found {len(frames)} frames; sequence length = {n_frames_total} "
            f"(missing: {len(missing_frames)})"
        )
        mark_step_complete(log, "frames_discovered")
        save_processing_log(out_root, log)

    dst_size = build_resize_sizes(id_map, src_cams, args.resize_long_edge)
    image_missing: list[str] = []

    transcoding_jobs, image_missing = build_transcoding_jobs(
        frames, id_map, out_root, args.resize_long_edge, args.jpeg_quality
    )
    from ..utils.preprocessing_validation import validate_source_media
    validate_source_media(transcoding_jobs, src_cams, id_map)
    signature = media_signature(transcoding_jobs, [src_cams, id_map])
    previous = log.get("media_signature")
    changed = previous is not None and previous != signature
    reuse_media = not changed and quick_media_complete(args, out_root, log)
    print("Quick resume: trusting completed media stage." if reuse_media
          else f"Validating {len(transcoding_jobs)} image/mask pairs...", flush=True)
    pair_signatures = {job[1]: media_pair_signature(job) for job in transcoding_jobs}
    pending_jobs = []
    if not reuse_media:
        for job in transcoding_jobs:
            receipt = read_media_receipt(out_root, job)
            stale = (receipt.get('signature') != pair_signatures[job[1]] if receipt
                     else changed)
            if stale or not valid_media_pair(job, check_binary_mask=not receipt):
                pending_jobs.append(job)
            else:
                record_media_pair(out_root, job, pair_signatures[job[1]])
    # Establish legacy fit provenance before any media replacement. If interrupted,
    # retries still compare against the actual inputs used by the completed fit.
    if ('smplx_media_signature' not in log
            and (out_root / 'smpl_params.npz').is_file()):
        log['smplx_media_signature'] = fitted_media_signature(out_root)
    if ('smplx_media_signature' not in log and (changed or not (out_root / 'smpl_params.npz').is_file())
            and ('smplx_created' in log['steps_completed'] or (out_root / 'mamma_output').exists())):
        log['smplx_force_pending'] = True
    if changed and args.skip_images:
        raise SystemExit("Media inputs/settings changed; rerun without --skip-images.")
    if previous is None:
        log["media_signature"] = signature
    if pending_jobs or "images_transcoded" in missing_steps or changed:
        log["steps_completed"].pop("images_transcoded", None)
    if pending_jobs or changed:
        # Prevent a quick skip if interrupted between media replacement and fit
        # invalidation/cleanup. Individual successful pairs have their own receipts.
        log['media_update_pending'] = True
    if not valid_json(out_root / "camera_id_mapping.json", id_map):
        (out_root / "camera_id_mapping.json").write_text(json.dumps(id_map, indent=2))
    calibration = thuman4_calibration(id_map, src_cams, dst_size)
    if not valid_json(out_root / "calibration.json", calibration):
        log["steps_completed"].pop("calibration_written", None)
    args.mamma_force = args.mamma_force or log.get("smplx_force_pending", False)
    if not valid_smplx(out_root / "smpl_params.npz", n_frames_total) or args.mamma_force:
        log["steps_completed"].pop("smplx_created", None)
    metadata = out_root / "missing_img_files.txt"
    expected_missing = "\n".join(image_missing) + ("\n" if image_missing else "")
    if not metadata.is_file() or metadata.read_text() != expected_missing:
        log["steps_completed"].pop("metadata_written", None)
    missing_steps = get_missing_conversion_steps(log)
    # Retain the old signature until replacement media finishes; an interrupted
    # settings change must not make untouched old pairs eligible for reuse.
    save_processing_log(out_root, log)

    if not args.skip_images and "images_transcoded" in missing_steps:
        if not transcoding_jobs:
            raise SystemExit(
                "No image/mask pairs were found to transcode. MAMMA needs "
                f"{out_root / 'images'} to contain per-camera JPEG frames.\n"
                "Expected each source frame to contain files like:\n"
                "  frame_000000/image/color_<SOURCE_CAMERA_ID>.avif\n"
                "  frame_000000/mask/mask_<SOURCE_CAMERA_ID>.avif\n"
                "  frame_000000/person_masks/mask_<SOURCE_CAMERA_ID>.png\n"
                f"Missing output entries recorded: {len(image_missing)}"
            )

        worker_count = _resolve_worker_count(args.workers)
        print(
            f"Transcoding {len(pending_jobs)} image+mask pairs "
            f"with {worker_count} worker(s)..."
        )
        n_fail = 0
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(transcode_pair, job): job for job in pending_jobs}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Transcode"
            ):
                _, err = future.result()
                if err:
                    n_fail += 1
                else:
                    job = futures[future]
                    record_media_pair(out_root, job, pair_signatures[job[1]])
        if n_fail:
            raise SystemExit(f"{n_fail} image/mask pairs failed to transcode; stage remains incomplete.")
        log["media_signature"] = signature
        mark_step_complete(log, "images_transcoded")
        save_processing_log(out_root, log)

    if not args.skip_images and "images_transcoded" not in get_missing_conversion_steps(log):
        expected_media = {path for job in transcoding_jobs for path in (job[1], job[3])}
        for directory in (out_root / "images", out_root / "masks"):
            for path in directory.rglob("*.jpg"):
                if path not in expected_media:
                    path.unlink()
            for camera in directory.iterdir():
                if camera.is_dir() and not any(camera.iterdir()):
                    camera.rmdir()

        # Receipts describe the same pair inventory as the exported media.
        for receipt in (out_root / '.media_state').rglob('*.json'):
            image = out_root / 'images' / receipt.relative_to(out_root / '.media_state').with_suffix('.jpg')
            if image not in expected_media:
                receipt.unlink()

    if "images_transcoded" not in get_missing_conversion_steps(log) and not pending_jobs:
        log["media_signature"] = signature
        save_processing_log(out_root, log)

    if "calibration_written" in missing_steps:
        write_thuman4_calibration(
            out_root / "calibration.json", id_map, src_cams, dst_size
        )
        print(f"  wrote calibration.json ({len(id_map)} cameras)")
        mark_step_complete(log, "calibration_written")
        save_processing_log(out_root, log)

    # Fit invalidation follows converted outputs, not a global source timestamp.
    # Identical re-encodes preserve output mtimes and all dependent stage caches.
    if not args.skip_images:
        for job in transcoding_jobs:
            record_media_pair(out_root, job, pair_signatures[job[1]])
    current_fit_signature = fitted_media_signature(out_root)
    old_fit_signature = log.get('smplx_media_signature')
    if old_fit_signature is not None and old_fit_signature != current_fit_signature:
        log['steps_completed'].pop('smplx_created', None)
        log['smplx_force_pending'] = True
        args.mamma_force = True
    if not args.skip_images:
        log.pop('media_update_pending', None)
    save_processing_log(out_root, log)
    missing_steps = get_missing_conversion_steps(log)

    if "smplx_created" in missing_steps:
        create_smpl_params(args, out_root, frames)
        if not valid_smplx(out_root / "smpl_params.npz", n_frames_total):
            raise SystemExit("SMPL-X estimation did not write complete valid parameters.")
        log.pop("smplx_force_pending", None)
        log["smplx_media_signature"] = current_fit_signature
        mark_step_complete(log, "smplx_created")
        save_processing_log(out_root, log)

    if "metadata_written" in missing_steps:
        (out_root / "missing_img_files.txt").write_text(
            "\n".join(image_missing) + ("\n" if image_missing else "")
        )
        print(f"  wrote missing_img_files.txt ({len(image_missing)} entries)")
        mark_step_complete(log, "metadata_written")
        save_processing_log(out_root, log)

    if 'smplx_created' in log['steps_completed']:
        log['smplx_dependency_signature'] = dependency_signature
        log['smplx_dependencies'] = dependencies
    if not get_missing_conversion_steps(log):
        log["quick_resume_settings"] = quick_settings(args)
        save_processing_log(out_root, log)
    _create_mhr_params_from_args(args, out_root)

    print("\nDone.")
    print(f"  THUman4-format subject at: {out_root}")
    print(f"  Frames: {n_frames_total}")
    first_cam = next(iter(id_map))
    print(f"  Cameras: {len(id_map)}  (e.g. {first_cam} -> {id_map[first_cam]})")
    return out_root


def _create_mhr_params_from_args(args: argparse.Namespace, out_root: Path) -> None:
    from ..utils.mamma_additional_fit import ensure_confidence_fit
    from ..utils.subject_validation import requested_fits
    from ..utils.semantic_parsing import options_from_args
    fits = requested_fits(args)
    log = load_processing_log(out_root)
    log['requested_steps'] = list(CONVERSION_STEPS)
    log['requested_steps'] += [kind for kind in fits if kind != 'smplx'] + ['fit_metrics']
    if options_from_args(args).enabled:
        log['requested_steps'].append('semantic_parsing')
    save_processing_log(out_root, log)

    def stage(name, action):
        state = load_processing_log(out_root)
        state['steps_completed'].pop(name, None)
        save_processing_log(out_root, state)
        result = action()
        state = load_processing_log(out_root)
        mark_step_complete(state, name)
        save_processing_log(out_root, state)
        return result

    if 'smplx_confidence' in fits:
        stage('smplx_confidence', lambda: ensure_confidence_fit(out_root, args))
    if 'mhrfromsmplx' in fits:
        stage('mhrfromsmplx', lambda: convert_subject_smplx_to_mhr(
            out_root, getattr(args, 'smplx_mhr_options', None)))
    def mhr():
        maybe_create_mhr_params(
            out_root, source=getattr(args, 'mhr_source', 'none'),
            repo_path=args.mhr_repo_path, python_executable=args.mhr_python,
            device=args.device, max_frames=args.mhr_max_frames,
            iterations=args.mhr_iterations, force=args.mhr_force,
            workers=getattr(args, 'mhr_workers', 1))
    if 'mhr' in fits:
        stage('mhr', mhr)
    state = load_processing_log(out_root)
    state['steps_completed'].pop('fit_metrics', None)
    save_processing_log(out_root, state)

    report = write_fit_metrics(out_root, options_from_pipeline(args))
    from ..utils.mamma_additional_fit import requested, validate_confidence_metrics
    required = requested_fits(args)
    validate_fit_metrics(report, required)
    if requested(args):
        validate_confidence_metrics(out_root, report)
    state = load_processing_log(out_root)
    mark_step_complete(state, 'fit_metrics')
    save_processing_log(out_root, state)
    from ..utils.semantic_parsing import ensure_semantic_parsing
    ensure_semantic_parsing(out_root, args)

def _resolve_worker_count(requested: int) -> int:
    """Resolve CLI worker count (validated >= 0); 0 means auto, capped at 64 workers."""
    available = max(1, os.cpu_count() or 1)
    if requested == 0:
        return min(available, MAX_TRANSCODE_WORKERS)
    return min(requested, MAX_TRANSCODE_WORKERS)
