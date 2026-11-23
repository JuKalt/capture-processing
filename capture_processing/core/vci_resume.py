"""Validate persisted VCI stages before reusing their completion markers."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def media_signature(jobs, context=None):
    payload = []
    for image, _, mask, _, resize, quality in jobs:
        inputs = [[str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]
                  for path in (image, mask)]
        payload.append([inputs, resize, quality])
    return hashlib.sha256(json.dumps([payload, context], sort_keys=True, default=lambda value: value.tolist()).encode()).hexdigest()


def media_pair_signature(job):
    """Independent source/settings signature, suitable for interrupted-run receipts."""
    return media_signature([job])


def media_receipt_path(output, job):
    return output / '.media_state' / job[1].relative_to(output / 'images').with_suffix('.json')


def read_media_receipt(output, job):
    try:
        value = json.loads(media_receipt_path(output, job).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _file_digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record_media_pair(output, job, signature):
    """Checkpoint one successful pair without rewriting a subject-wide manifest."""
    from ..utils.stage_cache import file_stamp
    receipt = read_media_receipt(output, job)
    outputs = {str(path.relative_to(output)): file_stamp(path) for path in (job[1], job[3])}
    previous = receipt.get('outputs', {})
    previous = previous if isinstance(previous, dict) else {}
    if (receipt.get('signature') == signature
            and all(isinstance(previous.get(key), dict) and previous[key].get('stamp') == stamp
                    for key, stamp in outputs.items())):
        return
    path = media_receipt_path(output, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {'signature': signature, 'outputs': {
        str(file.relative_to(output)): {'stamp': outputs[str(file.relative_to(output))],
                                       'sha256': _file_digest(file)} for file in (job[1], job[3])}}
    import uuid
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fitted_media_signature(output):
    """Track converted content; identical repairs/timestamp changes preserve fits.

    Hashes are cached against output stamps in per-pair receipts, so healthy
    full validations do not reread all JPEG bytes just to identify fit inputs.
    """
    from ..utils.stage_cache import file_stamp
    inputs = []
    for folder in ('images', 'masks'):
        for path in sorted((output / folder).rglob('*.jpg')):
            relative = path.relative_to(output / folder)
            name = relative.name.replace('mask_', 'color_', 1) if folder == 'masks' else relative.name
            receipt_path = output / '.media_state' / relative.with_name(name).with_suffix('.json')
            try:
                receipt = json.loads(receipt_path.read_text())
                saved = receipt.get('outputs', {}).get(str(path.relative_to(output)), {})
            except (OSError, ValueError, AttributeError):
                saved = {}
            saved = saved if isinstance(saved, dict) else {}
            content = (saved.get('sha256') if saved.get('stamp') == file_stamp(path) else None)
            inputs.append([str(path.relative_to(output)), content or _file_digest(path)])
    try:
        cameras = json.loads((output / 'calibration.json').read_text())
    except (OSError, ValueError):
        cameras = None
    return hashlib.sha256(json.dumps([inputs, cameras], sort_keys=True).encode()).hexdigest()


def valid_media_pair(job, *, check_binary_mask=True):
    from PIL import Image
    from capture_processing.utils.camera import resize_to_long_edge
    image, output, mask_source, mask_output, resize, _ = job
    if any(path.suffix.lower() == ".avif" for path in (image, mask_source)):
        import pillow_avif  # noqa: F401  (register AVIF support)
    try:
        with Image.open(image) as source:
            expected = resize_to_long_edge(*source.size, resize) if resize is not None else source.size
        for path in (output, mask_output):
            with Image.open(path) as media:
                if media.size != expected or media.format != 'JPEG':
                    return False
                media.load()
        # Adopt old runs, repairing only masks affected by the former 0/1 JPEG bug.
        if check_binary_mask:
            with Image.open(mask_output) as encoded:
                encoded_maximum = encoded.convert('L').getextrema()[1]
            if encoded_maximum < 128:
                with Image.open(mask_source) as source:
                    if source.convert('L').getextrema()[1] == 1:
                        return False
        return True
    except (OSError, ValueError):
        return False


def valid_json(path, expected=None):
    try:
        value = json.loads(Path(path).read_text())
        return isinstance(value, dict) and (expected is None or value == expected)
    except (OSError, ValueError):
        return False


def valid_smplx(path, frame_count):
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as data:
            shapes = {'global_orient': 3, 'transl': 3, 'body_pose': 63,
                      'jaw_pose': 3, 'left_hand_pose': 45, 'right_hand_pose': 45}
            return (all(data[key].shape == (frame_count, width)
                        and np.isfinite(data[key]).all() for key, width in shapes.items())
                    and data['betas'].ndim == 2 and data['betas'].shape[0] in (1, frame_count)
                    and data['betas'].shape[1] > 0 and np.isfinite(data['betas']).all()
                    and ('frame_ids' not in data or (data['frame_ids'].shape == (frame_count,)
                         and data['frame_ids'].dtype.kind in 'iu' and (data['frame_ids'] >= 0).all()
                         and len(np.unique(data['frame_ids'])) == frame_count)))
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        return False


def _quick_drop_cameras(args):
    from ..utils.camera import read_camera_list_arg
    return read_camera_list_arg(getattr(args, "drop_cameras", None)) or None


def quick_settings(args):
    """Portable conversion settings; interpreter paths/worker counts don't affect outputs."""
    calibration = args.input_path.expanduser().resolve() / args.calibration
    return {
        "calibration": args.calibration,
        "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "resize_long_edge": args.resize_long_edge,
        "jpeg_quality": args.jpeg_quality,
        "mask_source": "existing",  # Stored for log compatibility; the only mask source.
        "drop_cameras": _quick_drop_cameras(args),
        "smplx_source": getattr(args, "smplx_source", "mamma"),
        "mamma_preset": str(getattr(args, "mamma_preset", None)),
    }


def quick_media_complete(args, output, log):
    """Opt-in reuse of a logged media stage, treating source/media as immutable.

    Older logs lack settings. Adopt them only for the original CLI defaults;
    custom settings require a full validation before they can use quick resume.
    """
    if getattr(args, "resume_validation", "full") != "quick":
        return False
    if "images_transcoded" not in log.get("steps_completed", {}):
        return False
    try:
        # Legacy logs have no source hash. Compare expected geometry as well so a
        # repaired source can never quick-skip an old converted calibration.
        from ..utils.camera import (load_source_calibration, build_camera_id_mapping,
                                    resolve_drop_tokens, apply_camera_filter,
                                    build_resize_sizes, thuman4_calibration)
        cameras = load_source_calibration(args.input_path / args.calibration)
        mapping = build_camera_id_mapping(list(cameras))
        drops = _quick_drop_cameras(args)
        if drops:
            mapping = apply_camera_filter(mapping, resolve_drop_tokens(drops, set(cameras)))
        sizes = build_resize_sizes(mapping, cameras, args.resize_long_edge)
        if (not valid_json(output / 'calibration.json', thuman4_calibration(mapping, cameras, sizes))
                or not valid_json(output / 'camera_id_mapping.json', mapping)):
            return False
        settings = quick_settings(args)
        previous = log.get("quick_resume_settings")
        if previous is not None:
            if previous != settings:
                return False
        elif not (settings["calibration"] == "calibration_dome.json"
                  and settings["resize_long_edge"] is None
                  and settings["jpeg_quality"] == 95
                  and not settings["drop_cameras"]
                  and settings["smplx_source"] == "mamma"
                  and settings["mamma_preset"] == "None"):
            return False
        return all((output / name).is_dir() for name in ("images", "masks"))
    except (OSError, ValueError, AttributeError):
        return False


def quick_conversion_complete(args, output, log):
    from .processing_log import get_missing_conversion_steps
    if (get_missing_conversion_steps(log) or log.get("smplx_force_pending") or log.get("media_update_pending")
            or getattr(args, "mamma_force", False)
            or not quick_media_complete(args, output, log)):
        return False
    try:
        from ..utils.preprocessing_provenance import baseline_dependencies
        if log.get('smplx_dependency_signature') != baseline_dependencies(args, output)[0]:
            return False
        import numpy as np
        with np.load(output / "smpl_params.npz", allow_pickle=False) as data:
            frame_count = data["transl"].shape[0]
        from ..utils.smplx_hands import needs_hand_conversion
        return (frame_count > 0 and valid_smplx(output / "smpl_params.npz", frame_count)
                and not needs_hand_conversion(output)
                and valid_json(output / "calibration.json")
                and valid_json(output / "camera_id_mapping.json")
                and (output / "missing_img_files.txt").is_file())
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile):
        return False


def quick_subject_complete(args, output):
    """Check all requested stages before the subject wrapper reserves GPU memory.

    Quick mode trusts unchanged datasets/dependencies and historical stage markers.
    Full mode remains the integrity check for edits to source or media files.
    """
    from .processing_log import load_processing_log
    from ..utils.mhr import is_mhr_params_complete, DEFAULT_MHR_ITERATIONS, VISUALIZATION_SETTINGS
    from ..utils.stage_cache import rendering_complete
    log = load_processing_log(output)
    if not quick_conversion_complete(args, output, log):
        return False
    # The subject wrapper can return before run_dataset_creation writes provenance.
    # Missing/failed legacy manifests must therefore take the normal resume path.
    from ..utils.subject_manifest import SCHEMA_VERSION
    try:
        manifest = json.loads((output / 'subject_manifest.json').read_text())
        if manifest.get('schema_version') != SCHEMA_VERSION or manifest.get('complete') is not True:
            return False
    except (OSError, ValueError, AttributeError):
        return False
    from ..utils.mamma_additional_fit import requested as confidence_requested, confidence_fit_complete
    if confidence_requested(args):
        from ..utils.mamma import default_mamma_repo
        repo = Path(getattr(args, 'mamma_repo_path', None) or default_mamma_repo())
        if not confidence_fit_complete(output, repo, getattr(args, 'mamma_out_tag', 'vci')):
            return False
    from ..utils.semantic_parsing import options_from_args as semantic_options, semantic_complete
    semantic = semantic_options(args)
    if (semantic.enabled and "semantic_parsing" not in log["steps_completed"]) or not semantic_complete(output, semantic):
        return False
    conversion = getattr(args, "smplx_mhr_options", None)
    if (getattr(args, "mhr_force", False) or (conversion and conversion.force)
            or getattr(args, "mhr_max_frames", None) is not None
            or getattr(args, "mhr_iterations", DEFAULT_MHR_ITERATIONS) != DEFAULT_MHR_ITERATIONS
            or getattr(args, "mamma_preset", None) is not None
            or any(getattr(args, name, None) is not None for name in (
                "mamma_repo_path", "mhr_repo_path", "smplx_to_mhr_repo_path", "smplx_to_mhr_model_path"))):
        return False
    try:
        folders = []
        from ..utils.stage_cache import rendering_signature
        from ..utils.silhouette_metrics import resolve_paths, options_from_pipeline
        official, sam3d, model = resolve_paths(options_from_pipeline(args))
        requested = ["smplx"] + (["smplx_confidence"] if confidence_requested(args) else [])
        if conversion and conversion.enabled:
            from ..utils.subject_validation import converted_mhr_complete
            from ..utils.smplx_mhr import conversion_signature
            signature = conversion_signature(output / 'smpl_params.npz', model, official)
            if not converted_mhr_complete(output / 'mhr_from_smplx_params.npz', signature):
                return False
            folders.append("mhr_from_smplx_visualization")
            requested.append("mhrfromsmplx")
        if getattr(args, "mhr_source", "none") != "none":
            from ..utils.mhr import estimation_signature
            signature = estimation_signature(output, sam3d, None, DEFAULT_MHR_ITERATIONS)
            if not is_mhr_params_complete(output / "mhr_params.npz", signature):
                return False
            folders.append("mhr_visualization")
            requested.append("mhr")
        for folder in folders:
            kind = 'mhr' if folder == 'mhr_visualization' else 'mhrfromsmplx'
            signature = rendering_signature(output, sam3d if kind == 'mhr' else official, kind,
                                            dict(VISUALIZATION_SETTINGS))
            if not rendering_complete(output / folder, signature):
                return False
        from ..utils.subject_validation import cached_metrics_complete
        return cached_metrics_complete(output, requested, options_from_pipeline(args))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, zipfile.BadZipFile):
        return False
