"""Command-line interface and batch processing."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import os
import shlex
import sys
import threading
from pathlib import Path
from typing import Any, TextIO

from capture_processing.core.processing_log import (
    load_processing_log,
    get_missing_steps,
    PROCESSING_STEPS,
)
from capture_processing.core.vci_pipeline import run_dataset_creation
from capture_processing.core.thuman_pipeline import run_thuman_dataset_creation
from capture_processing.utils.devices import parse_device_pool
from capture_processing.utils.smplx_mhr import add_conversion_arguments, options_from_args
from capture_processing.utils.mhr import (
    DEFAULT_MHR_ITERATIONS,
    any_subject_needs_mhr,
    default_sam3d_body_rerun_repo,
    prepare_mhr_runtime,
    resolve_sam3d_body_rerun_repo,
)


class _ThreadLogRouter:
    """Route writes from worker threads to per-object logs."""

    def __init__(self, fallback: TextIO) -> None:
        self._fallback = fallback
        self._local = threading.local()
        self._lock = threading.Lock()
        self.encoding = getattr(fallback, "encoding", "utf-8")
        self.errors = getattr(fallback, "errors", "replace")

    def set_thread_stream(self, stream: TextIO | None) -> None:
        self._local.stream = stream

    def write(self, text: str) -> int:
        stream = getattr(self._local, "stream", None)
        if stream is not None:
            return stream.write(text)
        with self._lock:
            return self._fallback.write(text)

    def flush(self) -> None:
        stream = getattr(self._local, "stream", None)
        if stream is not None:
            stream.flush()
            return
        with self._lock:
            self._fallback.flush()

    def isatty(self) -> bool:
        stream = getattr(self._local, "stream", None)
        return stream.isatty() if stream is not None else self._fallback.isatty()

    def fileno(self) -> int:
        return self._fallback.fileno()


@contextmanager
def _route_thread_output(stdout: _ThreadLogRouter, stderr: _ThreadLogRouter, stream: TextIO):
    stdout.set_thread_stream(stream)
    stderr.set_thread_stream(stream)
    try:
        yield
    finally:
        stdout.set_thread_stream(None)
        stderr.set_thread_stream(None)


def _summarize_parallel_failure(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    first_line = text.splitlines()[0].strip()
    return first_line if len(first_line) <= 240 else first_line[:237] + "..."


THUMAN_IGNORED_OPTIONS = (
    "calibration", "resize_long_edge", "jpeg_quality", "workers", "skip_images",
    "drop_cameras", "resume_validation", "verbose",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for batch dataset creation."""
    parser = argparse.ArgumentParser(
        description="Batch convert object directories to THUman4.0 format. Supports resuming from checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help=(
            "Input dataset root. For --dataset-kind vci, this is a parent containing "
            "object subdirectories with frame_* folders. For thuman, this is a parent "
            "containing subject directories with images/ and calibration.json."
        ),
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Output parent directory; object folders will be created inside.",
    )
    parser.add_argument(
        "--dataset-kind",
        choices=("vci", "thuman"),
        default="vci",
        help="Input dataset layout to process.",
    )
    parser.add_argument(
        "--calibration",
        default="calibration_dome.json",
        help="Source calibration filename inside each object directory.",
    )
    parser.add_argument(
        "--smplx-source",
        choices=("mamma", "existing"),
        default="mamma",
        help="Where smpl_params.npz should come from: fit with the local MAMMA checkout, or (THuman only) reuse the copied file.",
    )
    parser.add_argument(
        "--mamma-repo-path",
        type=Path,
        default=None,
        help="Optional override path to a MAMMA checkout. Defaults to MAMMA_REPO_PATH, then external/mamma.",
    )
    parser.add_argument(
        "--mamma-preset",
        type=Path,
        default=None,
        help="MAMMA preset path. Relative paths are resolved inside the MAMMA repo.",
    )
    parser.add_argument(
        "--mamma-out-tag",
        default="vci",
        help="MAMMA output tag used under mamma_output/ma_*.",
    )
    parser.add_argument(
        "--mamma-python",
        default="conda run -n mamma python",
        help="Python command used to launch MAMMA, e.g. python or conda run -n mamma python.",
    )
    parser.add_argument(
        "--mamma-force",
        action="store_true",
        help="Force MAMMA to rerun steps even when DONE sentinels exist.",
    )
    parser.add_argument(
        "--mamma-workers",
        type=int,
        default=4,
        help="CPU frame preprocessing threads for MAMMA landmarks (one GPU model).",
    )
    parser.add_argument(
        "--mamma-batch-size", type=int, default=8,
        help="Frames per MAMMA landmark inference batch; reduce if GPU memory is limited.",
    )
    parser.add_argument(
        "--mamma-confidence-fit", action=argparse.BooleanOptionalAction, default=True,
        help="Also fit confidence-filtered MAMMA alongside the default; reuse its 2D detections. Enabled by default.",
    )
    from capture_processing.utils.semantic_parsing import add_arguments as add_semantic_arguments
    add_semantic_arguments(parser)
    add_conversion_arguments(parser)
    parser.add_argument(
        "--mhr-source",
        choices=("none", "sam3d-body-rerun"),
        default="sam3d-body-rerun",
        help="MHR parameterization source. Defaults to sam3d-body-rerun; use none to skip MHR.",
    )
    parser.add_argument(
        "--mhr-repo-path",
        type=Path,
        default=None,
        help="Optional override path to a sam3d-body-rerun checkout. Defaults to SAM3D_BODY_RERUN_REPO, then external/sam3d-body-rerun.",
    )
    parser.add_argument(
        "--mhr-python",
        default=None,
        help="Python command for MHR. Defaults to the SAM3D Pixi interpreter when installed, otherwise python.",
    )
    parser.add_argument(
        "--mhr-max-frames",
        type=int,
        default=None,
        help="Only estimate MHR parameters for the first N frames of each subject. Default: all frames.",
    )
    parser.add_argument(
        "--mhr-iterations",
        type=int,
        default=DEFAULT_MHR_ITERATIONS,
        help="Multiview MHR optimization iterations per frame.",
    )
    parser.add_argument(
        "--mhr-workers",
        type=int,
        default=1,
        help="Frame-shard worker subprocesses per subject for MHR estimation.",
    )
    parser.add_argument(
        "--mhr-force",
        action="store_true",
        help="Rerun MHR parameterization even when mhr_params.npz already exists.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Device for MAMMA/MHR GPU work. Examples: cpu, 0, 1, cuda:0, cuda:1. "
            "GPU values set CUDA_VISIBLE_DEVICES for GPU subprocesses."
        ),
    )
    parser.add_argument(
        "--devices",
        default=None,
        help=(
            "Comma-separated device pool for parallel subject/object processing, e.g. 0,1,2,3 "
            "or cuda:0,cuda:1. Each subject/object job is assigned one free GPU slot. Mutually "
            "exclusive with --device."
        ),
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=1,
        help=(
            "Concurrent subject/object jobs per selected GPU. For example, "
            "--device 0 --gpu-workers 4 runs four jobs on GPU 0; "
            "--devices 0,1 --gpu-workers 2 runs two jobs per GPU."
        ),
    )
    parser.add_argument(
        "--resize-long-edge",
        type=int,
        default=None,
        help="Resize images so their long edge equals this (px). Default: keep source resolution.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--workers",
        "--worker",
        type=int,
        default=32,
        help=(
            "Transcode worker processes. Use 0 for auto "
            "(all available CPU cores, capped at 64). Values above 64 are capped."
        ),
    )
    parser.add_argument(
        "--max-subjects",
        "--subjects",
        type=int,
        default=None,
        help="Only process the first N discovered subjects/objects. Default: process all.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In parallel --devices/--gpu-workers mode, stream full per-object output to the terminal instead of hiding it in batch_logs.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Only write calibration + smpl_params (debug).",
    )
    parser.add_argument(
        "--drop-cameras",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Source camera IDs to exclude (e.g. C0007 C0019), or a single path to a text file "
            "with one source ID per line ('#' comments allowed). Surviving cameras are renumbered "
            "contiguously to cam00..camNN; camera_id_mapping.json records the cam## -> source-ID mapping."
        ),
    )
    parser.add_argument(
        "--resume-validation", choices=("full", "quick"), default="full",
        help="Quick trusts logged media completion for immutable inputs; full validates every image/mask.",
    )
    args = parser.parse_args(argv)
    if args.dataset_kind == "thuman":
        # THuman subjects are copied as already-converted media; these options only
        # affect VCI conversion and would otherwise be silently ignored.
        ignored = [f"--{name.replace('_', '-')}" for name in THUMAN_IGNORED_OPTIONS
                   if getattr(args, name) != parser.get_default(name)]
        if ignored:
            parser.error(f"{', '.join(ignored)} only appl{'ies' if len(ignored) == 1 else 'y'} to --dataset-kind vci")
    elif args.smplx_source == "existing":
        parser.error("--smplx-source existing is only supported with --dataset-kind thuman")
    if args.mhr_python is None:
        sam3d_path = args.mhr_repo_path or os.environ.get("SAM3D_BODY_RERUN_REPO")
        sam3d_repo = Path(sam3d_path).expanduser().resolve() if sam3d_path else default_sam3d_body_rerun_repo()
        interpreter = sam3d_repo / ".pixi/envs/dev/bin/python"
        args.mhr_python = shlex.quote(str(interpreter)) if interpreter.is_file() else "python"
    from capture_processing.utils.preprocessing_validation import validate_options
    try:
        validate_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _find_object_directories(parent: Path) -> list[Path]:
    """Find all subdirectories that contain frame_* dirs (object directories)."""
    objects: list[Path] = []
    for item in sorted(parent.iterdir()):
        if not item.is_dir():
            continue
        # Check if it looks like an object (has frame_* subdirectories)
        if any(d.is_dir() and d.name.startswith("frame_") for d in item.iterdir()):
            objects.append(item)
    return objects


def run_batch_dataset_creation(
    input_path: Path,
    out_dir: Path,
    calibration: str,
    smplx_source: str,
    mamma_repo_path: Path | None,
    mamma_preset: Path | None,
    mamma_out_tag: str,
    mamma_python: str,
    mamma_force: bool,
    mhr_source: str,
    mamma_workers: int,
    mhr_repo_path: Path | None,
    mhr_python: str,
    mhr_max_frames: int | None,
    mhr_iterations: int,
    mhr_force: bool,
    mhr_workers: int,
    device: str | None,
    devices: str | None,
    resize_long_edge: int | None,
    gpu_workers: int,
    jpeg_quality: int,
    workers: int,
    skip_images: bool,
    drop_cameras: list[str] | None,
    max_subjects: int | None,
    verbose: bool,
    mamma_batch_size: int = 8,
    smplx_mhr_options=None,
    resume_validation="full",
    mamma_confidence_fit=True,
    semantic_options=None,
) -> list[Path]:
    """Process all object subdirectories. Return list of output paths."""
    input_path = input_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    objects = _find_object_directories(input_path)
    if not objects:
        raise SystemExit(
            f"No object directories found in {input_path}\n"
            f"  Expected: subdirectories containing frame_* folders"
        )

    if max_subjects is not None:
        if max_subjects <= 0:
            raise SystemExit("--max-subjects must be a positive integer")
        total_objects = len(objects)
        objects = objects[:max_subjects]
        print(f"Limiting to first {len(objects)} of {total_objects} discovered object(s)")

    output_roots = [out_dir / obj.name for obj in objects]
    if mhr_source != "none" and any_subject_needs_mhr(output_roots, mhr_force):
        mhr_repo = resolve_sam3d_body_rerun_repo(mhr_repo_path)
        prepare_mhr_runtime(mhr_python, mhr_repo)

    device_pool = parse_device_pool(device, devices, gpu_workers)

    print(f"Found {len(objects)} object(s) to process")
    if len(device_pool) > 1:
        print(
            f"Parallel object processing enabled on {len(device_pool)} device(s): "
            f"{', '.join(device_pool)}"
        )
    elif device_pool:
        print(f"Using device: {device_pool[0]}")

    object_options = {
        "resume_validation": resume_validation,
        "calibration": calibration,
        "smplx_source": smplx_source,
        "mamma_repo_path": mamma_repo_path,
        "mamma_preset": mamma_preset,
        "mamma_out_tag": mamma_out_tag,
        "mamma_python": mamma_python,
        "mamma_force": mamma_force,
        "mhr_source": mhr_source,
        "smplx_mhr_options": smplx_mhr_options,
        "mamma_workers": mamma_workers,
        "mamma_batch_size": mamma_batch_size,
        "mamma_confidence_fit": mamma_confidence_fit,
        "semantic_options": semantic_options,
        "mhr_repo_path": mhr_repo_path,
        "mhr_python": mhr_python,
        "mhr_max_frames": mhr_max_frames,
        "mhr_iterations": mhr_iterations,
        "mhr_force": mhr_force,
        "mhr_workers": mhr_workers,
        "resize_long_edge": resize_long_edge,
        "jpeg_quality": jpeg_quality,
        "workers": workers,
        "skip_images": skip_images,
        "drop_cameras": drop_cameras,
    }

    if len(device_pool) > 1:
        output_paths = _run_objects_parallel(
            objects=objects,
            out_dir=out_dir,
            devices=device_pool,
            verbose=verbose,
            object_options=object_options,
        )
    else:
        active_device = device_pool[0] if device_pool else None
        output_paths = [
            _run_one_object(
                obj_dir=obj_dir,
                out_dir=out_dir,
                device=active_device,
                object_options=object_options,
            )
            for obj_dir in objects
        ]

    print(f"\n{'='*60}")
    print("Batch processing complete!")
    print(f"{'='*60}")
    print(f"Processed {len(output_paths)} object(s)")
    for out_path in output_paths:
        print(f"  - {out_path}")

    return output_paths


def _run_one_object(
    *,
    obj_dir: Path,
    out_dir: Path,
    device: str | None,
    object_options: dict[str, Any],
) -> Path:
    print(f"\n{'='*60}")
    suffix = f" on device {device}" if device else ""
    print(f"Processing: {obj_dir.name}{suffix}")
    print(f"{'='*60}")

    out_root = out_dir / obj_dir.name
    if out_root.exists():
        log = load_processing_log(out_root)
        from capture_processing.utils.semantic_parsing import options_from_args as semantic_options
        semantic = semantic_options(argparse.Namespace(**object_options))
        missing = get_missing_steps(log, semantic_enabled=semantic.enabled)
        if not missing:
            print(
                "✓ Requested processing checklist complete; checking cached artifacts."
            )
        else:
            print(f"Resuming: {len(missing)} step(s) remaining")
            for step in missing:
                print(f"  - {PROCESSING_STEPS[step]}")

    obj_args = argparse.Namespace(
        input_path=obj_dir,
        out_dir=out_dir,
        device=device,
        **object_options,
    )

    try:
        return run_dataset_creation(obj_args)
    except SystemExit as e:
        print(f"[ERROR] Failed to process {obj_dir.name}: {e}", file=sys.stderr)
        raise


def _run_objects_parallel(
    *,
    objects: list[Path],
    out_dir: Path,
    devices: list[str],
    verbose: bool,
    object_options: dict[str, Any],
) -> list[Path]:
    device_queue: Queue[str] = Queue()
    for item in devices:
        device_queue.put(item)

    log_dir = out_dir / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_router = _ThreadLogRouter(sys.stdout)
    stderr_router = _ThreadLogRouter(sys.stderr)
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    def console(message: str, *, error: bool = False) -> None:
        target = original_stderr if error else original_stdout
        print(message, file=target, flush=True)

    def log_path_for(obj_dir: Path) -> Path:
        return log_dir / f"{obj_dir.name}.log"

    def run_with_device(obj_dir: Path) -> Path:
        active_device = device_queue.get()
        path = log_path_for(obj_dir)
        try:
            if verbose:
                console(
                    f"[start] {obj_dir.name} on device {active_device} "
                    "(verbose output enabled)"
                )
                return _run_one_object(
                    obj_dir=obj_dir,
                    out_dir=out_dir,
                    device=active_device,
                    object_options=object_options,
                )

            console(f"[start] {obj_dir.name} on device {active_device} (log: {path})")
            with path.open("w", encoding="utf-8") as log_file:
                with _route_thread_output(stdout_router, stderr_router, log_file):
                    return _run_one_object(
                        obj_dir=obj_dir,
                        out_dir=out_dir,
                        device=active_device,
                        object_options=object_options,
                    )
        finally:
            device_queue.put(active_device)

    from capture_processing.utils.subject_lock import SubjectBusyError
    output_paths: list[Path] = []
    failures: list[str] = []
    all_busy = True
    if not verbose:
        sys.stdout = stdout_router
        sys.stderr = stderr_router
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(run_with_device, obj_dir): obj_dir for obj_dir in objects}
            for future in as_completed(futures):
                obj_dir = futures[future]
                path = log_path_for(obj_dir)
                try:
                    output_path = future.result()
                    output_paths.append(output_path)
                    console(f"[done]  {obj_dir.name} -> {output_path}")
                except (Exception, SystemExit) as exc:
                    summary = _summarize_parallel_failure(exc)
                    log_suffix = "" if verbose else f" (log: {path})"
                    failures.append(f"{obj_dir.name}: {summary}{log_suffix}")
                    all_busy = all_busy and isinstance(exc, SubjectBusyError)
                    console(f"[failed] {obj_dir.name}: {summary}{log_suffix}", error=True)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    if failures and all_busy:
        raise SubjectBusyError("All selected subjects are locked by other processes:\n"
                               + "\n".join(f"  - {failure}" for failure in failures))
    if failures:
        raise SystemExit(
            "One or more objects failed during parallel processing:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
    return output_paths


def main(argv: list[str] | None = None) -> list[Path]:
    """CLI entrypoint. Processes all selected subjects in the input folder."""
    args = parse_args(argv)
    from capture_processing.utils.semantic_parsing import options_from_args as semantic_options_from_args
    semantic_options = semantic_options_from_args(args)
    semantic_options.validate()
    if args.dataset_kind == "thuman":
        return run_thuman_dataset_creation(
            input_path=args.input_path,
            out_dir=args.out_dir,
            smplx_source=args.smplx_source,
            mamma_repo_path=args.mamma_repo_path,
            mamma_preset=args.mamma_preset,
            mamma_out_tag=args.mamma_out_tag,
            mamma_python=args.mamma_python,
            mamma_force=args.mamma_force,
            mhr_source=args.mhr_source,
            smplx_mhr_options=options_from_args(args),
            mamma_workers=args.mamma_workers,
            mamma_batch_size=args.mamma_batch_size,
            mamma_confidence_fit=args.mamma_confidence_fit,
            semantic_options=semantic_options,
            mhr_repo_path=args.mhr_repo_path,
            mhr_python=args.mhr_python,
            mhr_max_frames=args.mhr_max_frames,
            mhr_iterations=args.mhr_iterations,
            mhr_force=args.mhr_force,
            mhr_workers=args.mhr_workers,
            device=args.device,
            devices=args.devices,
            max_subjects=args.max_subjects,
            gpu_workers=args.gpu_workers,
        )

    return run_batch_dataset_creation(
        input_path=args.input_path,
        out_dir=args.out_dir,
        resume_validation=args.resume_validation,
        calibration=args.calibration,
        smplx_source=args.smplx_source,
        mamma_repo_path=args.mamma_repo_path,
        mamma_preset=args.mamma_preset,
        mamma_out_tag=args.mamma_out_tag,
        mamma_python=args.mamma_python,
        mamma_force=args.mamma_force,
        mhr_source=args.mhr_source,
        smplx_mhr_options=options_from_args(args),
        mamma_workers=args.mamma_workers,
        mamma_batch_size=args.mamma_batch_size,
        mamma_confidence_fit=args.mamma_confidence_fit,
        semantic_options=semantic_options,
        mhr_repo_path=args.mhr_repo_path,
        mhr_python=args.mhr_python,
        mhr_max_frames=args.mhr_max_frames,
        mhr_iterations=args.mhr_iterations,
        mhr_force=args.mhr_force,
        mhr_workers=args.mhr_workers,
        device=args.device,
        devices=args.devices,
        resize_long_edge=args.resize_long_edge,
        gpu_workers=args.gpu_workers,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        skip_images=args.skip_images,
        drop_cameras=args.drop_cameras,
        max_subjects=args.max_subjects,
        verbose=args.verbose,
    )
