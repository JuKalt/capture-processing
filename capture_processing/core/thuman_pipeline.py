"""Copy and process stored THuman-style subjects."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from tqdm import tqdm

from capture_processing.core.smplx_pipeline import ensure_subject_smplx
from capture_processing.utils.fit_metrics import write_fit_metrics, validate_fit_metrics
from capture_processing.utils.silhouette_metrics import SilhouetteOptions
from capture_processing.utils.smplx_mhr import convert_subject_smplx_to_mhr
from capture_processing.utils.mhr import (
    any_subject_needs_mhr,
    maybe_create_mhr_params,
    prepare_mhr_runtime,
    resolve_sam3d_body_rerun_repo,
)
from capture_processing.utils.devices import parse_device_pool
from capture_processing.utils.paths import paths_overlap

COPY_DIRS = ("images", "masks")
COPY_INPUT_FILES = (
    "calibration.json",
    "camera_id_mapping.json",
    "missing_img_files.txt",
    "missing_msk_files.txt",
)
COPY_RESUMABLE_FILES = ("smpl_params.npz", "mhr_params.npz")


def find_thuman_subjects(root: Path) -> list[Path]:
    subjects: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "images").is_dir() and (path / "calibration.json").is_file():
            subjects.append(path)
    return subjects


def run_thuman_dataset_creation(
    *,
    input_path: Path,
    out_dir: Path,
    smplx_source: str,
    mamma_repo_path: Path | None,
    mamma_preset: Path | None,
    mamma_out_tag: str,
    mamma_python: str,
    mamma_force: bool,
    mhr_source: str,
    mamma_workers: int,
    mamma_batch_size: int = 8,
    mhr_repo_path: Path | None,
    mhr_python: str,
    mhr_max_frames: int | None,
    mhr_iterations: int,
    mhr_force: bool,
    mhr_workers: int,
    device: str | None,
    devices: str | None,
    max_subjects: int | None,
    gpu_workers: int,
    smplx_mhr_options=None,
    mamma_confidence_fit=True,
    semantic_options=None,
) -> list[Path]:
    source_root = input_path.expanduser().resolve()
    output_root = out_dir.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Not a directory: {source_root}")
    if paths_overlap(source_root, output_root):
        raise SystemExit(
            "THuman output parent must not overlap the source dataset.\n"
            f"  source: {source_root}\n"
            f"  output: {output_root}"
        )
    device_slots = parse_device_pool(device, devices, gpu_workers)

    subjects = find_thuman_subjects(source_root)
    if not subjects:
        raise SystemExit(
            f"No THuman subjects found in {source_root}\n"
            "  Expected subdirectories with images/ and calibration.json"
        )
    if max_subjects is not None:
        if max_subjects <= 0:
            raise SystemExit("--max-subjects must be a positive integer")
        total = len(subjects)
        subjects = subjects[:max_subjects]
        print(f"Limiting to first {len(subjects)} of {total} THuman subject(s)")

    target_roots = [output_root / subject.name for subject in subjects]
    if mhr_source != "none" and any_subject_needs_mhr(target_roots, mhr_force):
        repo = resolve_sam3d_body_rerun_repo(mhr_repo_path)
        prepare_mhr_runtime(mhr_python, repo)

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(subjects)} THuman subject(s) to copy/process")
    subject_options = {
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
    }
    if len(device_slots) > 1:
        print(
            f"Parallel THuman processing enabled with {len(device_slots)} worker(s): "
            f"{', '.join(device_slots)}"
        )
        output_paths = _process_thuman_subjects_parallel(
            subjects,
            output_root=output_root,
            device_slots=device_slots,
            subject_options=subject_options,
        )
    else:
        active_device = device_slots[0] if device_slots else None
        output_paths = [
            _process_thuman_subject(
                subject,
                output_root / subject.name,
                device=active_device,
                **subject_options,
            )
            for subject in tqdm(subjects, desc="THuman subjects", unit="subject")
        ]

    print(f"\nTHuman processing complete: {len(output_paths)} subject(s)")
    for path in output_paths:
        print(f"  - {path}")
    return output_paths


def _process_thuman_subjects_parallel(
    subjects: list[Path],
    *,
    output_root: Path,
    device_slots: list[str],
    subject_options: dict,
) -> list[Path]:
    device_queue: Queue[str] = Queue()
    for item in device_slots:
        device_queue.put(item)

    def run_with_device(subject: Path) -> Path:
        active_device = device_queue.get()
        try:
            print(f"[start] {subject.name} on device {active_device}", flush=True)
            path = _process_thuman_subject(
                subject,
                output_root / subject.name,
                device=active_device,
                **subject_options,
            )
            print(f"[done]  {subject.name} -> {path}", flush=True)
            return path
        finally:
            device_queue.put(active_device)

    output_paths: list[Path] = []
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=len(device_slots)) as pool:
        futures = {pool.submit(run_with_device, subject): subject for subject in subjects}
        for future in as_completed(futures):
            subject = futures[future]
            try:
                output_paths.append(future.result())
            except (Exception, SystemExit) as exc:
                message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
                failures.append(f"{subject.name}: {message}")

    if failures:
        raise SystemExit(
            "One or more THuman subjects failed during parallel processing:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )
    return output_paths


def _process_thuman_subject(
    source: Path,
    target: Path,
    *,
    smplx_source: str,
    mamma_repo_path: Path | None,
    mamma_preset: Path | None,
    mamma_out_tag: str,
    mamma_python: str,
    mamma_force: bool,
    mhr_source: str,
    mamma_workers: int,
    mamma_batch_size: int = 8,
    mhr_repo_path: Path | None,
    mhr_python: str,
    mhr_max_frames: int | None,
    mhr_iterations: int,
    mhr_force: bool,
    mhr_workers: int,
    device: str | None,
    smplx_mhr_options=None,
    mamma_confidence_fit=True,
    semantic_options=None,
) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for name in COPY_DIRS:
        src = source / name
        if src.is_dir():
            shutil.copytree(src, target / name, dirs_exist_ok=True, copy_function=_copy_if_changed)
    for name in COPY_INPUT_FILES:
        src = source / name
        if src.is_file():
            _copy_if_changed(src, target / name)
    for name in COPY_RESUMABLE_FILES:
        src = source / name
        dst = target / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)

    ensure_subject_smplx(
        target,
        smplx_source=smplx_source,
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
    from ..utils.smplx_hands import needs_hand_conversion, normalize_subject_hands
    if smplx_source == 'mamma' and needs_hand_conversion(target):
        from ..utils.mamma import resolve_mamma_repo
        normalize_subject_hands(target, resolve_mamma_repo(mamma_repo_path), mamma_out_tag)

    convert_subject_smplx_to_mhr(target, smplx_mhr_options)
    maybe_create_mhr_params(
        target,
        source=mhr_source,
        repo_path=mhr_repo_path,
        python_executable=mhr_python,
        device=device,
        max_frames=mhr_max_frames,
        iterations=mhr_iterations,
        force=mhr_force,
        workers=mhr_workers,
    )
    report = write_fit_metrics(target, SilhouetteOptions(
        smplx_python=mamma_python, sam3d_python=mhr_python, sam3d_repo=mhr_repo_path,
        official_python=smplx_mhr_options.python if smplx_mhr_options else None,
        official_repo=smplx_mhr_options.repo_path if smplx_mhr_options else None,
        smplx_model=smplx_mhr_options.model_path if smplx_mhr_options else None))
    required = ['smplx']
    if mhr_source != 'none':
        required.append('mhr')
    if smplx_mhr_options and smplx_mhr_options.enabled:
        required.append('mhrfromsmplx')
    if mamma_confidence_fit and smplx_source == 'mamma':
        required.append('smplx_confidence')
    validate_fit_metrics(report, required)
    if mamma_confidence_fit and smplx_source == 'mamma':
        from capture_processing.utils.mamma_additional_fit import validate_confidence_metrics
        validate_confidence_metrics(target, report)
    from dataclasses import replace
    from capture_processing.utils.semantic_parsing import SemanticOptions, ensure_semantic_parsing
    ensure_semantic_parsing(target, replace(semantic_options or SemanticOptions(), device=device))
    return target


def _copy_if_changed(source, destination):
    """Reuse prior copied media; adding semantic labels must not recopy the dataset."""
    source, destination = Path(source), Path(destination)
    if destination.is_file():
        old, new = destination.stat(), source.stat()
        if old.st_size == new.st_size and old.st_mtime_ns == new.st_mtime_ns:
            return str(destination)
    return shutil.copy2(source, destination)
