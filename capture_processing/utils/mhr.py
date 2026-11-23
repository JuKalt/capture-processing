"""SAM3D Body / MHR integration helpers."""

from __future__ import annotations

from capture_processing.utils.subject_lock import subject_locked

import os
import shlex
import subprocess
import json
from pathlib import Path

# Pipeline defaults shared by estimation, rendering and quick-resume checks.
DEFAULT_MHR_ITERATIONS = 100
VISUALIZATION_SETTINGS = dict(cameras=None, max_frames=120, frame_step=1, fps=30.,
                              long_edge=1024, reference_camera=None)


def default_sam3d_body_rerun_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "sam3d-body-rerun"


def resolve_sam3d_body_rerun_repo(path: Path | None) -> Path:
    raw = path or os.environ.get("SAM3D_BODY_RERUN_REPO")
    repo = Path(raw).expanduser().resolve() if raw else default_sam3d_body_rerun_repo()
    if not (repo / "src" / "sam3d_body").is_dir():
        raise SystemExit(
            f"sam3d-body-rerun checkout is not initialized at {repo}. "
            "Run scripts/setup/setup_sam3d_body_rerun_env.sh."
        )
    return repo


def expected_mhr_params_path(subject_root: Path) -> Path:
    return subject_root / "mhr_params.npz"


def is_mhr_params_complete(path: Path, signature: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        import numpy as np

        with np.load(path) as data:
            if not bool(data["complete"].item()):
                return False
            ids = data["frame_ids"]
            if ids.ndim != 1 or not len(ids) or len(set(ids.tolist())) != len(ids):
                return False
            shapes = {"num_views": (), "global_rot": (3,), "body_pose": (133,),
                      "translation": (3,), "shape": (45,), "scale": (28,),
                      "hand_pose": (108,), "expression": (72,),
                      "triangulated_keypoints_3d": (70, 4), "loss_total": ()}
            if any(data[key].shape != (len(ids), *shape) for key, shape in shapes.items()):
                return False
            if int(data["checkpointed_frames"].item()) != len(ids):
                return False
            return signature is None or str(data["estimation_signature"].item()) == signature
    except Exception:
        return False


def estimation_signature(subject_root: Path, repo: Path, max_frames: int | None, iterations: int) -> str:
    from capture_processing.utils.stage_cache import subject_signature, code_stamp
    from capture_processing.utils.silhouette_metrics import _repo_stamp, _sam_asset_stamps
    return subject_signature(subject_root, dict(max_frames=max_frames, iterations=iterations),
                             fit_files=[], dependencies=[_repo_stamp(repo), _sam_asset_stamps(),
                             code_stamp('capture_processing/tools/run_mhr.py')])


def prepare_mhr_runtime(python_executable: str, repo: Path) -> None:
    """Prompt for auth and prewarm shared model caches before MHR workers run."""
    python_cmd = shlex.split(python_executable)
    if not python_cmd:
        raise SystemExit("--mhr-python must not be empty")

    project_root = Path(__file__).resolve().parents[2]
    cmd = [
        *python_cmd,
        "-m",
        "capture_processing.tools.prepare_mhr_runtime",
        "--repo-path",
        str(repo),
    ]
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "MHR Python command was not found: "
            f"{shlex.join(python_cmd)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"MHR runtime preparation failed (exit code {exc.returncode}).") from exc


def subject_needs_mhr(subject_root: Path, force: bool) -> bool:
    return force or not is_mhr_params_complete(expected_mhr_params_path(subject_root))


def any_subject_needs_mhr(subject_roots: list[Path], force: bool) -> bool:
    return any(subject_needs_mhr(subject_root, force) for subject_root in subject_roots)


def run_mhr_visualization(subject_root: Path, repo: Path, python_executable: str,
                          device: str | None, *, kind: str = 'mhr', force: bool = False) -> Path:
    """Render cached MHR parameters in the same environment as estimation."""
    from capture_processing.utils.stage_cache import rendering_signature, rendering_complete
    folder = 'mhr_visualization' if kind == 'mhr' else 'mhr_from_smplx_visualization'
    output = subject_root / folder
    signature = rendering_signature(subject_root, repo, kind, dict(VISUALIZATION_SETTINGS))
    if not force and rendering_complete(output, signature):
        print(f'Using existing complete visualization: {output}', flush=True)
        return output
    python_cmd = shlex.split(python_executable)
    if not python_cmd:
        raise SystemExit("--mhr-python must not be empty")
    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    _apply_device_env(env, device)
    env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [*python_cmd, '-m', 'capture_processing.tools.visualize_mhr', '--subject-root', str(subject_root),
           '--repo-path', str(repo), '--kind', kind,
           '--max-frames', str(VISUALIZATION_SETTINGS['max_frames']),
           '--frame-step', str(VISUALIZATION_SETTINGS['frame_step']),
           '--fps', str(VISUALIZATION_SETTINGS['fps']),
           '--long-edge', str(VISUALIZATION_SETTINGS['long_edge'])]
    if force:
        cmd += ['--force']
    if device == 'cpu':
        cmd += ['--device', 'cpu']
    subprocess.run(cmd, cwd=project_root, env=env, check=True)
    return output


def maybe_create_mhr_params(
    subject_root: Path,
    *,
    source: str,
    repo_path: Path | None,
    python_executable: str,
    device: str | None,
    max_frames: int | None,
    iterations: int,
    force: bool,
    workers: int = 1,
) -> None:
    if source == "none":
        return
    if source != "sam3d-body-rerun":
        raise SystemExit(f"Unsupported MHR source: {source}")

    if not (subject_root / "images").is_dir() or not (subject_root / "masks").is_dir():
        raise SystemExit(
            f"MHR parameterization needs images and masks under {subject_root}. "
            "Run without --skip-images first."
        )
    if not (subject_root / "calibration.json").is_file():
        raise SystemExit(f"MHR parameterization needs calibration.json at {subject_root}")

    repo = resolve_sam3d_body_rerun_repo(repo_path)
    params_path = run_mhr_parameterization(
        subject_root=subject_root,
        repo=repo,
        python_executable=python_executable,
        device=device,
        max_frames=max_frames,
        iterations=iterations,
        force=force,
        workers=workers,
    )
    print(f"  MHR parameters ready: {params_path}")


def _apply_device_env(env: dict[str, str], device: str | None) -> None:
    if device is None:
        return
    raw = device.strip().lower()
    if not raw:
        return
    if raw == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        return
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1]
    elif raw == "cuda":
        raw = "0"
    if not raw.isdigit():
        raise SystemExit(
            "--device must be cpu, cuda, cuda:<index>, or a numeric GPU index "
            f"(got: {device!r})"
        )
    env["CUDA_VISIBLE_DEVICES"] = raw


def _mhr_frame_ids(subject_root: Path) -> list[int]:
    calibration = json.loads((subject_root / "calibration.json").read_text())
    cam_names = sorted(calibration)
    if not cam_names:
        return []
    first = subject_root / "images" / cam_names[0]
    ids: list[int] = []
    for path in sorted(first.glob("color_*.jpg")):
        try:
            ids.append(int(path.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if ids:
        return ids
    for path in sorted(first.glob("*.jpg")):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return ids


def _mhr_shard_path(output_path: Path, index: int) -> Path:
    return output_path.with_name(f"{output_path.stem}.worker{index:02d}{output_path.suffix}")


def _merge_mhr_shards(output_path: Path, shard_paths: list[Path], frame_ids: list[int]) -> None:
    import numpy as np

    if not shard_paths:
        raise SystemExit("No MHR worker shards to merge.")
    loaded = []
    try:
        for path in shard_paths:
            if not is_mhr_params_complete(path):
                raise SystemExit(f"MHR worker shard is incomplete: {path}")
            loaded.append(np.load(path, allow_pickle=False))

        first = loaded[0]
        cam_names = first["cam_names"]
        faces = first["faces"]
        n_frames = len(frame_ids)
        frame_to_index = {frame_id: idx for idx, frame_id in enumerate(frame_ids)}
        arrays = {
            "reference_cam_names": np.full(n_frames, "", dtype=f"U{max(map(len, cam_names), default=1)}"),
            "num_views": np.zeros((n_frames,), dtype=np.int32),
            "global_rot": np.full((n_frames, 3), np.nan, dtype=np.float32),
            "body_pose": np.full((n_frames, 133), np.nan, dtype=np.float32),
            "translation": np.full((n_frames, 3), np.nan, dtype=np.float32),
            "shape": np.full((n_frames, 45), np.nan, dtype=np.float32),
            "scale": np.full((n_frames, 28), np.nan, dtype=np.float32),
            "hand_pose": np.full((n_frames, 108), np.nan, dtype=np.float32),
            "expression": np.full((n_frames, 72), np.nan, dtype=np.float32),
            "triangulated_keypoints_3d": np.full((n_frames, 70, 4), np.nan, dtype=np.float32),
            "loss_total": np.full((n_frames,), np.nan, dtype=np.float32),
        }
        for key in ("reprojection_error_px", "triangulated_keypoint_error_mm"):
            if all(key in shard for shard in loaded):
                arrays[key] = np.full((n_frames,), np.nan, dtype=np.float32)
        seen = np.zeros((n_frames,), dtype=np.int32)
        signature = str(first["estimation_signature"].item()) if "estimation_signature" in first else ""
        for shard in loaded:
            shard_signature = str(shard["estimation_signature"].item()) if "estimation_signature" in shard else ""
            if shard_signature != signature:
                raise SystemExit("MHR worker shards have different estimation signatures.")
            shard_frame_ids = [int(item) for item in shard["frame_ids"]]
            for shard_idx, frame_id in enumerate(shard_frame_ids):
                if frame_id not in frame_to_index:
                    raise SystemExit(f"MHR worker shard contains unexpected frame {frame_id}")
                out_idx = frame_to_index[frame_id]
                seen[out_idx] += 1
                for key, target in arrays.items():
                    if key == "reference_cam_names" and key not in shard:
                        continue
                    target[out_idx] = shard[key][shard_idx]
        missing = [frame_ids[index] for index, count in enumerate(seen) if count != 1]
        if missing:
            raise SystemExit(f"MHR worker shards did not cover every frame exactly once: {missing[:10]}")

        tmp_path = output_path.with_name(output_path.name + ".tmp.npz")
        np.savez_compressed(
            tmp_path,
            frame_ids=np.asarray(frame_ids, dtype=np.int32),
            cam_names=cam_names,
            faces=faces,
            checkpointed_frames=np.asarray(n_frames, dtype=np.int32),
            complete=np.asarray(True, dtype=np.bool_),
            estimation_signature=np.asarray(signature),
            **arrays,
        )
        tmp_path.replace(output_path)
    finally:
        for shard in loaded:
            shard.close()


@subject_locked
def run_mhr_parameterization(
    *,
    subject_root: Path,
    repo: Path,
    python_executable: str,
    device: str | None,
    max_frames: int | None,
    iterations: int,
    force: bool,
    workers: int = 1,
) -> Path:
    """Reuse or estimate parameters, then render if the visualization is stale.

    Rendering failures leave parameters available for a cached retry.
    """
    output_path = expected_mhr_params_path(subject_root)
    signature = estimation_signature(subject_root, repo, max_frames, iterations)
    if is_mhr_params_complete(output_path, signature) and not force:
        print(f"Using existing complete MHR parameterization: {output_path}")
        run_mhr_visualization(subject_root, repo, python_executable, device)
        return output_path
    if output_path.is_file() and not force:
        print(f"Existing MHR checkpoint is incomplete; rerunning: {output_path}")

    python_cmd = shlex.split(python_executable)
    if not python_cmd:
        raise SystemExit("--mhr-python must not be empty")

    if workers <= 0:
        raise SystemExit("--mhr-workers must be a positive integer")

    project_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["VCI_MHR_ESTIMATION_SIGNATURE"] = signature
    _apply_device_env(env, device)
    env["PYTHONPATH"] = str(project_root) + os.pathsep + str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")

    if device:
        visible_devices = env.get("CUDA_VISIBLE_DEVICES", "")
        print(f"  MHR device: {device} (CUDA_VISIBLE_DEVICES={visible_devices!r})")

    base_cmd = [
        *python_cmd,
        "-m",
        "capture_processing.tools.run_mhr",
        "--subject-root",
        str(subject_root),
        "--repo-path",
        str(repo),
        "--iterations",
        str(iterations),
    ]
    if max_frames is not None:
        base_cmd += ["--max-frames", str(max_frames)]
    if force:
        base_cmd += ["--force"]

    frame_ids = _mhr_frame_ids(subject_root)
    if max_frames is not None:
        frame_ids = frame_ids[:max_frames]
    active_workers = min(workers, len(frame_ids)) if frame_ids else 1

    if active_workers > 1:
        shard_paths = [_mhr_shard_path(output_path, index) for index in range(active_workers)]
        print(f"Running SAM3D Body multiview MHR parameterization with {active_workers} worker(s)...")
        procs = []
        for index, shard_path in enumerate(shard_paths):
            cmd = [
                *base_cmd,
                "--output",
                str(shard_path),
                "--shard-index",
                str(index),
                "--shard-count",
                str(active_workers),
            ]
            procs.append(subprocess.Popen(cmd, cwd=project_root, env=env))
        failures = []
        for index, proc in enumerate(procs):
            rc = proc.wait()
            if rc != 0:
                failures.append(f"worker {index}: exit code {rc}")
        if failures:
            raise SystemExit("MHR parameterization failed:\n" + "\n".join(f"  - {item}" for item in failures))
        _merge_mhr_shards(output_path, shard_paths, frame_ids)
    else:
        cmd = [*base_cmd, "--output", str(output_path)]
        print("Running SAM3D Body multiview MHR parameterization...")
        subprocess.run(cmd, cwd=project_root, env=env, check=True)

    if not is_mhr_params_complete(output_path, signature):
        raise SystemExit(f"MHR parameterization did not write a complete matching output: {output_path}")
    run_mhr_visualization(subject_root, repo, python_executable, device, force=force)
    return output_path
