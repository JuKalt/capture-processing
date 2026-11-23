"""MAMMA integration for creating SMPL-X estimates from multi-view images."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import zipfile
from pathlib import Path

from tqdm import tqdm

MAMMA_STEPS = ("ma_cap", "ma_masks", "ma_2d", "ma_3d", "ma_vis")
MAMMA_STATUS_RE = re.compile(r"status:\s+(?P<step>ma_\w+)\[[^\]]+\]\s+->\s+(?P<status>\w+)")


def default_mamma_repo() -> Path:
    """Return the default local MAMMA checkout path."""
    return Path(__file__).resolve().parents[2] / "external" / "mamma"


def resolve_mamma_repo(path: Path | None) -> Path:
    """Resolve an explicit path, MAMMA_REPO_PATH, or the default checkout."""
    raw = path or os.environ.get("MAMMA_REPO_PATH")
    repo = Path(raw).expanduser().resolve() if raw else default_mamma_repo()
    if not (repo / "inference" / "cli" / "run.py").is_file():
        raise SystemExit(
            f"MAMMA checkout is not initialized at {repo}. "
            "Run scripts/setup/setup_mamma_env.sh to clone the pinned revision and install its environment."
        )
    return repo


def resolve_mamma_preset(repo: Path, preset: Path | None) -> Path:
    """Resolve a preset path, defaulting to MAMMA's full preset."""
    if preset is None:
        preset = Path("configs/examples/presets/full.yaml")
    preset = preset.expanduser()
    if not preset.is_absolute():
        preset = repo / preset
    preset = preset.resolve()
    if not preset.is_file():
        raise SystemExit(f"MAMMA preset not found: {preset}")
    return preset



def _preset_block(lines: list[str], name: str) -> tuple[int, int] | None:
    start = next((i for i, line in enumerate(lines) if line == f"{name}:"), None)
    if start is None:
        return None
    end = next(
        (
            i
            for i in range(start + 1, len(lines))
            if lines[i] and not lines[i].startswith(" ")
        ),
        len(lines),
    )
    return start, end


def prepare_mamma_runtime_preset(
    preset: Path,
    work_dir: Path,
    workers: int = 4,
    masks_root: Path | None = None,
    batch_size: int = 8,
) -> Path:
    """Use the tracked batched landmark adapter and disable camera previews.

    Custom scripts retain their own implementation and flags.
    """
    if workers <= 0 or batch_size <= 0:
        raise SystemExit("MAMMA workers and batch size must be positive integers")
    lines = preset.read_text().splitlines()
    block = _preset_block(lines, "ma_2d")
    if block is None:
        return preset
    start, end = block
    script_line = next((i for i in range(start + 1, end)
                        if re.match(r"^\s+script:", lines[i])), None)
    adapter = Path(__file__).with_name("mamma_landmarks.py").resolve()
    standard_script = script_line is not None and lines[script_line].split(":", 1)[1].strip().strip("\"'") in {"run_ma_2d.py", str(adapter)}
    changed = False
    if standard_script:
        # JSON quoting is also valid YAML and protects paths containing spaces.
        import json
        lines[script_line] = f"  script: {json.dumps(str(adapter))}"
        changed = True
    desired_flags = ["--no-save_cam_output"]
    if standard_script:
        desired_flags += [f"--workers {workers}", f"--batch-size {batch_size}"]
    flags_line = next((i for i in range(start + 1, end)
                       if re.match(r"^\s+flags:", lines[i])), None)
    if flags_line is None:
        lines[end:end] = ["  flags:", *[f"  - {flag}" for flag in desired_flags]]
        changed = True
    else:
        value = lines[flags_line].split(":", 1)[1].strip()
        if value == "[]":
            lines[flags_line:flags_line + 1] = ["  flags:", *[f"  - {flag}" for flag in desired_flags]]
            changed = True
        elif not value:
            # Append options last: argparse uses the final occurrence, so CLI
            # settings override preset worker/batch counts and preview flags.
            flag_end = flags_line + 1
            while flag_end < end and re.match(r"^\s+- ", lines[flag_end]):
                flag_end += 1
            lines[flag_end:flag_end] = [f"  - {flag}" for flag in desired_flags]
            changed = True
        else:
            raise SystemExit("MAMMA runtime preset requires flags: [] or a YAML block list")

    if masks_root is not None:
        mask_block = _preset_block(lines, "ma_masks")
        if mask_block is None:
            raise SystemExit("MAMMA preset has no ma_masks block to disable.")
        mask_start, mask_end = mask_block
        enabled_line = next(
            (
                i
                for i in range(mask_start + 1, mask_end)
                if re.match(r"^\s+enabled:", lines[i])
            ),
            None,
        )
        if enabled_line is None:
            lines.insert(mask_start + 1, "  enabled: false")
            changed = True
        elif lines[enabled_line].strip() != "enabled: false":
            indent = lines[enabled_line][
                : len(lines[enabled_line]) - len(lines[enabled_line].lstrip())
            ]
            lines[enabled_line] = f"{indent}enabled: false"
            changed = True

        ma2d_block = _preset_block(lines, "ma_2d")
        assert ma2d_block is not None
        start, end = ma2d_block
        masks_line = next(
            (
                i
                for i in range(start + 1, end)
                if re.match(r"^\s+ma_masks_dir:", lines[i])
            ),
            None,
        )
        desired = f"  ma_masks_dir: {masks_root.resolve()}"
        if masks_line is None:
            lines.insert(start + 1, desired)
            changed = True
        elif lines[masks_line] != desired:
            lines[masks_line] = desired
            changed = True

    if not changed:
        return preset

    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_preset = work_dir / "mamma_preset_runtime.yaml"
    runtime_preset.write_text("\n".join(lines) + "\n")
    return runtime_preset


def expected_mamma_params_path(
    output_root: Path,
    tag: str,
    footage_root: Path,
    seq_name: str,
    body_id: int = 0,
) -> Path:
    """Return the ma_3d params path produced by `python -m inference run`."""
    return (
        output_root
        / "ma_3d"
        / tag
        / footage_root.name
        / seq_name
        / f"smplx_params_body_id-{body_id:02d}.npz"
    )


def _mamma_subprocess_env() -> dict[str, str]:
    """Start fitting without the caller's Conda activation stack.

    MAMMA activates its own environment for each step. Inherited compiler
    hooks can otherwise try to deactivate the export environment against the
    wrong prefix. Keep Conda executable paths, model settings, and user paths.
    """
    env = os.environ.copy()
    for key in list(env):
        if (
            key == "CONDA_PREFIX"
            or key.startswith(("CONDA_PREFIX_", "CONDA_BACKUP_"))
            or key in ("CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER")
        ):
            env.pop(key)
    env["CONDA_SHLVL"] = "0"
    return env


def run_mamma(
    *,
    repo: Path,
    preset: Path,
    footage_root: Path,
    seq_name: str,
    calibration_path: Path,
    output_root: Path,
    tag: str,
    python_executable: str,
    force: bool,
    device: str | None,
) -> Path:
    """Run MAMMA and return its first-body SMPL-X params file."""
    output_root.mkdir(parents=True, exist_ok=True)
    python_cmd = shlex.split(python_executable)
    if not python_cmd:
        raise SystemExit("--mamma-python must not be empty")

    cmd = [
        *python_cmd,
        "-m",
        "inference",
        "run",
        "--cfg",
        str(preset),
        "--footage",
        str(footage_root),
        "--seq_name",
        seq_name,
        "--calib",
        str(calibration_path),
        "--out-tag",
        tag,
        "--out-dir",
        str(output_root),
        "-v",
    ]
    if force:
        cmd.append("--force")

    if find_existing_mamma_params(output_root, tag, footage_root, seq_name) is None:
        for step in ("ma_3d", "ma_vis"):
            done = output_root / step / tag / footage_root.name / seq_name / "DONE"
            done.unlink(missing_ok=True)

    env = _mamma_subprocess_env()
    _apply_mamma_device_env(env, device)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    if device:
        print(f"  MAMMA device: {device} (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')!r})")
    _check_mamma_python_dependencies(python_cmd, repo, env)
    _check_mamma_assets(repo, preset, env)
    enabled_steps = _enabled_mamma_steps(preset)
    _run_mamma_with_progress(
        cmd,
        repo,
        env,
        output_root,
        tag,
        footage_root.name,
        seq_name,
        enabled_steps,
    )

    params_path = expected_mamma_params_path(output_root, tag, footage_root, seq_name)
    if find_existing_mamma_params(output_root, tag, footage_root, seq_name) is None:
        raise SystemExit(f"MAMMA did not write valid complete SMPL-X parameters: {params_path}")
    return params_path


def _apply_mamma_device_env(env: dict[str, str], device: str | None) -> None:
    if device is None:
        return

    raw = device.strip().lower()
    if not raw:
        return
    if raw == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("EGL_DEVICE_ID", "0")
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
    env["EGL_DEVICE_ID"] = raw



def _run_mamma_with_progress(
    cmd: list[str],
    repo: Path,
    env: dict[str, str],
    output_root: Path,
    tag: str,
    dataset_name: str,
    seq_name: str,
    enabled_steps: tuple[str, ...] = MAMMA_STEPS,
) -> None:
    completed: set[str] = _existing_mamma_done_steps(
        output_root, tag, dataset_name, seq_name
    ).intersection(enabled_steps)
    failed_step: str | None = None
    output_tail: list[str] = []

    with subprocess.Popen(
        cmd,
        cwd=repo,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc, tqdm(
        total=len(enabled_steps),
        initial=len(completed),
        desc="MAMMA",
        unit="step",
        dynamic_ncols=True,
    ) as progress:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_tail.append(line.rstrip())
            output_tail = output_tail[-80:]
            _update_mamma_progress(
                line, progress, completed, enabled_steps=enabled_steps
            )
            match = MAMMA_STATUS_RE.search(line)
            if match and match.group("status") == "Failed":
                failed_step = match.group("step")

        return_code = proc.wait()

    if return_code:
        detail = "\n".join(output_tail)
        step_msg = f" during {failed_step}" if failed_step else ""
        raise SystemExit(
            f"MAMMA failed{step_msg} (exit code {return_code})."
            + (f"\nLast MAMMA output:\n{detail}" if detail else "")
        )


def _existing_mamma_done_steps(
    output_root: Path, tag: str, dataset_name: str, seq_name: str
) -> set[str]:
    done_steps: set[str] = set()
    for step in MAMMA_STEPS:
        done = output_root / step / tag / dataset_name / seq_name / "DONE"
        if done.is_file():
            done_steps.add(step)
    return done_steps


def mamma_steps_complete(output_root: Path, tag: str, dataset_name: str, seq_name: str, preset: Path) -> bool:
    return set(_enabled_mamma_steps(preset)) <= _existing_mamma_done_steps(
        output_root, tag, dataset_name, seq_name
    )


def _enabled_mamma_steps(preset: Path) -> tuple[str, ...]:
    lines = preset.read_text().splitlines()
    enabled = []
    for step in MAMMA_STEPS:
        block = _preset_block(lines, step)
        if block is None:
            continue
        start, end = block
        value = next(
            (
                line.split(":", 1)[1].strip().lower()
                for line in lines[start + 1 : end]
                if re.match(r"^\s+enabled:", line)
            ),
            "false",
        )
        if value == "true":
            enabled.append(step)
    return tuple(enabled)


def _update_mamma_progress(
    line: str,
    progress,
    completed: set[str],
    enabled_steps: tuple[str, ...] = MAMMA_STEPS,
) -> None:
    match = MAMMA_STATUS_RE.search(line)
    if not match:
        return

    step = match.group("step")
    status = match.group("status")
    if step not in enabled_steps:
        return

    progress.set_description(f"MAMMA {step}: {status}")
    if status == "Done" and step not in completed:
        completed.add(step)
        progress.update(1)


def _check_mamma_python_dependencies(
    python_cmd: list[str], repo: Path, env: dict[str, str]
) -> None:
    check_cmd = [*python_cmd, "-c", "import yaml"]
    try:
        subprocess.run(
            check_cmd,
            cwd=repo,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "MAMMA Python command was not found: "
            f"{shlex.join(python_cmd)}\n"
            "Install the MAMMA environment or pass --mamma-python with the "
            "Python command for that environment."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if (
            "EnvironmentLocationNotFound" in detail
            or "Could not find conda environment" in detail
        ):
            raise SystemExit(
                "MAMMA conda environment was not found.\n"
                f"  command: {shlex.join(python_cmd)}\n"
                "Install the MAMMA environment from external/mamma/docs/INSTALL.md, "
                "or pass --mamma-python with the Python command for the environment "
                "that has MAMMA installed."
                + (f"\n  error: {detail}" if detail else "")
            ) from exc
        raise SystemExit(
            "MAMMA Python is missing PyYAML, which is required to read "
            "MAMMA preset/config YAML files.\n"
            f"  command: {shlex.join(python_cmd)}\n"
            "Install pyyaml into the selected environment, install the full MAMMA "
            "environment from external/mamma/docs/INSTALL.md, or pass "
            "--mamma-python with the correct MAMMA Python command."
            + (f"\n  error: {detail}" if detail else "")
        ) from exc


def _check_mamma_assets(repo: Path, preset: Path, env: dict[str, str]) -> None:
    preset_text = preset.read_text()
    missing: list[tuple[str, Path]] = []
    seen_missing: set[Path] = set()

    def require_file(label: str, env_key: str, default: str) -> None:
        path = _resolve_mamma_asset_path(repo, env, env_key, default)
        if not path.is_file() and path not in seen_missing:
            missing.append((label, path))
            seen_missing.add(path)

    def require_dir(label: str, env_key: str, default: str) -> None:
        path = _resolve_mamma_asset_path(repo, env, env_key, default)
        if not path.is_dir() and path not in seen_missing:
            missing.append((label, path))
            seen_missing.add(path)

    if _preset_step_enabled(preset_text, "ma_masks"):
        require_file("YOLO12x detector", "MAMMA_YOLO_CHECKPOINT", "data/weights/yolo/yolo12x.pt")
        if _preset_ma_masks_uses_sam2(preset_text):
            require_file("SAM 2.1 checkpoint", "MAMMA_SAM2_CHECKPOINT", "data/weights/sam2/sam2.1_hiera_large.pt")

    if _preset_step_enabled(preset_text, "ma_2d"):
        ma2d_default = _preset_ma2d_weights(preset_text) or "data/weights/ma_2d/mamma_mask_full_cvpr.ckpt"
        require_file("MammaNet landmark checkpoint", "MAMMA_MA2D_CHECKPOINT", ma2d_default)
        require_file("Downsampled SMPL-X vertices", "MAMMA_DOWNSAMPLED_VERTS_PKL", "data/body_models/downsampled_verts/verts_512.pkl")

    if _preset_step_enabled(preset_text, "ma_3d"):
        require_dir("SMPL-X locked-head body model", "MAMMA_SMPLX_LOCKHEAD_MODELS", "data/body_models/smplx_locked_head")
        require_file("Downsampled SMPL-X vertices", "MAMMA_DOWNSAMPLED_VERTS_PKL", "data/body_models/downsampled_verts/verts_512.pkl")

    if not missing:
        return

    lines = ["MAMMA required asset files are missing; refusing to start the DAG.", "Missing:"]
    for label, path in missing:
        lines.append(f"  - {label}: {path}")
    lines.extend([
        "",
        "Install/download MAMMA assets, then rerun the converter.",
        "Useful commands from external/mamma:",
        "  bash data/download_mamma_weights.sh --all",
        "  bash data/download_smplx_locked_head.sh",
        "  wget -O data/weights/yolo/yolo12x.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x.pt",
        "  wget -O data/weights/sam2/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "",
        "If these files live elsewhere, set the corresponding MAMMA_* env vars",
        "or external/mamma/.env.local paths before running.",
    ])
    raise SystemExit("\n".join(lines))


def _resolve_mamma_asset_path(repo: Path, env: dict[str, str], env_key: str, default: str) -> Path:
    value = env.get(env_key) or _read_mamma_dotenv(repo).get(env_key) or default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo / path
    return path.resolve()


def _read_mamma_dotenv(repo: Path) -> dict[str, str]:
    dotenv_path = repo / ".env.local"
    values: dict[str, str] = {}
    if not dotenv_path.is_file():
        return values
    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip(chr(34)).strip(chr(39))
        values[key.strip()] = value
    return values


def _preset_step_enabled(preset_text: str, step: str) -> bool:
    match = re.search(rf"(?ms)^\s*{re.escape(step)}:\s*$([\s\S]*?)(?=^\S|\Z)", preset_text)
    if not match:
        return False
    enabled = re.search(r"(?m)^\s*enabled:\s*(\w+)", match.group(1))
    return enabled is None or enabled.group(1).lower() != "false"


def _preset_ma_masks_uses_sam2(preset_text: str) -> bool:
    match = re.search(r"(?ms)^\s*ma_masks:\s*$([\s\S]*?)(?=^\S|\Z)", preset_text)
    if not match:
        return False
    return bool(re.search(r"--sam_version(?:=|\s+)sam2\b", match.group(1)))


def _preset_ma2d_weights(preset_text: str) -> str | None:
    match = re.search(r"(?ms)^\s*ma_2d:\s*$([\s\S]*?)(?=^\S|\Z)", preset_text)
    if not match:
        return None
    weights = re.search(r"(?m)^\s*weights:\s*([^#\n]+)", match.group(1))
    if not weights:
        return None
    return weights.group(1).strip().strip(chr(34)).strip(chr(39)) or None


def find_existing_mamma_params(
    output_root: Path,
    tag: str,
    footage_root: Path,
    seq_name: str,
) -> Path | None:
    """Find an existing first-body MAMMA output, if present."""
    params_path = expected_mamma_params_path(output_root, tag, footage_root, seq_name)
    if not params_path.is_file():
        return None
    try:
        import numpy as np
        with np.load(params_path, allow_pickle=False) as data:
            pose = np.asarray(data["smplx_pose"])
            translation = np.asarray(data["smplx_translation"])
            betas = np.asarray(data["smplx_betas"])
            if (pose.ndim == 2 and pose.shape[0] > 0 and pose.shape[1] >= 165
                    and translation.shape == (pose.shape[0], 3) and betas.size
                    and all(np.isfinite(value).all() for value in (pose, translation, betas))):
                return params_path
    except (OSError, ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile):
        pass
    return None
