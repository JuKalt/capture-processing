"""Prepare shared MHR runtime caches before parallel GPU workers start."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import warnings
from pathlib import Path

from capture_processing.tools.ensure_hf_login import main as ensure_hf_login
from capture_processing.tools.run_mhr import _install_simplecv_triangulate_compat

MODEL_ID = "facebook/sam-3d-body-dinov3"


def cached_model_paths():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        directory = Path(snapshot_download(repo_id=MODEL_ID, local_files_only=True))
    except LocalEntryNotFoundError:
        return None
    checkpoint, model = directory / "model.ckpt", directory / "assets/mhr_model.pt"
    paths = (checkpoint, model, directory / "model_config.yaml")
    if all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return checkpoint, model
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Only check Hugging Face authentication; do not prewarm model caches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_src = args.repo_path.expanduser().resolve() / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", message="Momentum is not enabled")

    if args.skip_model_load:
        ensure_hf_login()
        return

    paths = cached_model_paths()
    if paths is None:
        ensure_hf_login()

    # MHRHead selects CUDA during construction even when the outer loader is
    # passed device="cpu". Hide GPUs before importing the model so cache
    # preparation cannot compete with the subject workers for VRAM. This helper
    # runs in its own process; inference workers retain their assigned GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    _install_simplecv_triangulate_compat()

    from sam3d_body.build_models import _hf_download, load_sam_3d_body

    log_path = Path.cwd() / "mhr_runtime_prepare.log"
    print(f"Preparing SAM3D Body model cache before parallel MHR workers... (details: {log_path})")
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                ckpt_path, mhr_path = paths or _hf_download(MODEL_ID)
                load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path, device="cpu")
    except Exception as exc:
        text = str(exc)
        if "dinov3" in text.lower():
            raise SystemExit(
                "Failed while preparing the shared Dinov3 Torch Hub cache. "
                "A previous parallel run may have left a partial cache. "
                "Remove the Dinov3 entries under ~/.cache/torch/hub/ and rerun. "
                f"Original error: {exc}"
            ) from exc
        raise
    print("SAM3D Body model cache is ready.")


if __name__ == "__main__":
    main()
