#!/usr/bin/env python3
"""Extract DanceNet3D multiview data and generate canonical SMPL-X parameters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture_processing.utils.smplx_mhr import add_conversion_arguments
from capture_processing.core.dancenet3d_pipeline import process_dancenet3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract DanceNet3D multiview videos, build the unified zero-copy "
            "layout, and generate smpl_params.npz with MAMMA."
        )
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--session", choices=("s4", "s5", "s6"), required=True)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--extracted-root", type=Path)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-smplx", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mamma-repo-path", type=Path)
    parser.add_argument("--mamma-preset", type=Path)
    parser.add_argument("--mamma-out-tag", default="dancenet3d")
    parser.add_argument("--mamma-python", default="conda run -n mamma python")
    parser.add_argument("--mamma-workers", type=int, default=4)
    parser.add_argument("--mamma-batch-size", type=int, default=8)
    parser.add_argument("--mamma-force", action="store_true")
    add_conversion_arguments(parser)
    args = parser.parse_args()
    if args.parallel <= 0:
        parser.error("--parallel must be positive")
    return args


def main() -> int:
    subjects = process_dancenet3d(parse_args())
    print("\nDanceNet3D processing complete:")
    for subject in subjects:
        state = "with SMPL-X" if (subject / "smpl_params.npz").is_file() else "without SMPL-X"
        print(f"  - {subject} ({state})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
