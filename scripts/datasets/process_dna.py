#!/usr/bin/env python3
"""Export DNA-Rendering SMC data and generate new SMPL-X fits with MAMMA."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from capture_processing.utils.smplx_mhr import add_conversion_arguments
from capture_processing.core.dna_pipeline import process_dna


def default_mamma_python():
    # A direct interpreter avoids nesting conda run around MAMMA's own
    # per-step conda launches. Environments may live beside this interpreter.
    prefix = Path(sys.prefix)
    environments = prefix.parent if prefix.parent.name == "envs" else prefix / "envs"
    python = environments / "mamma" / "bin" / "python"
    if python.is_file():
        import shlex
        return shlex.quote(str(python))
    return "conda run -n mamma python"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--cameras", nargs="+", type=int, default=list(range(48)), help="Source 5MP camera IDs (0–47); default: all")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stop", type=int, help="Exclusive source frame stop; default: all frames")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024, help="Maximum output dimension; 0 keeps source resolution")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mamma-repo-path", type=Path)
    parser.add_argument("--mamma-preset", type=Path)
    parser.add_argument("--mamma-out-tag", default="dna")
    parser.add_argument("--mamma-python", default=default_mamma_python(), help="Fitting interpreter; prefers a sibling mamma environment")
    parser.add_argument("--mamma-workers", type=int, default=4)
    parser.add_argument("--mamma-batch-size", type=int, default=8)
    parser.add_argument("--mamma-force", action="store_true")
    add_conversion_arguments(parser)
    args = parser.parse_args()
    if args.frame_start < 0 or args.frame_step < 1 or args.resolution < 0 or args.mamma_workers < 1 or args.mamma_batch_size < 1:
        parser.error("Frame start/resolution must be nonnegative; frame step/workers/batch size must be positive")
    if len(set(args.cameras)) != len(args.cameras) or any(not 0 <= c < 48 for c in args.cameras):
        parser.error("Cameras must be unique IDs between 0 and 47")
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("Subjects must be unique")
    return args


if __name__ == "__main__":
    process_dna(parse_args())
