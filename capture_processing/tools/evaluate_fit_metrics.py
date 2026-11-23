"""Compute missing metrics for cached subjects without rerunning their fits."""
import argparse
from pathlib import Path
from capture_processing.utils.fit_metrics import FIT_FILES, write_fit_metrics, validate_fit_metrics
from capture_processing.utils.silhouette_metrics import SilhouetteOptions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subjects = parser.add_mutually_exclusive_group(required=True)
    subjects.add_argument('--subject-root', type=Path)
    subjects.add_argument('--subjects-root', type=Path, help='Process every direct subject folder with saved fits.')
    parser.add_argument('--smplx-python', help='SMPL-X mesh reconstruction interpreter; auto-detected by default.')
    parser.add_argument('--mhr-python', help='SAM3D mesh reconstruction interpreter; auto-detected by default.')
    parser.add_argument('--official-python', help='Official MHR mesh reconstruction interpreter; auto-detected by default.')
    parser.add_argument('--smplx-model-path', type=Path)
    parser.add_argument('--official-repo-path', type=Path)
    parser.add_argument('--mhr-repo-path', type=Path)
    parser.add_argument('--cameras', nargs='+', help='Optional shared camera subset; default: all calibrated cameras.')
    parser.add_argument('--max-frames', type=int, help='Optional shared frame limit; default: all common frames.')
    parser.add_argument('--force', action='store_true', help='Recompute silhouette metrics even if cached.')
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    options = SilhouetteOptions(smplx_python=args.smplx_python, sam3d_python=args.mhr_python,
                                official_python=args.official_python, smplx_model=args.smplx_model_path,
                                official_repo=args.official_repo_path, sam3d_repo=args.mhr_repo_path,
                                cameras=tuple(args.cameras) if args.cameras else None,
                                max_frames=args.max_frames, force=args.force)
    roots = [args.subject_root] if args.subject_root else sorted(
        path for path in args.subjects_root.iterdir() if path.is_dir() and
        any((path / name).is_file() for name in FIT_FILES.values()))
    if not roots:
        parser.error('No subjects with saved fits found')
    failures = []
    for root in roots:
        try:
            required = [kind for kind, name in FIT_FILES.items() if (root / name).is_file()]
            if not required:
                raise RuntimeError(f'No saved fits in {root}')
            report = write_fit_metrics(root, options)
            validate_fit_metrics(report, required)
        except (RuntimeError, OSError, ValueError) as exc:
            failures.append(str(exc))
    for failure in failures:
        print(f'ERROR: {failure}')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
