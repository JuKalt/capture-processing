"""Add the confidence-filtered fit to an already processed MAMMA subject."""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
from capture_processing.utils.mamma_additional_fit import ensure_confidence_fit, validate_confidence_metrics
from capture_processing.utils.fit_metrics import write_fit_metrics
from capture_processing.utils.silhouette_metrics import SilhouetteOptions, PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, required=True)
    parser.add_argument('--mamma-python', default='conda run -n mamma python')
    parser.add_argument('--mamma-repo-path', type=Path)
    parser.add_argument('--mamma-out-tag', default='vci')
    parser.add_argument('--device')
    parser.add_argument('--force', action='store_true', help='Recompute only the additional fit.')
    parser.add_argument('--mhr-python')
    parser.add_argument('--official-python')
    parser.add_argument('--metrics-max-frames', type=int, help='Limit the common metric evaluation subset; default: all frames.')
    parser.add_argument('--metrics-cameras', nargs='+', help='Common metric camera subset; default: all cameras.')
    args = parser.parse_args()
    if args.metrics_max_frames is not None and args.metrics_max_frames < 1:
        parser.error('--metrics-max-frames must be positive')
    subject = args.subject_root.expanduser().resolve()
    fit_args = SimpleNamespace(smplx_source='mamma', mamma_confidence_fit=True,
        mamma_repo_path=args.mamma_repo_path, mamma_out_tag=args.mamma_out_tag,
        mamma_python=args.mamma_python, mamma_force=args.force, device=args.device)
    ensure_confidence_fit(subject, fit_args)
    official = PROJECT / 'external/MHR/.pixi/envs/default/bin/python'
    sam = PROJECT / 'external/sam3d-body-rerun/.pixi/envs/dev/bin/python'
    report = write_fit_metrics(subject, SilhouetteOptions(smplx_python=args.mamma_python,
        official_python=args.official_python or (str(official) if official.is_file() else None),
        sam3d_python=args.mhr_python or (str(sam) if sam.is_file() else None),
        cameras=tuple(args.metrics_cameras) if args.metrics_cameras else None,
        max_frames=args.metrics_max_frames))
    validate_confidence_metrics(subject, report)
    steps = ['smplx_confidence']
    if args.metrics_max_frames is None and args.metrics_cameras is None:
        steps.append('fit_metrics')  # Subset metrics are not the pipeline's validated report.
    from capture_processing.utils.subject_manifest import record_backfill
    record_backfill(subject, steps, 'fit_mamma_confidence')


if __name__ == '__main__':
    main()
