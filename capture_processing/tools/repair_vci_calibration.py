"""Repair the original CVPR 036–048 calibration exports with backups."""
import argparse
import json
from pathlib import Path
from capture_processing.utils.calibration_repair import repair_calibrations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True,
                        help='CVPR dataset root containing the 2026_* batch directories.')
    parser.add_argument('--reference', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='Write repairs; otherwise preflight only.')
    mode.add_argument('--stage-dir', type=Path, help='Write reviewable corrected copies without touching originals.')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    reference = args.reference or args.source_root / '2026_09_17/2026_09_17_cvpr-dataset_035/calibration_dome.json'
    result = repair_calibrations(args.source_root, reference, apply=args.apply, stage_dir=args.stage_dir)
    report = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.write_text(report)
    print(report, end='')


if __name__ == '__main__':
    main()
