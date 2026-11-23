"""Validate saved preprocessing products without inference or output mutation."""
import argparse
import json
from pathlib import Path
from capture_processing.utils.subject_validation import validate_subject


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument('--subject-root', type=Path)
    roots.add_argument('--subjects-root', type=Path)
    parser.add_argument('--require', nargs='+', choices=('media','smplx','smplx_confidence','mhr',
                         'mhrfromsmplx','metrics','semantics','visualizations'),
                         default=['media','smplx','metrics'])
    parser.add_argument('--deep', action='store_true', help='Decode media, verify saved media hashes and semantic contents.')
    args = parser.parse_args(argv)
    subjects = [args.subject_root] if args.subject_root else sorted(
        path for path in args.subjects_root.iterdir() if path.is_dir() and (path/'images').is_dir())
    if not subjects:
        parser.error('No processed subjects found')
    reports = [validate_subject(path, args.require, deep=args.deep) for path in subjects]
    print(json.dumps(reports, indent=2))
    return 0 if all(report['complete'] for report in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
