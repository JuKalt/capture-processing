"""Add semantic labels to an old processed subject without calling any earlier stage."""
import argparse
from pathlib import Path
from capture_processing.utils.semantic_parsing import add_arguments, ensure_semantic_parsing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, required=True)
    parser.add_argument('--device')
    add_arguments(parser)
    args = parser.parse_args()
    from capture_processing.utils.semantic_parsing import options_from_args
    if not options_from_args(args).enabled:
        parser.error('--no-semantic-parsing leaves nothing to backfill')
    ensure_semantic_parsing(args.subject_root, args)
    from capture_processing.utils.subject_manifest import record_backfill
    record_backfill(args.subject_root, ['semantic_parsing'], 'parse_subject_semantics')


if __name__ == '__main__':
    main()
