"""Entrypoint for VCI and THuman dataset processing."""

from __future__ import annotations

import sys

from capture_processing.cli import main
from capture_processing.utils.subject_lock import SUBJECT_BUSY_EXIT, SubjectBusyError


def entrypoint():
    try:
        main()
    except SubjectBusyError as exc:
        print(f"[busy] {exc}", file=sys.stderr)
        return SUBJECT_BUSY_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
