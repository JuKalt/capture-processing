"""Path handling helpers."""

from __future__ import annotations

from pathlib import Path


def paths_overlap(left: Path, right: Path) -> bool:
    """Return True when either path contains the other."""
    left = left.resolve()
    right = right.resolve()
    if left == right:
        return True
    try:
        right.relative_to(left)
        return True
    except ValueError:
        pass
    try:
        left.relative_to(right)
        return True
    except ValueError:
        return False
