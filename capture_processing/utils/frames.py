"""Frame discovery and organization utilities."""

from __future__ import annotations

from pathlib import Path


def discover_frames(anchor: Path) -> list[tuple[int, Path]]:
    """Return sorted ``(frame_id, frame_dir)`` pairs."""
    frames: list[tuple[int, Path]] = []
    seen = set()
    for directory in sorted(anchor.iterdir()):
        if not directory.is_dir() or not directory.name.startswith("frame_"):
            continue
        try:
            frame_id = int(directory.name.removeprefix("frame_"))
        except ValueError as exc:
            raise ValueError(f"Invalid frame directory: {directory}") from exc
        if frame_id < 0 or frame_id in seen:
            raise ValueError(f"Negative or duplicate frame ID: {directory}")
        seen.add(frame_id)
        frames.append((frame_id, directory))
    return sorted(frames)
