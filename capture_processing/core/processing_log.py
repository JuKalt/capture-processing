"""Metadata tracking for resumable dataset processing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

PROCESSING_LOG_FILE = ".processing_log.json"

CONVERSION_STEPS = {
    "calibration_loaded": "Source calibration file loaded",
    "cameras_mapped": "Camera ID mapping and filtering completed",
    "frames_discovered": "Frame directories discovered",
    "smplx_created": "SMPL-X parameters created",
    "images_transcoded": "Images transcoded to JPEG",
    "calibration_written": "THUman4 calibration file written",
    "metadata_written": "Metadata files written",
}


# New postprocessing stages must never invalidate legacy conversion markers.
PROCESSING_STEPS = {
    **CONVERSION_STEPS,
    "smplx_confidence": "Confidence-filtered SMPL-X fit and artifacts validated",
    "mhrfromsmplx": "Official SMPL-X to MHR conversion and visualization completed",
    "mhr": "SAM3D MHR fit and visualization completed",
    "fit_metrics": "Required shared-mask fit metrics validated",
    "semantic_parsing": "Detailed/general semantic labels and validation metrics completed",
}


def load_processing_log(out_root: Path) -> dict:
    """Load the processing log for this object, or create an empty one."""
    log_path = out_root / PROCESSING_LOG_FILE
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text())
            if isinstance(log, dict) and isinstance(log.get("steps_completed"), dict):
                return log
        except (OSError, ValueError):
            pass
    return {
        "started": datetime.now().isoformat(),
        "steps_completed": {},
        "errors": [],
    }


def save_processing_log(out_root: Path, log: dict) -> None:
    """Save the processing log."""
    log["last_updated"] = datetime.now().isoformat()
    path = out_root / PROCESSING_LOG_FILE
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(log, indent=2))
    temporary.replace(path)


def mark_step_complete(log: dict, step: str) -> None:
    """Mark a step as completed."""
    if step in PROCESSING_STEPS:
        log["steps_completed"][step] = datetime.now().isoformat()


def get_missing_steps(log: dict, *, semantic_enabled: bool = True) -> list[str]:
    """Get incomplete requested checklist entries in pipeline order.

    Artifact validation remains authoritative even when a marker is present.
    """
    completed = set(log["steps_completed"])
    requested = log.get("requested_steps", [*CONVERSION_STEPS, "semantic_parsing"])
    return [step for step in requested if step not in completed
            and (semantic_enabled or step != "semantic_parsing")]


def get_missing_conversion_steps(log: dict) -> list[str]:
    """Keep legacy conversion independently resumable as new stages are added."""
    return [step for step in CONVERSION_STEPS if step not in log["steps_completed"]]


def set_semantic_complete(out_root: Path, complete: bool) -> None:
    """Synchronize the checklist with validated semantic artifacts."""
    log = load_processing_log(out_root)
    present = "semantic_parsing" in log["steps_completed"]
    if complete and not present:
        mark_step_complete(log, "semantic_parsing")
    elif not complete and present:
        del log["steps_completed"]["semantic_parsing"]
    else:
        return
    save_processing_log(out_root, log)
