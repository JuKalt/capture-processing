"""Camera calibration, mapping, and filtering utilities."""

from __future__ import annotations

import json
from pathlib import Path


def load_source_calibration(path: Path) -> dict[str, dict]:
    """Read the dome calibration JSON and normalize it by source camera ID."""
    import numpy as np

    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Source calibration must be a JSON object")
    from .calibration_repair import validate_source_calibration
    validate_source_calibration(data)
    from .preprocessing_validation import validate_cameras
    validate_cameras(data)
    cams: dict[str, dict] = {}
    for camera in data["cameras"]:
        cid = camera["camera_id"].upper()
        K = np.array(camera["intrinsics"]["camera_matrix"], dtype=np.float64).reshape(
            3, 3
        )
        vm = np.array(camera["extrinsics"]["view_matrix"], dtype=np.float64).reshape(
            4, 4
        )
        resolution = camera["intrinsics"]["resolution"]
        cams[cid] = {
            "K": K,
            "R": vm[:3, :3],
            "T": vm[:3, 3],
            "W": int(resolution[0]),
            "H": int(resolution[1]),
        }
    return cams


def build_camera_id_mapping(source_cam_ids: list[str]) -> dict[str, str]:
    """Assign sequential THUman4 camera IDs to sorted source IDs."""
    sorted_ids = sorted(source_cam_ids)
    width = max(2, len(str(max(len(sorted_ids) - 1, 0))))
    return {f"cam{i:0{width}d}": cid for i, cid in enumerate(sorted_ids)}


def read_camera_list_arg(values: list[str] | None) -> list[str]:
    """Read inline camera IDs or a file containing one ID per line."""
    if not values:
        return []
    if len(values) == 1:
        maybe_path = Path(values[0]).expanduser()
        if maybe_path.is_file():
            tokens: list[str] = []
            for line in maybe_path.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    tokens.append(line)
            return tokens
    return list(values)


def resolve_drop_tokens(tokens: list[str], valid_src_ids: set[str]) -> set[str]:
    """Validate and normalize dropped source IDs."""
    resolved: set[str] = set()
    unknown: list[str] = []
    for raw in tokens:
        token = raw.strip().upper()
        if not token:
            continue
        if token in valid_src_ids:
            resolved.add(token)
        else:
            unknown.append(raw)

    if unknown:
        sample = sorted(valid_src_ids)[:6]
        raise SystemExit(
            f"Unknown source camera ID(s) in --drop-cameras: {unknown}\n"
            f"  Expected source IDs like: {sample} ...\n"
            f"  Total known cameras: {len(valid_src_ids)}"
        )
    return resolved


def apply_camera_filter(
    full_id_map: dict[str, str], drop_set: set[str]
) -> dict[str, str]:
    """Drop source IDs and renumber the surviving cameras contiguously."""
    survivors_src = [
        src_id for src_id in full_id_map.values() if src_id not in drop_set
    ]
    if not survivors_src:
        raise SystemExit("Camera filter removed all cameras; nothing to do.")

    width = max(2, len(str(max(len(survivors_src) - 1, 0))))
    return {f"cam{i:0{width}d}": src_id for i, src_id in enumerate(survivors_src)}


def scale_intrinsics(
    K,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
):
    """Adjust the intrinsic matrix for image resizing."""
    K = K.copy()
    sx = dst_w / src_w
    sy = dst_h / src_h
    K[0, :] *= sx
    K[1, :] *= sy
    return K


def write_thuman4_calibration(
    out_path: Path,
    id_map: dict[str, str],
    src_cams: dict[str, dict],
    dst_size: dict[str, tuple[int, int]] | None,
) -> None:
    """Write the THUman4-style calibration file."""
    out_path.write_text(json.dumps(thuman4_calibration(id_map, src_cams, dst_size), indent=2))


def thuman4_calibration(id_map, src_cams, dst_size):
    """Build expected calibration for writing and resume validation."""
    cal: dict[str, dict] = {}
    for new_id, src_id in id_map.items():
        camera = src_cams[src_id]
        K = camera["K"]
        if dst_size is not None and new_id in dst_size:
            dst_w, dst_h = dst_size[new_id]
            K = scale_intrinsics(K, camera["W"], camera["H"], dst_w, dst_h)
        cal[new_id] = {
            "K": K.flatten().tolist(),
            "R": camera["R"].flatten().tolist(),
            "T": camera["T"].flatten().tolist(),
        }
    return cal


def resize_to_long_edge(
    width: int, height: int, resize_long_edge: int
) -> tuple[int, int]:
    """Return resized dimensions that preserve aspect ratio."""
    if min(width, height, resize_long_edge) <= 0:
        raise ValueError("Image dimensions and resize long edge must be positive")
    long_edge = max(width, height)
    if long_edge <= resize_long_edge:
        return width, height
    scale = resize_long_edge / long_edge
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def build_resize_sizes(
    id_map: dict[str, str],
    src_cams: dict[str, dict],
    resize_long_edge: int | None,
) -> dict[str, tuple[int, int]] | None:
    """Pre-compute output sizes for calibration scaling."""
    if resize_long_edge is None:
        return None

    dst_size: dict[str, tuple[int, int]] = {}
    for new_id, src_id in id_map.items():
        camera = src_cams[src_id]
        dst_size[new_id] = resize_to_long_edge(
            camera["W"], camera["H"], resize_long_edge
        )
    return dst_size
