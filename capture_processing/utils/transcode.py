"""Image and mask transcoding utilities."""

from __future__ import annotations

from pathlib import Path

from .camera import resize_to_long_edge

IMAGE_EXTENSIONS = (".avif", ".png", ".jpg", ".jpeg")
MASK_DIRS = ("mask", "person_masks", "masks")
MASK_EXTENSIONS = (".avif", ".png", ".jpg", ".jpeg")
IMAGES_DIR = "images"
MASKS_DIR = "masks"


def _save_jpeg_if_changed(image, path, quality):
    """Atomically replace changed bytes; preserve downstream caches for identical output."""
    import hashlib
    import uuid
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    def digest(file):
        h = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        return h.digest()
    try:
        image.save(temporary, 'JPEG', quality=quality, optimize=False)
        if not path.is_file() or digest(path) != digest(temporary):
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def transcode_pair(args: tuple) -> tuple[str, str | None]:
    """Worker: transcode one image/mask pair to JPEG."""
    from PIL import Image

    img_src, img_dst, mask_src, mask_dst, resize_long, jpeg_quality = args
    if any(path.suffix.lower() == ".avif" for path in (img_src, mask_src)):
        import pillow_avif  # noqa: F401  (register AVIF support)
    try:
        with Image.open(img_src) as source:
            img = source.convert("RGB")
        if resize_long is not None:
            new_size = resize_to_long_edge(*img.size, resize_long)
            if new_size != img.size:
                img = img.resize(new_size, Image.LANCZOS)
        _save_jpeg_if_changed(img, img_dst, jpeg_quality)

        with Image.open(mask_src) as source:
            mask = source.convert("L")
        # Binary PNGs may encode foreground as 1. JPEG quantization must operate
        # on 0/255 values, otherwise tiny ringing values trigger an empty mask.
        if mask.getextrema()[1] <= 1:
            mask = mask.point(lambda value: value * 255)
        if mask.size != img.size:
            mask = mask.resize(img.size, Image.NEAREST)
        _save_jpeg_if_changed(mask, mask_dst, jpeg_quality)
        return (str(img_dst), None)
    except Exception as exc:
        return (str(img_dst), f"{type(exc).__name__}: {exc}")


def build_transcoding_jobs(
    frames: list[tuple[int, Path]],
    id_map: dict[str, str],
    out_root: Path,
    resize_long_edge: int | None,
    jpeg_quality: int,
) -> tuple[list[tuple], list[str]]:
    """Collect transcode work items and missing-image records from the source masks."""
    present_lookup = {fid: fdir for fid, fdir in frames}
    n_frames_total = max(fid for fid, _ in frames) + 1
    jobs: list[tuple] = []
    missing: list[str] = []

    for fid in range(n_frames_total):
        if fid not in present_lookup:
            for new_id in id_map:
                missing.append(str(_image_relpath(new_id, fid)))
            continue

        frame_dir = present_lookup[fid]
        for new_id, src_id in id_map.items():
            img_src = _find_camera_file(
                frame_dir / "image", f"color_{src_id}", IMAGE_EXTENSIONS
            )
            mask_src = _find_mask_file(frame_dir, src_id)
            if not img_src.exists() or not mask_src.exists():
                missing.append(str(_image_relpath(new_id, fid)))
                continue

            jobs.append(
                (
                    img_src,
                    out_root / _image_relpath(new_id, fid),
                    mask_src,
                    out_root / _mask_relpath(new_id, fid),
                    resize_long_edge,
                    jpeg_quality,
                )
            )

    return jobs, missing


def _find_camera_file(directory: Path, stem: str, extensions: tuple[str, ...]) -> Path:
    """Return the first existing camera file for a stem and extension set."""
    for extension in extensions:
        path = directory / f"{stem}{extension}"
        if path.exists():
            return path
    return directory / f"{stem}{extensions[0]}"


def _find_mask_file(frame_dir: Path, src_id: str) -> Path:
    """Return the first existing mask file for a source camera."""
    for directory_name in MASK_DIRS:
        path = _find_camera_file(
            frame_dir / directory_name, f"mask_{src_id}", MASK_EXTENSIONS
        )
        if path.exists():
            return path
    return frame_dir / MASK_DIRS[0] / f"mask_{src_id}{MASK_EXTENSIONS[0]}"


def _image_relpath(camera_id: str, frame_id: int) -> Path:
    return Path(IMAGES_DIR) / camera_id / f"color_{frame_id:08d}.jpg"


def _mask_relpath(camera_id: str, frame_id: int) -> Path:
    return Path(MASKS_DIR) / camera_id / f"mask_{frame_id:08d}.jpg"
