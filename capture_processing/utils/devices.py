"""GPU slot parsing shared by the batch preprocessing entry points."""
from __future__ import annotations


def parse_device_pool(device: str | None, devices: str | None, gpu_workers: int) -> list[str]:
    """Expand --device/--devices/--gpu-workers into one entry per concurrent job slot."""
    if gpu_workers <= 0:
        raise SystemExit("--gpu-workers must be a positive integer")
    if device and devices:
        raise SystemExit("Use either --device for one device or --devices for a pool, not both.")
    if device:
        return [device] * gpu_workers
    if not devices:
        if gpu_workers != 1:
            raise SystemExit("--gpu-workers requires --device or --devices")
        return []
    parsed = [item.strip() for item in devices.split(",") if item.strip()]
    if not parsed:
        raise SystemExit("--devices must contain at least one device, e.g. --devices 0,1")
    return [item for item in parsed for _ in range(gpu_workers)]
