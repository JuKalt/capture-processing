#!/usr/bin/env python3
"""Install the controller's SMPL-X model from shared storage with SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('external/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz')
# Lab shared-storage default; VCI_MAMMA_ASSET_SOURCE overrides it everywhere.
DEFAULT_ASSET_SOURCE = ROOT / 'models/authorized-mamma-assets'
SHARED_MODEL = (Path(os.environ.get('VCI_MAMMA_ASSET_SOURCE', DEFAULT_ASSET_SOURCE))
                / 'data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz')
# The existing locked-head model on the controller, not the standard SMPL-X model.
MODEL_SHA256 = '43d8f3a1375d7c5baae207870a5d51def0f7e6b507df709b4937598b5e7d965d'


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def install(source, destination, expected, check_only=False, label='SMPL-X'):
    source, destination = Path(source), Path(destination)
    if destination.is_file() and checksum(destination) == expected:
        print(f'{label} model verified: {destination}', flush=True)
        return
    if check_only:
        raise ValueError(f'{label} model missing or checksum mismatch: {destination}')
    if not source.is_file():
        raise ValueError(f'{label} source missing: {source}; seed the shared model from the controller '
                         'or set VCI_SMPLX_MODEL_SOURCE to an existing model file')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.smplx-', delete=False) as stream:
            temporary = Path(stream.name)
            with source.open('rb') as input_stream:
                shutil.copyfileobj(input_stream, stream)
        actual = checksum(temporary)
        if actual != expected:
            raise ValueError(f'{label} source checksum mismatch: expected {expected}, got {actual}')
        temporary.chmod(0o644)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'{label} model copied and verified: {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=os.environ.get('VCI_SMPLX_MODEL_SOURCE', SHARED_MODEL))
    parser.add_argument('--destination', type=Path, default=ROOT / MODEL)
    parser.add_argument('--sha256', default=MODEL_SHA256)
    parser.add_argument('--check', action='store_true', help='Verify the installed model without copying')
    args = parser.parse_args()
    if len(args.sha256) != 64 or any(c not in '0123456789abcdef' for c in args.sha256):
        parser.error('--sha256 must be a lowercase SHA-256 digest')
    try:
        install(args.source, args.destination, args.sha256, args.check)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
