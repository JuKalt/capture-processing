#!/usr/bin/env python3
"""Install and verify every MAMMA asset required by the VCI worker pipeline."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from zipfile import BadZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.setup.setup_smplx_model import install, MODEL_SHA256, DEFAULT_ASSET_SOURCE as SHARED_ASSETS

# SHA-256 of the working controller's assets. Keep these with the matching
# model versions; never accept a nonempty, incomplete checkpoint as ready.
ASSETS = {
    'data/weights/yolo/yolo12x.pt': '682ce8dadee004dbe964950f1bf3eda451671815a6ed62db80b398916b9b7c6f',
    'data/weights/sam2/sam2.1_hiera_large.pt': '2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318',
    'data/weights/ma_2d/mamma_mask_full_cvpr.ckpt': '3fd6793cf72f9994eded544d902a30b2cd5e9e5d225c1be110980ee4cadd8b29',
    'data/body_models/downsampled_verts/verts_512.pkl': '85c6acd5618a2b93d96640285a6f3e1e02383a046140d882663e6de5a6185c82',
    'configs/sam2.1/sam2.1_hiera_l.yaml': '1dbd6cb6dfebeaf588c7006ee222c6efbfa9049a7ad472a3cdfb2f5d919e8107',
    # MAMMA loads all three variants before fitting, even for neutral fits.
    'data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz': MODEL_SHA256,
    'data/body_models/smplx_locked_head/smplx/SMPLX_MALE.npz': '44e57050d127538e20e10e27f8274ba9ca3dd09c992e954d3d2b5ae5e5bb8313',
    'data/body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz': 'a32f91b24e13023b979ec0b5c8e9086f5918197132329c8a258ad1f1c6c557f6',
}


def setup(source, destination, check_only=False):
    errors = []
    for relative, digest in ASSETS.items():
        try:
            install(Path(source) / relative, Path(destination) / relative, digest,
                    check_only=check_only, label='MAMMA asset')
        except (OSError, ValueError) as exc:
            errors.append(f'{relative}: {exc}')
    if errors:
        raise ValueError('MAMMA assets are not ready:\n  - ' + '\n  - '.join(errors)
                         + '\nSeed shared assets on the controller with:\n'
                         '  python3 scripts/setup/setup_mamma_assets.py --seed\n'
                         'Then rerun worker setup. VCI_MAMMA_ASSET_SOURCE can override the shared source directory.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT / 'external/mamma')
    parser.add_argument('--source', type=Path, default=os.environ.get('VCI_MAMMA_ASSET_SOURCE', SHARED_ASSETS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Check all installed checksums without changing files')
    mode.add_argument('--seed', action='store_true', help='Copy this controller repository\'s assets to the shared source')
    mode.add_argument('--download', action='store_true', help='Download missing assets from the official sources')
    mode.add_argument('--public-only', action='store_true', help='Download only public assets')
    mode.add_argument('--gated-only', action='store_true', help='Download only gated assets')
    args = parser.parse_args()
    try:
        if args.download or args.public_only or args.gated_only:
            from scripts.setup.download_assets import download

            download(args.repo, ASSETS, public=not args.gated_only, gated=not args.public_only)
            if not args.download:
                return
            setup(args.repo, args.repo, check_only=True)
            return
        setup(args.repo if args.seed else args.source,
              args.source if args.seed else args.repo, args.check)
    except (OSError, ValueError, EOFError, BadZipFile) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
