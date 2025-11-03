#!/usr/bin/env python3
"""Compute FID between two directories of images using project's utilities.

Usage example:
  python compute_fid_between_dirs.py --dir1 fid_out/generated_from_text --dir2 fid_out/generated_from_mapper --batch_size 16

This wraps `compute_fid_from_paths` from `generate_and_compute_fid.py` so it reuses the same Inception extractor and preprocessing.
"""
import argparse
import json
from pathlib import Path
from typing import List

import torch

try:
    from generate_and_compute_fid import compute_fid_from_paths
except Exception as e:
    compute_fid_from_paths = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def gather_image_paths(dir_path: Path, exts=None, max_items: int = None, shuffle: bool = False, seed: int = 42) -> List[str]:
    if exts is None:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_paths = [str(p) for p in dir_path.rglob("*") if p.suffix.lower() in exts]
    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(all_paths)
    if max_items is not None:
        return all_paths[:max_items]
    return all_paths

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir1", required=True, help="First directory of images (e.g., generated_from_text)")
    p.add_argument("--dir2", required=True, help="Second directory of images (e.g., generated_from_mapper)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default=None, help="torch device string, e.g. 'cuda' or 'cpu'")
    p.add_argument("--max_items", type=int, default=None, help="Limit images per dir (useful for quick checks)")
    p.add_argument("--shuffle", action="store_true", help="Shuffle collected image lists before truncation")
    p.add_argument("--out_json", default=None, help="Optional path to save results as JSON")
    args = p.parse_args()

    if compute_fid_from_paths is None:
        print("Error: failed to import compute_fid_from_paths from generate_and_compute_fid.py")
        print("Import error:", _IMPORT_ERR)
        raise SystemExit(1)

    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    if not dir1.exists() or not dir1.is_dir():
        raise SystemExit(f"dir1 not found or not a directory: {dir1}")
    if not dir2.exists() or not dir2.is_dir():
        raise SystemExit(f"dir2 not found or not a directory: {dir2}")

    paths1 = gather_image_paths(dir1, max_items=args.max_items, shuffle=args.shuffle)
    paths2 = gather_image_paths(dir2, max_items=args.max_items, shuffle=args.shuffle)

    if len(paths1) == 0 or len(paths2) == 0:
        raise SystemExit("No images found in one of the directories")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Computing FID between {len(paths1)} images in {dir1} and {len(paths2)} images in {dir2} on device={device}")

    fid_value = compute_fid_from_paths(paths1, paths2, batch_size=args.batch_size, device=device)
    print(f"FID(dir1, dir2) = {fid_value:.6f}")

    result = {"dir1": str(dir1), "n1": len(paths1), "dir2": str(dir2), "n2": len(paths2), "fid": float(fid_value)}
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2))
        print(f"Saved JSON to {args.out_json}")


if __name__ == "__main__":
    main()