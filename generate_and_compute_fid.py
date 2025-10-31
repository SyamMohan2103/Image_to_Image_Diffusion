#!/usr/bin/env python3
"""
Generate images from captions (SD) and from mapper+SD (image-conditioned), then compute FID between the two sets.

This script:
 - samples a subset of rows from a CSV with captions and image filenames
 - for each selected row:
     * uses a Stable Diffusion text-to-image pipeline to generate N variations from the caption
     * uses the project's `generate_variation` (mapper -> SD) to generate N variations from the source image
 - computes FID between the two sets of generated images (per-row) and saves a JSON report

Notes:
 - The script tries to be flexible regarding CSV column names; pass --caption_col and --image_col to match your CSV.
 - FID uses a torchvision InceptionV3 feature extractor (pool3). SciPy is optional for matrix sqrt; if not available a numpy eig-based sqrt is used.
 - This process is heavy on GPU memory when generating with Stable Diffusion. Consider running with CUDA_VISIBLE_DEVICES or on CPU for very small tests.

Example:
  python generate_and_compute_fid.py \
    --csv data/filtered_metadata_parallel.csv \
    --image_root laion_images/ \
    --mapper mapper_ckpt/mapper_best.pth \
    --out_dir fid_out \
    --num_rows 10 \
    --variations 5

"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from transformers import CLIPProcessor
from diffusers import StableDiffusionPipeline

try:
    import scipy.linalg as spla
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# try to import Ignite's FID metric
try:
    from ignite.metrics import FID as IgniteFID
    _HAS_IGNITE = True
except Exception:
    IgniteFID = None
    _HAS_IGNITE = False

# reuse project generation helper
try:
    from run_mapper import generate_variation as run_generate_variation
except Exception:
    run_generate_variation = None


def read_csv_subset(csv_path: str, num_rows: int, caption_col: str, image_col: str):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if caption_col not in r or image_col not in r:
                raise KeyError(f"CSV missing required columns: {caption_col} or {image_col}")
            rows.append({"caption": r[caption_col], "image": r[image_col]})
            if len(rows) >= num_rows:
                break
    return rows


def generate_from_caption(pipe: StableDiffusionPipeline, caption: str, out_dir: Path, variations: int, seed: int = 42, guidance: float = 7.5, steps: int = 50):
    os.makedirs(out_dir, exist_ok=True)
    device = next(pipe.parameters()).device if hasattr(pipe, "parameters") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    results = []
    for i in range(variations):
        gen_seed = int(seed) + i
        generator = torch.Generator(device=device).manual_seed(gen_seed)
        out = pipe(prompt=caption, num_inference_steps=steps, guidance_scale=guidance, generator=generator, output_type="pil")
        img = out.images[0]
        out_path = out_dir / f"caption_var_{i}_seed{gen_seed}.png"
        img.save(out_path)
        results.append(str(out_path))
    return results


def _preprocess_for_inception(img: Image.Image):
    # Inception expects 299x299, normalized
    import torchvision.transforms as T

    tf = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img)


class InceptionV3FeatureExtractor(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        import torchvision.models as models

        self.device = device
        # Use torchvision's inception_v3 and extract features before final FC
        self.model = models.inception_v3(pretrained=True, aux_logits=True)
        self.model.to(device)
        self.model.eval()

    def forward(self, x: torch.Tensor):
        # x shape: [B,3,299,299]
        with torch.no_grad():
            orig_fc = self.model.fc
            self.model.fc = torch.nn.Identity()
            feats = self.model(x)
            self.model.fc = orig_fc
            return feats


def compute_activations(image_paths: List[str], batch_size: int, device: torch.device, num_workers: int = 0):
    """
    Compute Inception activations for a list of image paths.

    When num_workers > 0, uses a torch DataLoader with background workers to parallelize
    image decoding and preprocessing. Corrupted images are skipped with a warning.
    """
    extractor = InceptionV3FeatureExtractor(device=device)
    act_list = []

    if num_workers and num_workers > 0:
        # Parallelized dataloader path
        from torch.utils.data import Dataset, DataLoader

        class ImageDataset(Dataset):
            def __init__(self, paths: List[str]):
                self.paths = paths

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                p = self.paths[idx]
                try:
                    im = Image.open(p).convert("RGB")
                    tensor = _preprocess_for_inception(im)
                    return tensor
                except Exception as e:
                    print(f"Warning: failed to open image {p}: {e}; skipping")
                    return None

        def collate_skip_none(batch):
            batch = [b for b in batch if b is not None]
            if len(batch) == 0:
                return None
            return torch.stack(batch, dim=0)

        dataset = ImageDataset(image_paths)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_skip_none,
        )

        for tensors in loader:
            if tensors is None:
                continue
            tensors = tensors.to(device, non_blocking=True)
            feats = extractor(tensors)
            act_list.append(feats.cpu().numpy())
    else:
        # Original single-threaded path
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    im = Image.open(p).convert("RGB")
                    imgs.append(im)
                except Exception as e:
                    print(f"Warning: failed to open image {p}: {e}; skipping")

            if len(imgs) == 0:
                # nothing valid in this batch
                continue

            tensors = torch.stack([_preprocess_for_inception(im) for im in imgs], dim=0).to(device)
            feats = extractor(tensors)
            act_list.append(feats.cpu().numpy())

    if len(act_list) == 0:
        return np.empty((0, 2048), dtype=np.float32)

    acts = np.concatenate(act_list, axis=0)
    return acts


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    covmean = None
    try:
        if _HAS_SCIPY:
            covmean = spla.sqrtm(sigma1.dot(sigma2))
        else:
            import numpy.linalg as la
            vals, vecs = la.eig(sigma1.dot(sigma2))
            sqrt_vals = np.sqrt(np.where(vals < 0, 0, vals))
            covmean = (vecs * sqrt_vals) @ la.inv(vecs)
    except Exception:
        offset = np.eye(sigma1.shape[0]) * eps
        if _HAS_SCIPY:
            covmean = spla.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        else:
            import numpy.linalg as la
            vals, vecs = la.eig((sigma1 + offset).dot(sigma2 + offset))
            sqrt_vals = np.sqrt(np.where(vals < 0, 0, vals))
            covmean = (vecs * sqrt_vals) @ la.inv(vecs)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(np.real(fd))


def compute_fid_from_paths(paths1: List[str], paths2: List[str], batch_size: int = 8, device: torch.device = None):
    """
    Compute FID between two sets of image file paths.

    Prefer Ignite's FID if available (it accepts PIL images or tensors). Otherwise fall back to the
    existing Inception-based frechet_distance implementation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If Ignite is available, use it. Ignite's FID expects images as torch tensors in [0,1]
    if _HAS_IGNITE:
        # Build a simple loader that yields batches of tensors in range [0,1]
        def pil_to_tensor_batch(paths):
            batches = []
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i+batch_size]
                print(batch_paths)
                imgs = [Image.open(p).convert("RGB") for p in batch_paths]
                tensors = torch.stack([T.ToTensor()(im) for im in imgs], dim=0).to(device)
                batches.append(tensors)
            return batches

        # Import torchvision.transforms locally (alias T used above)
        import torchvision.transforms as T

        fid_metric = IgniteFID(device=device)

        # update with first set
        for b in pil_to_tensor_batch(paths1):
            # Ignite FID expects pixel values in [0, 255] as uint8 or [0,1] float; using [0,1] float ok
            fid_metric.update((b,))

        # compute stats1 by cloning metric internals
        stats1 = fid_metric._compute_stats()

        # reset metric for second set
        fid_metric.reset()

        for b in pil_to_tensor_batch(paths2):
            fid_metric.update((b,))

        stats2 = fid_metric._compute_stats()

        # call ignite's internal distance (it exposes compute but needs both stats)
        try:
            # newer ignite versions allow passing two state dicts via compute()
            fid_value = float(fid_metric.compute(stats1=stats1, stats2=stats2))
        except TypeError:
            # fallback: compute frechet directly from stats
            mu1, sigma1 = stats1["mu"].cpu().numpy(), stats1["cov"].cpu().numpy()
            mu2, sigma2 = stats2["mu"].cpu().numpy(), stats2["cov"].cpu().numpy()
            fid_value = frechet_distance(mu1, sigma1, mu2, sigma2)

        return fid_value

    # Fallback: compute activations with our Inception extractor and calculate FID
    acts1 = compute_activations(paths1, batch_size, device)
    acts2 = compute_activations(paths2, batch_size, device)
    mu1 = np.mean(acts1, axis=0)
    mu2 = np.mean(acts2, axis=0)
    sigma1 = np.cov(acts1, rowvar=False)
    sigma2 = np.cov(acts2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--mapper", required=True)
    p.add_argument("--mapper_image_col", default="image_path", help="CSV column with image filename (joined with --image_root)")
    p.add_argument("--caption_col", default="caption", help="CSV column with caption text")
    p.add_argument("--num_rows", type=int, default=10)
    p.add_argument("--variations", type=int, default=5)
    p.add_argument("--out_dir", default="fid_out")
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = read_csv_subset(args.csv, args.num_rows, args.caption_col, args.mapper_image_col)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # load SD pipeline for caption -> images
    print("Loading Stable Diffusion pipeline for caption generation (this may be large)...")
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32))
    pipe = pipe.to(device)

    fid_results = {}
    # accumulate all generated image paths for aggregated FID
    all_caption_paths = []
    all_mapper_paths = []

    for idx, r in enumerate(rows):
        caption = r["caption"]
        image_fname = r["image"]
        src_img_path = Path(args.image_root) / image_fname
        if not src_img_path.exists():
            print(f"Warning: source image not found: {src_img_path}; skipping row {idx}")
            continue

        print(f"Processing row {idx}: caption='{caption[:80]}...' image={src_img_path}")
        caption_dir = out_root / f"row{idx}_caption"
        mapper_dir = out_root / f"row{idx}_mapper"
        caption_dir.mkdir(parents=True, exist_ok=True)
        mapper_dir.mkdir(parents=True, exist_ok=True)

        # --- caption generation (skip if already present) ---
        existing_caption_imgs = sorted([p for p in caption_dir.glob("*.png")])
        if len(existing_caption_imgs) >= args.variations:
            cap_paths = [str(p) for p in existing_caption_imgs[: args.variations]]
            print(f"Found {len(cap_paths)} existing caption-generated images in {caption_dir}; skipping generation.")
        else:
            cap_paths = generate_from_caption(pipe, caption, caption_dir, variations=args.variations, seed=args.seed)

        # --- mapper generation (skip if already present) ---
        existing_mapper_imgs = sorted([p for p in mapper_dir.glob("*.png")])
        if len(existing_mapper_imgs) >= args.variations:
            mapper_paths = [str(p) for p in existing_mapper_imgs[: args.variations]]
            print(f"Found {len(mapper_paths)} existing mapper-generated images in {mapper_dir}; skipping generation.")
        else:
            if run_generate_variation is None:
                raise RuntimeError("run_mapper.generate_variation not available; ensure run_mapper.py is importable from this script")
            mapper_paths = run_generate_variation(
                mapper_path=args.mapper,
                clip_model_name="openai/clip-vit-large-patch14",
                sd_model_name=args.sd_model,
                input_image_path=str(src_img_path),
                out_dir=str(mapper_dir),
                device=device,
                num_inference_steps=50,
                guidance_scale=7.5,
                variations=args.variations,
                seed=args.seed,
            )

        # compute FID between cap_paths and mapper_paths
        fid = compute_fid_from_paths(cap_paths, mapper_paths, batch_size=args.batch_size, device=device)
        print(f"Row {idx} FID: {fid:.4f}")
        fid_results[f"row_{idx}"] = {"caption": caption, "image": str(src_img_path), "fid": fid, "caption_images": cap_paths, "mapper_images": mapper_paths}
        # accumulate for aggregated FID
        all_caption_paths.extend(cap_paths)
        all_mapper_paths.extend(mapper_paths)

    out_json = out_root / "fid_results.json"
    # compute aggregated FID across all rows (if we have at least one image per set)
    if len(all_caption_paths) > 0 and len(all_mapper_paths) > 0:
        try:
            agg_fid = compute_fid_from_paths(all_caption_paths, all_mapper_paths, batch_size=args.batch_size, device=device)
            fid_results["aggregated"] = {"num_caption_images": len(all_caption_paths), "num_mapper_images": len(all_mapper_paths), "fid": agg_fid}
            print(f"Aggregated FID across all rows: {agg_fid:.4f}")
        except Exception as e:
            print(f"Warning: failed to compute aggregated FID: {e}")

    out_json.write_text(json.dumps(fid_results, indent=2))
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
