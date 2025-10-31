import argparse
import json
import os
import random
import numpy as np
from diffusers import StableDiffusionPipeline
from pathlib import Path
from typing import List

import torch
from generate_and_compute_fid import (
    read_csv_subset,
    compute_activations,
    frechet_distance,
    generate_from_caption,
)
try:
    from generate_and_compute_fid import compute_fid_from_paths
except Exception:
    compute_fid_from_paths = None

# reuse run_mapper helper to generate mapper-based variants
from run_mapper import generate_variation as run_generate_variation


def sample_true_image_paths_from_csv(csv_path: str, image_root: str, image_col: str, sample_size: int, seed: int):
    rows = read_csv_subset(csv_path, sample_size, caption_col="caption", image_col=image_col)
    paths = [str(Path(image_root) / r["image"]) for r in rows if (Path(image_root) / r["image"]).exists()]
    if len(paths) < sample_size:
        # try to read more rows to reach sample_size
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_paths = []
            for r in reader:
                p = Path(image_root) / r[image_col]
                if p.exists():
                    all_paths.append(str(p))
        random.seed(seed)
        if len(all_paths) < sample_size:
            raise RuntimeError(f"Not enough images found under {image_root} to sample {sample_size}")
        paths = random.sample(all_paths, sample_size)
    return paths[:sample_size]


def sample_true_image_paths_from_dir(image_root: str, sample_size: int, seed: int):
    root = Path(image_root)
    all_images = [str(p) for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if len(all_images) < sample_size:
        raise RuntimeError(f"Not enough images under {image_root} to sample {sample_size}")
    random.seed(seed)
    return random.sample(all_images, sample_size)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=False, help="CSV with captions and image filenames (used for generating and/or sampling true set)")
    p.add_argument("--image_root", required=True, help="Root dir for LAION images")
    p.add_argument("--image_col", default="image_path", help="CSV column name for image filenames")
    p.add_argument("--true_sample_size", type=int, default=5000, help="Number of true images to sample for 'true' distribution")
    p.add_argument("--num_rows", type=int, default=1000, help="Number of caption rows to sample for generating text images (will generate num_rows * variations images)")
    p.add_argument("--variations", type=int, default=5, help="Variations per caption / per mapper image")
    p.add_argument("--mapper", required=True, help="Path to mapper checkpoint for mapper-based generation")
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model id")
    p.add_argument("--out_dir", default="fid_true_vs_gen_out")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for Inception activation extraction")
    p.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for image decoding/preprocessing during activation extraction")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_csv_for_true", action="store_true", help="If set, sample true images from the provided CSV (requires --csv). Otherwise sample from image_root dir.")
    p.add_argument("--caption_col", default="caption", help="CSV column name for caption text")
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) sample true distribution images
    print("Sampling true images for LAION true distribution...")
    if args.use_csv_for_true:
        if not args.csv:
            raise RuntimeError("--use_csv_for_true requires --csv")
        true_paths = sample_true_image_paths_from_csv(args.csv, args.image_root, args.image_col, args.true_sample_size, args.seed)
    else:
        true_paths = sample_true_image_paths_from_dir(args.image_root, args.true_sample_size, args.seed)

    print(f"Computing activations for {len(true_paths)} true images (this may take time)...")
    acts_true = compute_activations(true_paths, batch_size=args.batch_size, device=device, num_workers=args.num_workers)
    mu_true = acts_true.mean(axis=0)
    sigma_true = np.cov(acts_true, rowvar=False)
    true_stats_path = out_root / "true_stats.npz"
    
    np.savez_compressed(str(true_stats_path), mu=mu_true, sigma=sigma_true)
    print(f"Saved true stats to {true_stats_path}")

    # 2) generate images from captions (sample captions from CSV)
    if not args.csv:
        raise RuntimeError("Caption generation requires --csv pointing to filtered_metadata_parallel.csv")
    print(f"Sampling {args.num_rows} caption rows to generate text-based images...")
    rows = read_csv_subset(args.csv, args.num_rows, args.caption_col, args.image_col)
    # load SD pipeline for caption generation
    
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32)).to(device)
    text_gen_dir = out_root / "generated_from_text"
    text_gen_dir.mkdir(parents=True, exist_ok=True)
    all_text_paths: List[str] = []
    print("Generating caption->image variants...")
    for i, r in enumerate(rows):
        caption = r["caption"]
        subdir = text_gen_dir / f"row{i}"
        subdir.mkdir(parents=True, exist_ok=True)
        # check for existing generated images and skip generation if present
        existing = [str(p) for p in subdir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if len(existing) >= args.variations:
            # sort for deterministic order
            existing = sorted(existing)
            cap_paths = existing[: args.variations]
            print(f"Reusing {len(cap_paths)} existing text-generated images for row {i} from {subdir}")
        else:
            cap_paths = generate_from_caption(pipe, caption, subdir, variations=args.variations, seed=args.seed + i)
        all_text_paths.extend(cap_paths)

    # 3) generate images using mapper + SD from source images
    print("Generating mapper->image variants...")
    mapper_gen_dir = out_root / "generated_from_mapper"
    mapper_gen_dir.mkdir(parents=True, exist_ok=True)
    all_mapper_paths: List[str] = []
    for i, r in enumerate(rows):
        image_fname = r["image"]
        src_img_path = Path(args.image_root) / image_fname
        if not src_img_path.exists():
            print(f"Warning: source image not found: {src_img_path}; skipping")
            continue
        subdir = mapper_gen_dir / f"row{i}"
        subdir.mkdir(parents=True, exist_ok=True)
        # check for existing mapper-generated images and skip generation if present
        existing_map = [str(p) for p in subdir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if len(existing_map) >= args.variations:
            existing_map = sorted(existing_map)
            mapper_paths = existing_map[: args.variations]
            print(f"Reusing {len(mapper_paths)} existing mapper-generated images for row {i} from {subdir}")
        else:
            mapper_paths = run_generate_variation(
                mapper_path=args.mapper,
                clip_model_name="openai/clip-vit-large-patch14",
                sd_model_name=args.sd_model,
                input_image_path=str(src_img_path),
                out_dir=str(subdir),
                device=device,
                num_inference_steps=50,
                guidance_scale=7.5,
                variations=args.variations,
                seed=args.seed + i,
                use_source_latents=False,  # decoupled
            )
        all_mapper_paths.extend(mapper_paths)

    # 4) compute activations & stats for generated sets and compute FID(true, gen)
    print(f"Computing activations for {len(all_text_paths)} text-generated images...")
    acts_text = compute_activations(all_text_paths, batch_size=args.batch_size, device=device, num_workers=args.num_workers)
    mu_text = acts_text.mean(axis=0)
    sigma_text = np.cov(acts_text, rowvar=False)

    print(f"Computing activations for {len(all_mapper_paths)} mapper-generated images...")
    acts_mapper = compute_activations(all_mapper_paths, batch_size=args.batch_size, device=device, num_workers=args.num_workers)
    mu_mapper = acts_mapper.mean(axis=0)
    sigma_mapper = np.cov(acts_mapper, rowvar=False)

    fid_text = frechet_distance(mu_true, sigma_true, mu_text, sigma_text)
    fid_mapper = frechet_distance(mu_true, sigma_true, mu_mapper, sigma_mapper)

    results = {
        "true": {"n": len(true_paths), "stats": str(true_stats_path)},
        "gen_text": {"n": len(all_text_paths), "fid_vs_true": float(fid_text)},
        "gen_mapper": {"n": len(all_mapper_paths), "fid_vs_true": float(fid_mapper)},
    }

    out_json = out_root / "fid_true_vs_generated.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {out_json}")
    print(f"FID(true, gen_text) = {fid_text:.4f}")
    print(f"FID(true, gen_mapper) = {fid_mapper:.4f}")


if __name__ == "__main__":
    main()