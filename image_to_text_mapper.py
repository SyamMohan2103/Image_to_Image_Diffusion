"""
Image → Text-Embedding Mapper for Frozen Stable Diffusion
-------------------------------------------------------

This script trains a small MLP that maps CLIP *vision* pooled features
to a sequence of token embeddings compatible with the CLIP *text* encoder
used by Stable Diffusion (so they can be used as encoder_hidden_states
in the U-Net cross-attention). The Stable Diffusion model remains frozen.

What the script does:
  - Loads a CLIP model (vision + text) and a dataset of (image_path, caption).
  - Freezes CLIP weights and trains only a small Image->Text mapper (MLP).
  - Saves the mapper to disk.
  - Includes an inference helper that shows how to plug the mapper's output
    into a frozen Stable Diffusion pipeline to generate variations of an
    input image (img2img-style denoising with the mapped embeddings).

Requirements (run locally):
  - Python 3.8+
  - torch
  - transformers
  - diffusers
  - torchvision
  - pillow
  - pandas (for reading CSV dataset)

Dataset format expected:
  - A CSV file with two columns: `image_path`, `caption`.
  - `image_path` should be a filesystem path to the image (relative or absolute).

Usage examples (after installing requirements):
  Train:
    python image_to_text_mapper.py --mode train --dataset my_data.csv --out_dir ./mapper_ckpt \
      --clip_model openai/clip-vit-large-patch14 --epochs 5 --batch_size 16 --lr 1e-4

  Inference / generate a variation:
    python image_to_text_mapper.py --mode gen --mapper ./mapper_ckpt/mapper.pth \
      --input_image my_image.jpg --out_dir ./out --sd_model runwayml/stable-diffusion-v1-5 \
      --strength 0.7 --guidance 3.0

Read the code docstrings and comments for details.

"""

import os
import argparse
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

from transformers import (
    CLIPProcessor,
    CLIPModel,
    CLIPTextModel,
    CLIPTokenizer,
)

# NOTE: diffusers is used only in the generation helper. The training part
# only needs transformers + torch.
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler, StableDiffusionPipeline


# ------------------------------ Dataset ------------------------------
class ImageCaptionDataset(Dataset):
    """Simple dataset that reads `image_path, caption` from a CSV file.

    CSV must contain columns named `image_path` and `caption` (customizable via args).
    We return PIL images (not tensors) and the caption string so batching can
    use the CLIPProcessor for images and the CLIPTokenizer for captions.
    """

    def __init__(self, csv_path: str, image_col: str = "image_path", caption_col: str = "caption"):
        df = pd.read_csv(csv_path)
        if image_col not in df.columns or caption_col not in df.columns:
            raise ValueError(f"CSV must contain columns '{image_col}' and '{caption_col}'")
        self.df = df
        self.image_col = image_col
        self.caption_col = caption_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        caption = str(row[self.caption_col])
        img = Image.open(img_path).convert("RGB")
        return img, caption


# ------------------------------ Mapper (MLP) ------------------------------
class ImageToTextMapper(nn.Module):
    """MLP that maps CLIP pooled image features (B, I) to a target
    token embedding sequence (B, L, H) where L is text seq length (e.g. 77)
    and H is the text hidden dim (e.g. 768).

    The final linear layer outputs L*H values which are reshaped.
    """

    def __init__(self, in_dim: int, out_seq_len: int, out_dim: int, hidden_dim: int = 4096, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        cur = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        # final projection to sequence*dim
        layers.append(nn.Linear(cur, out_seq_len * out_dim))
        self.net = nn.Sequential(*layers)
        self.out_seq_len = out_seq_len
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        b = x.shape[0]
        out = self.net(x)
        out = out.view(b, self.out_seq_len, self.out_dim)
        return out


# ------------------------------ Training Utilities ------------------------------

def collate_batch(batch: List[Tuple[Image.Image, str]]):
    images, captions = zip(*batch)
    return list(images), list(captions)


def init_models(clip_model_name: str, device: torch.device):
    """Load CLIP components and return processor, clip_model, text_model, tokenizer.

    We use:
      - CLIPModel (for get_image_features)
      - CLIPTokenizer + CLIPTextModel (for per-token text embeddings targets)

    The function also runs a single dummy pass to infer dims.
    """
    print(f"Loading CLIP model: {clip_model_name}")
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)

    # Freeze CLIP weights
    clip_model.eval()
    text_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    for p in text_model.parameters():
        p.requires_grad = False

    # run dummy forward to infer dims
    dummy_img = Image.new("RGB", (224, 224), color="white")
    inputs = processor(images=dummy_img, return_tensors="pt")
    with torch.no_grad():
        img_inputs = {k: v.to(device) for k, v in inputs.items()}
        img_feats = clip_model.get_image_features(**img_inputs)

    in_dim = img_feats.shape[-1]
    out_seq_len = tokenizer.model_max_length
    out_dim = text_model.config.hidden_size

    print(f"in_dim (vision pooled features) = {in_dim}")
    print(f"out_seq_len (token length) = {out_seq_len}")
    print(f"out_dim (text hidden size) = {out_dim}")

    return processor, clip_model, tokenizer, text_model, in_dim, out_seq_len, out_dim


# ------------------------------ Train Loop ------------------------------

def train_mapper(
    dataset_csv: str,
    clip_model_name: str,
    out_dir: str,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-4,
    hidden_dim: int = 4096,
    num_layers: int = 3,
    save_every: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)

    # init models
    processor, clip_model, tokenizer, text_model, in_dim, out_seq_len, out_dim = init_models(clip_model_name, device)

    # create mapper
    mapper = ImageToTextMapper(in_dim=in_dim, out_seq_len=out_seq_len, out_dim=out_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    # dataset + dataloader
    ds = ImageCaptionDataset(dataset_csv)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch, num_workers=4)

    optimizer = torch.optim.AdamW(mapper.parameters(), lr=lr, weight_decay=0.01)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        mapper.train()
        running_loss = 0.0
        total_tokens = 0
        total_batches = 0

        for images, captions in dl:
            # images: list[PIL], captions: list[str]
            # 1) image -> image pooled features
            inputs = processor(images=images, return_tensors="pt")
            img_inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                img_feats = clip_model.get_image_features(**img_inputs)  # [B, in_dim]

            # 2) captions -> tokenized -> text encoder outputs (target sequence)
            tokenized = tokenizer(list(captions), padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            with torch.no_grad():
                text_outs = text_model(input_ids=input_ids, attention_mask=attention_mask)
                text_embeds = text_outs.last_hidden_state  # [B, L, H]

            # 3) mapper prediction
            pred = mapper(img_feats)  # [B, L, H]

            # 4) compute masked MSE loss (only over tokens where attention_mask == 1)
            # MSE per token, then averaged only over non-masked tokens
            diff2 = (pred - text_embeds).pow(2).mean(dim=-1)  # [B, L]
            masked = diff2 * attention_mask
            loss = masked.sum() / (attention_mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / max(1, total_batches)
        print(f"Epoch {epoch}/{epochs} — avg masked MSE loss: {avg_loss:.6f}")

        # save checkpoint
        ckpt_path = os.path.join(out_dir, f"mapper_epoch{epoch}.pth")
        torch.save({"state_dict": mapper.state_dict(), "config": {"in_dim": in_dim, "out_seq_len": out_seq_len, "out_dim": out_dim}}, ckpt_path)
        print(f"Saved mapper checkpoint to {ckpt_path}")

    # final save
    final_path = os.path.join(out_dir, "mapper.pth")
    torch.save({"state_dict": mapper.state_dict(), "config": {"in_dim": in_dim, "out_seq_len": out_seq_len, "out_dim": out_dim}}, final_path)
    print(f"Saved final mapper to {final_path}")
    return final_path


# ------------------------------ Inference: generate variations ------------------------------

def generate_variation(
    mapper_path: str,
    clip_model_name: str,
    sd_model_name: str,
    input_image_path: str,
    out_dir: str,
    device: torch.device,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    strength: float = 0.7,
    seed: int = 42,
):
    """Generate an image variation by:
      - computing CLIP image features for the input image
      - mapping them to text-token embeddings via the trained mapper
      - running a frozen SD denoising loop conditioned on the mapped embeddings

    This follows the img2img approach (encode image to latents, add noise at a
    timestep corresponding to `strength`, then denoise conditioned on mapped tokens).
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    # load clip models
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    # freeze
    clip_model.eval(); text_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    for p in text_model.parameters():
        p.requires_grad = False

    # load mapper
    data = torch.load(mapper_path, map_location=device)
    cfg = data.get("config")
    mapper = ImageToTextMapper(in_dim=cfg["in_dim"], out_seq_len=cfg["out_seq_len"], out_dim=cfg["out_dim"]).to(device)
    mapper.load_state_dict(data["state_dict"])
    mapper.eval()

    # load SD pipeline components (we'll use pipeline to get vae, unet, scheduler)
    pipe = StableDiffusionPipeline.from_pretrained(sd_model_name, torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None  # optional
    vae = pipe.vae
    unet = pipe.unet
    scheduler = pipe.scheduler

    # prepare image -> latents
    img = Image.open(input_image_path).convert("RGB")
    init_image = img.resize((512, 512))
    # preprocess for VAE: pipeline has a helper -> but we prepare tensor directly
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img_tensor = preprocess(init_image).unsqueeze(0).to(device=device, dtype=torch.float32)

    # encode with vae
    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample()  # [B, C, H/8, W/8]
        latents = latents * vae.config.scaling_factor

    # choose a timestep to add noise based on strength
    scheduler.set_timesteps(num_inference_steps)
    init_timestep = int(strength * (num_inference_steps - 1))
    t = scheduler.timesteps[init_timestep]

    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    # compute CLIP image features -> mapper -> token embeddings
    inputs = processor(images=init_image, return_tensors="pt")
    img_inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_feats = clip_model.get_image_features(**img_inputs)  # [1, in_dim]
        mapped = mapper(img_feats)  # [1, L, H]

    # compute unconditional embeddings (empty prompt) to allow classifier-free guidance
    uncond_tokens = tokenizer([""], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
    uncond_input_ids = uncond_tokens.input_ids.to(device)
    uncond_attn = uncond_tokens.attention_mask.to(device)
    with torch.no_grad():
        uncond_emb = text_model(input_ids=uncond_input_ids, attention_mask=uncond_attn).last_hidden_state

    # prepare for guidance: duplicate latents and embeddings
    mapped_emb_cat = torch.cat([uncond_emb, mapped], dim=0)  # [2, L, H]

    # denoising loop
    latents = noisy_latents
    for i, t in enumerate(scheduler.timesteps[init_timestep:]):
        # for classifier-free guidance we duplicate the latents
        latent_model_input = torch.cat([latents] * 2)

        # predict noise
        with torch.no_grad():
            model_out = unet(latent_model_input, t, encoder_hidden_states=mapped_emb_cat).sample

        eps_uncond, eps_cond = model_out.chunk(2)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # scheduler step
        step = scheduler.step(eps, t, latents)
        latents = step.prev_sample

    # decode
    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        dec = vae.decode(latents).sample

    # convert to PIL
    dec = (dec / 2 + 0.5).clamp(0, 1)
    dec = dec.cpu().permute(0, 2, 3, 1).numpy()
    dec_images = [Image.fromarray((img * 255).round().astype('uint8')) for img in (dec[0])]

    # NOTE: above conversion assumes single image; for robust saving we'll use a standard pipeline
    out_path = os.path.join(out_dir, "variation.png")
    # easier: reuse pipeline's decode helper for stable results (if available)
    try:
        image = pipe.numpy_to_pil(dec)[0]
    except Exception:
        # fallback: build PIL from numpy as best effort
        image = Image.fromarray((dec[0] * 255).astype('uint8'))

    image.save(out_path)
    print(f"Saved variation to {out_path}")
    return out_path


# ------------------------------ CLI ------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "gen"], required=True)

    # common
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # train
    p.add_argument("--dataset", help="CSV file with image_path,caption")
    p.add_argument("--out_dir", default="./mapper_ckpt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=4096)
    p.add_argument("--num_layers", type=int, default=3)

    # gen
    p.add_argument("--mapper", help="Path to trained mapper (mapper.pth)")
    p.add_argument("--input_image", help="Input image to vary")
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--strength", type=float, default=0.7)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    if args.mode == "train":
        if not args.dataset:
            raise ValueError("--dataset is required for training mode")
        train_mapper(
            dataset_csv=args.dataset,
            clip_model_name=args.clip_model,
            out_dir=args.out_dir,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )

    elif args.mode == "gen":
        if not args.mapper or not args.input_image:
            raise ValueError("--mapper and --input_image are required for gen mode")
        generate_variation(
            mapper_path=args.mapper,
            clip_model_name=args.clip_model,
            sd_model_name=args.sd_model,
            input_image_path=args.input_image,
            out_dir=args.out_dir,
            device=device,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance,
            strength=args.strength,
        )

