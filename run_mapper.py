import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline
import wandb
from tqdm import tqdm
from mapper_model import ImageToTextMapper
from laion_dataset import ImageCaptionDataset
from typing import List, Tuple, Optional
from torch.utils.tensorboard import SummaryWriter


# ==========================================================
# Data utilities
# ==========================================================
def collate_batch(batch: List[Tuple[Image.Image, str]]):
    images, captions = zip(*batch)
    return list(images), list(captions)


# ==========================================================
# Model initialization
# ==========================================================
def init_models(clip_model_name: str, device: torch.device):
    print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Loading CLIP model: {clip_model_name}")
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)

    clip_model.eval()
    text_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    for p in text_model.parameters():
        p.requires_grad = False

    # Dummy forward to infer dims
    dummy_img = Image.new("RGB", (224, 224), color="white")
    inputs = processor(images=dummy_img, return_tensors="pt")
    with torch.no_grad():
        img_inputs = {k: v.to(device) for k, v in inputs.items()}
        img_feats = clip_model.get_image_features(**img_inputs)

    in_dim = img_feats.shape[-1]
    out_seq_len = tokenizer.model_max_length
    out_dim = text_model.config.hidden_size
    if dist.get_rank() == 0 or not dist.is_initialized():
        print(f"in_dim = {in_dim}, out_seq_len = {out_seq_len}, out_dim = {out_dim}")
    return processor, clip_model, tokenizer, text_model, in_dim, out_seq_len, out_dim


# ==========================================================
# DDP training
# ==========================================================
def train_mapper_ddp(
    rank,
    world_size,
    dataset_csv,
    clip_model_name,
    out_dir,
    epochs=3,
    batch_size=8,
    lr=1e-4,
    hidden_dim=4096,
    num_layers=3,
    prefix: str = "laion_subset",
    image_path_prefix=None,
):
    # 1. Setup DDP environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 2. Logging (rank 0 only)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        # writer = SummaryWriter(os.path.join(out_dir, "tb_logs"))
        wandb.init(project="image-to-text-mapper-ddp")

    # 3. Init CLIP + Mapper
    processor, clip_model, tokenizer, text_model, in_dim, out_seq_len, out_dim = init_models(clip_model_name, device)

    base_mapper = ImageToTextMapper(
        in_dim=in_dim, out_seq_len=out_seq_len, out_dim=out_dim,
        hidden_dim=hidden_dim, num_layers=num_layers
    ).to(device)

    mapper = nn.parallel.DistributedDataParallel(base_mapper, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # 4. Dataset + Sampler
    dataset = ImageCaptionDataset(dataset_csv, image_path_prefix=image_path_prefix)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            collate_fn=collate_batch, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(mapper.parameters(), lr=lr, weight_decay=0.01)

    # 5. Training loop
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        mapper.train()
        running_loss = 0.0
        total_batches = 0

        for images, captions in tqdm(dataloader, disable=(rank != 0)):
            images = [img.convert("RGB") for img in images]
            inputs = processor(images=images, return_tensors="pt", do_normalize=True, do_resize=True, do_center_crop=True)
            img_inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                img_feats = clip_model.get_image_features(**img_inputs)

            tokenized = tokenizer(list(captions), padding="max_length", truncation=True,
                                  max_length=tokenizer.model_max_length, return_tensors="pt")
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            with torch.no_grad():
                text_embeds = text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            pred = mapper(img_feats)
            diff2 = (pred - text_embeds).pow(2).mean(dim=-1)
            masked = diff2 * attention_mask
            loss = masked.sum() / (attention_mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / max(1, total_batches)

        # Only rank 0 logs & saves
        if rank == 0:
            print(f"[Rank 0] Epoch {epoch}/{epochs} — avg masked MSE loss: {avg_loss:.6f}")
            wandb.log({"epoch": epoch, "avg_loss": avg_loss})
            ckpt_path = os.path.join(out_dir, f"mapper_epoch{epoch}.pth")
            torch.save({
                "state_dict": mapper.module.state_dict(),
                "config": {"in_dim": in_dim, "out_seq_len": out_seq_len, "out_dim": out_dim}
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    if rank == 0:
        final_path = os.path.join(out_dir, "mapper_final.pth")
        torch.save({
            "state_dict": mapper.module.state_dict(),
            "config": {"in_dim": in_dim, "out_seq_len": out_seq_len, "out_dim": out_dim}
        }, final_path)
        print(f"✅ Training completed, saved final mapper to {final_path}")
        wandb.finish()
        # writer.close()

    dist.destroy_process_group()


# ==========================================================
# Generation (unchanged)
# ==========================================================
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
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device)
    clip_model.eval(); text_model.eval()
    for p in list(clip_model.parameters()) + list(text_model.parameters()):
        p.requires_grad = False

    data = torch.load(mapper_path, map_location=device)
    cfg = data.get("config")
    mapper = ImageToTextMapper(in_dim=cfg["in_dim"], out_seq_len=cfg["out_seq_len"], out_dim=cfg["out_dim"]).to(device)
    state_dict = data["state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        state_dict = new_state_dict
    mapper.load_state_dict(state_dict)
    mapper.eval()

    pipe = StableDiffusionPipeline.from_pretrained(sd_model_name, torch_dtype=torch.float16).to(device)
    pipe.safety_checker = None
    vae, unet, scheduler = pipe.vae, pipe.unet, pipe.scheduler

    img = Image.open(input_image_path).convert("RGB").resize((512, 512))
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device=device, dtype=torch.float16)
    with torch.no_grad():
        latents = vae.encode(img_tensor).latent_dist.sample() * vae.config.scaling_factor

    scheduler.set_timesteps(num_inference_steps)
    init_timestep = int(strength * (num_inference_steps - 1))
    t = scheduler.timesteps[init_timestep]
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, t)

    inputs = processor(images=img, return_tensors="pt")
    img_inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_feats = clip_model.get_image_features(**img_inputs)
        mapped = mapper(img_feats)

    uncond_tokens = tokenizer([""], padding="max_length", truncation=True,
                              max_length=tokenizer.model_max_length, return_tensors="pt")
    uncond_emb = text_model(**{k: v.to(device) for k, v in uncond_tokens.items()}).last_hidden_state
    mapped_emb_cat = torch.cat([uncond_emb, mapped], dim=0)

    latents = noisy_latents
    for i, t in enumerate(scheduler.timesteps[init_timestep:]):
        latent_in = torch.cat([latents] * 2).to(dtype=torch.float16)
        with torch.no_grad():
            model_out = unet(latent_in, t, encoder_hidden_states=mapped_emb_cat.to(dtype=torch.float16)).sample
        eps_uncond, eps_cond = model_out.chunk(2)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        latents = scheduler.step(eps, t, latents).prev_sample

    with torch.no_grad():
        latents = latents / vae.config.scaling_factor
        dec = vae.decode(latents).sample
    dec = (dec / 2 + 0.5).clamp(0, 1)
    img_out = transforms.ToPILImage()(dec[0].cpu())
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "variation.png")
    img_out.save(out_path)
    print(f"Saved generated variation: {out_path}")
    return out_path


# ==========================================================
# CLI and launcher
# ==========================================================
def parse_cli():
    p = argparse.ArgumentParser(description="Run mapper training or generation")
    p.add_argument("--mode", choices=["train", "gen"], required=True)
    p.add_argument("--dataset", help="Path to dataset CSV (for train mode)")
    p.add_argument("--prefix", default="laion_subset", help="Batch prefix when scanning a directory")
    p.add_argument("--image_path_prefix", default=None)
    p.add_argument("--out_dir", default="./mapper_ckpt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=4096)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--mapper", help="Path to trained mapper (.pth) for gen mode")
    p.add_argument("--input_image", help="Input image path for gen mode")
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--strength", type=float, default=0.7)
    return p.parse_args()


def main():
    cli = parse_cli()
    if cli.mode == "train":
        n_gpus = torch.cuda.device_count()
        print(f"Launching DDP training on {n_gpus} GPUs")
        mp.spawn(
            train_mapper_ddp,
            args=(
                n_gpus,
                cli.dataset,
                cli.clip_model,
                cli.out_dir,
                cli.epochs,
                cli.batch_size,
                cli.lr,
                cli.hidden_dim,
                cli.num_layers,
                "laion_subset",
                cli.image_path_prefix,
            ),
            nprocs=n_gpus,
            join=True,
        )
    elif cli.mode == "gen":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_variation(
            mapper_path=cli.mapper,
            clip_model_name=cli.clip_model,
            sd_model_name=cli.sd_model,
            input_image_path=cli.input_image,
            out_dir=cli.out_dir,
            device=device,
            num_inference_steps=cli.num_inference_steps,
            guidance_scale=cli.guidance,
            strength=cli.strength,
        )


if __name__ == "__main__":
    main()
