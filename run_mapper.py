import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline
import wandb
from mapper_model import ImageToTextMapper
from laion_dataset import ImageCaptionDataset
from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter


def collate_batch(batch: List[Tuple[Image.Image, str]]):
    images, captions = zip(*batch)
    return list(images), list(captions)

def init_models(clip_model_name: str, device: torch.device, device_ids=None):
    """Load CLIP components and return processor, clip_model, text_model, tokenizer.

    We use:
      - CLIPModel (for get_image_features)
      - CLIPTokenizer + CLIPTextModel (for per-token text embeddings targets)

    The function also runs a single dummy pass to infer dims.
    """
    print(f"Loading CLIP model: {clip_model_name}")
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    
    # Load CLIP image encoder and place it on the target device.
    # Do NOT wrap CLIPModel in DataParallel because DataParallel does not
    # proxy custom methods like `get_image_features` — that causes
    # AttributeError when calling clip_model.get_image_features.
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    # Load CLIP text encoder and place it on the target device. Also avoid
    # wrapping in DataParallel for the same reason as above (we call
    # text_model(...) and access last_hidden_state).
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

def send_to_gpus(model, device, device_ids=None):
    # Wrap model in DataParallel only when CUDA and multiple GPUs are available.
    if str(device).startswith("cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # device_ids can be provided as a list of ints; otherwise use all available GPUs
        if device_ids:
            # ensure ints
            dev_ids = [int(x) for x in device_ids]
        else:
            dev_ids = list(range(torch.cuda.device_count()))
        try:
            model = nn.DataParallel(model, device_ids=dev_ids)
        except Exception:
            # fallback: don't wrap
            pass
    model.to(device)
    return model

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
    prefix: str = "laion_subset",
    device_ids=None,
    tb_hist_freq: int = 50,
):
    os.makedirs(out_dir, exist_ok=True)

    # TensorBoard writer
    tb_logdir = os.path.join(out_dir, "tb_logs")
    writer = SummaryWriter(tb_logdir)

    # Initialize wandb
    wandb.init(
        project="image-to-text-mapper",
        config={
            "dataset_csv": dataset_csv,
            "clip_model_name": clip_model_name,
            "out_dir": out_dir,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "prefix": prefix,
        }
    )

    # init models
    processor, clip_model, tokenizer, text_model, in_dim, out_seq_len, out_dim = init_models(clip_model_name, device, device_ids=device_ids)

    # create mapper
    base_mapper = ImageToTextMapper(in_dim=in_dim, out_seq_len=out_seq_len, out_dim=out_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    # write model graph to TensorBoard (before DataParallel wrapping)
    try:
        base_mapper.to(device)
        dummy_input = torch.randn(1, in_dim, device=device)
        writer.add_graph(base_mapper, dummy_input)
        print(f"✅ TensorBoard: model graph written to {tb_logdir}")
    except Exception as e:
        print(f"⚠️ Could not write model graph to TensorBoard: {e}")

    mapper = base_mapper
    # Use DataParallel if requested/available
    if device_ids and len(device_ids) > 1:
        print(f"Using device_ids={device_ids} for DataParallel")
        try:
            mapper = nn.DataParallel(mapper, device_ids=device_ids)
        except Exception as e:
            print(f"Warning: failed to wrap mapper in DataParallel: {e}")
    elif torch.cuda.device_count() > 1 and (not device_ids):
        # default: use all GPUs
        devs = list(range(torch.cuda.device_count()))
        print(f"Using all available GPUs for DataParallel: {devs}")
        try:
            mapper = nn.DataParallel(mapper, device_ids=devs)
        except Exception as e:
            print(f"Warning: failed to wrap mapper in DataParallel: {e}")
    mapper = mapper.to(device)

    # dataset + dataloader
    ds = ImageCaptionDataset(dataset_csv, prefix=prefix)
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
            # Ensure images are RGB
            images = [img.convert("RGB") for img in images]
            # 1) image -> image pooled features
            inputs = processor(
                images=images,
                return_tensors="pt",
                do_normalize=True,
                do_resize=True,
                do_center_crop=True,
                # Explicitly set mean/std for RGB
                # These are the defaults for CLIP RGB models
                # If you use a grayscale model, set mean=[0.5], std=[0.5]
                # For RGB:
                # mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                # If you still get errors, try mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
                # But the below should work for openai/clip-vit-large-patch14
                # If you want to override:
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            img_inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                img_feats = clip_model.get_image_features(**img_inputs)  # [B, in_dim]

            # Log input features and shape (throttled)
            try:
                step_idx = epoch * 100000 + total_batches
                if tb_hist_freq and (total_batches % tb_hist_freq == 0):
                    writer.add_histogram('input/img_feats', img_feats.cpu().detach(), global_step=step_idx)
                writer.add_text('shapes', f'img_feats: {tuple(img_feats.shape)} -> expected in_dim: {in_dim}', global_step=step_idx)
            except Exception:
                pass

            # 2) captions -> tokenized -> text encoder outputs (target sequence)
            tokenized = tokenizer(list(captions), padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)
            with torch.no_grad():
                text_outs = text_model(input_ids=input_ids, attention_mask=attention_mask)
                text_embeds = text_outs.last_hidden_state  # [B, L, H]

            # 3) mapper prediction
            pred = mapper(img_feats)  # [B, L, H]

            # Log prediction histogram and shape (throttled)
            try:
                step_idx = epoch * 100000 + total_batches
                if tb_hist_freq and (total_batches % tb_hist_freq == 0):
                    writer.add_histogram('output/pred', pred.cpu().detach(), global_step=step_idx)
                writer.add_text('shapes', f'pred: {tuple(pred.shape)} -> expected (B, {out_seq_len}, {out_dim})', global_step=step_idx)
            except Exception:
                pass

            # 4) compute masked MSE loss (only over tokens where attention_mask == 1)
            diff2 = (pred - text_embeds).pow(2).mean(dim=-1)  # [B, L]
            masked = diff2 * attention_mask
            loss = masked.sum() / (attention_mask.sum() + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            optimizer.step()
            # Log loss to TensorBoard
            try:
                writer.add_scalar('loss/train_masked_mse', loss.item(), epoch * 100000 + total_batches)
            except Exception:
                pass

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / max(1, total_batches)
        print(f"Epoch {epoch}/{epochs} — avg masked MSE loss: {avg_loss:.6f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss
        })

        # save checkpoint
        ckpt_path = os.path.join(out_dir, f"mapper_epoch{epoch}.pth")
        torch.save({"state_dict": mapper.state_dict(), "config": {"in_dim": in_dim, "out_seq_len": out_seq_len, "out_dim": out_dim}}, ckpt_path)
        print(f"Saved mapper checkpoint to {ckpt_path}")

        # Optionally log checkpoint as artifact
        wandb.save(ckpt_path)

    # final save
    final_path = os.path.join(out_dir, "mapper.pth")
    torch.save({"state_dict": mapper.state_dict(), "config": {"in_dim": in_dim, "out_seq_len": out_seq_len, "out_dim": out_dim}}, final_path)
    print(f"Saved final mapper to {final_path}")
    wandb.save(final_path)
    wandb.finish()
    # close TensorBoard writer
    try:
        writer.flush()
        writer.close()
    except Exception:
        pass
    return final_path

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
    state_dict = data["state_dict"]
    # Remove 'module.' prefix if present (from DataParallel)
    if any(k.startswith("module.") for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1)
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    mapper.load_state_dict(state_dict)
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
    # FIX: convert to float16 for VAE
    img_tensor = preprocess(init_image).unsqueeze(0).to(device=device, dtype=torch.float16)

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
    uncond_tokens = tokenizer(["add green color shades to the eye"], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
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
        latent_model_input = torch.cat([latents] * 2).to(dtype=torch.float16)

        # predict noise
        with torch.no_grad():
            model_out = unet(latent_model_input, t, encoder_hidden_states=mapped_emb_cat.to(dtype=torch.float16)).sample

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

    # Log generated image to TensorBoard if writer exists in scope (best-effort)
    try:
        # Create a temporary writer if the module didn't have one
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(os.path.dirname(out_dir), "tb_logs"))
        tb.add_image('gen/variation', transforms.ToTensor()(image), dataformats='CHW')
        tb.flush(); tb.close()
    except Exception:
        pass
    print(f"Saved variation to {out_path}")
    return out_path



def get_args(args_dict=None, **kwargs):
    """Return args as an object from a dict or kwargs."""
    class Args:
        pass
    args = Args()
    if args_dict:
        for k, v in args_dict.items():
            setattr(args, k, v)
    for k, v in kwargs.items():
        setattr(args, k, v)
    # Set defaults if not provided
    if not hasattr(args, "clip_model"):
        args.clip_model = "openai/clip-vit-large-patch14"

    # device should be a string like 'cuda', 'cuda:0' or 'cpu'
    if not hasattr(args, "device") or args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU selection flags: allow None or explicit values
    if not hasattr(args, "device_ids"):
        args.device_ids = None
    if not hasattr(args, "gpus"):
        args.gpus = None

    if not hasattr(args, "out_dir"):
        args.out_dir = "./mapper_ckpt"
    if not hasattr(args, "epochs"):
        args.epochs = 3
    if not hasattr(args, "batch_size"):
        args.batch_size = 8
    if not hasattr(args, "lr"):
        args.lr = 1e-4
    if not hasattr(args, "hidden_dim"):
        args.hidden_dim = 4096
    if not hasattr(args, "num_layers"):
        args.num_layers = 3
    if not hasattr(args, "sd_model"):
        args.sd_model = "runwayml/stable-diffusion-v1-5"
    if not hasattr(args, "num_inference_steps"):
        args.num_inference_steps = 50
    if not hasattr(args, "guidance"):
        args.guidance = 3.0
    if not hasattr(args, "strength"):
        args.strength = 0.7
    return args


def parse_cli():
    p = argparse.ArgumentParser(description="Run mapper training or generation")
    p.add_argument("--mode", choices=["train", "gen"], required=True)
    # training args
    p.add_argument("--dataset", help="Path to dataset directory or file (for train mode)")
    p.add_argument("--prefix", default="laion_subset", help="Batch prefix when scanning a directory")
    p.add_argument("--out_dir", default="./mapper_ckpt", help="Output directory for checkpoints / outputs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=4096)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    # GPU selection: either a short form --gpus (number) or explicit --device_ids '0,1'
    # Accept as string to avoid argparse int-conversion errors when user passes an empty string
    p.add_argument("--gpus", type=str, default=None, help="Number of GPUs to use (optional). Provide an integer (e.g. '2') or leave unset.")
    p.add_argument("--device_ids", type=str, default=None, help="Comma-separated list of GPU device ids to use, e.g. '0,1,2'")
    # generation args
    p.add_argument("--mapper", help="Path to trained mapper (.pth) (for gen mode)")
    p.add_argument("--input_image", help="Input image path (for gen mode)")
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--strength", type=float, default=0.7)
    p.add_argument("--device", default=None, help="Device string like 'cuda:0' or 'cpu' (optional)")
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_cli()
    # Build args object expected by run_main using get_args helper from notebook.
    args_dict = {
        "mode": cli.mode,
        "dataset": cli.dataset,
        "prefix": cli.prefix,
        "out_dir": cli.out_dir,
        "epochs": cli.epochs,
        "batch_size": cli.batch_size,
        "lr": cli.lr,
        "hidden_dim": cli.hidden_dim,
        "num_layers": cli.num_layers,
        "clip_model": cli.clip_model,
        "mapper": cli.mapper,
        "input_image": cli.input_image,
        "sd_model": cli.sd_model,
        "num_inference_steps": cli.num_inference_steps,
        "guidance": cli.guidance,
        "strength": cli.strength,
    }
    # Forward device and GPU selection flags into args
    if cli.device:
        args_dict["device"] = cli.device
    if getattr(cli, 'device_ids', None):
        # forward non-empty device_ids string
        devs = str(cli.device_ids).strip()
        if devs != "":
            args_dict["device_ids"] = devs
    # normalize gpus: accept None or a numeric string; ignore blank strings
    if getattr(cli, 'gpus', None) is not None:
        g = str(cli.gpus).strip()
        if g != "":
            try:
                args_dict["gpus"] = int(g)
            except ValueError:
                raise ValueError(f"--gpus must be an integer (got {cli.gpus!r})")

    args = get_args(args_dict)
    device = torch.device(args.device)

    # Parse device ids
    device_ids = None
    if getattr(args, 'device_ids', None):
        device_ids = [int(x.strip()) for x in args.device_ids.split(',') if x.strip()]
    elif getattr(args, 'gpus', None):
        # use first N GPUs
        device_ids = list(range(args.gpus))

    if getattr(args, "mode", None) == "train":
        if not hasattr(args, "dataset"):
            raise ValueError("'dataset' is required for training mode")
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
            device_ids=device_ids,
        )
    elif getattr(args, "mode", None) == "gen":
        if not hasattr(args, "mapper") or not hasattr(args, "input_image"):
            raise ValueError("'mapper' and 'input_image' are required for gen mode")
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
    else:
        raise ValueError("Unknown mode. Set args.mode to 'train' or 'gen'.")