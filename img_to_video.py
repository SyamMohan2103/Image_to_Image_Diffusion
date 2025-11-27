import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import List
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline
import wandb
from torch.utils.tensorboard import SummaryWriter
from diffusers import TextToVideoSDPipeline

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
from diffusers.utils import export_to_video

import gradio as gr
import torch
from PIL import Image

from run_mapper import generate_variation as run_generate_variation
from run_mapper import _load_mapper, _restore_clip_text_from_ckpt, clip_get_image_features
from transformers import CLIPModel, CLIPTextModel, CLIPProcessor, CLIPTokenizer


CLIP_NAME = "openai/clip-vit-large-patch14"
SD_NAME = "runwayml/stable-diffusion-v1-5"
MAPPER_PATH = Path("/home/subhankar/koustav/Image_to_Image_Diffusion/mapper_model_1024/mapper_epoch4.pth")
OUT_DIR = Path("/home/subhankar/koustav/Image_to_Image_Diffusion/gradio_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _prepare_image(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((512, 512))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    t = preprocess(img).unsqueeze(0)  # [1,C,H,W]
    return img, t

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
    use_source_latents: bool = False,  # keep default for backward compat
    variations: int = 3,               # <-- added parameter
    use_checkpoint_clip: bool = False,
) -> List[str]:
    """
    Generate image variation(s) using the Stable Diffusion pipeline.
    The mapper is used only to produce prompt_embeds conditioning for the pipeline.
    Returns list of saved file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    # device = torch.device(device)

    # choose pipeline dtype (float16 on CUDA for speed)
    # pipe_dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe_dtype = torch.float16

    # load CLIP and text models (kept in float32 for stability)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    # text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device).eval()
    # for p in list(clip_model.parameters()) + list(text_model.parameters()):
        # p.requires_grad = False

    # Optionally restore fine-tuned CLIP/text weights from the checkpoint if provided
    if use_checkpoint_clip:
        try:
            restored = _restore_clip_text_from_ckpt(mapper_path, clip_model, text_model, device)
            if restored and is_main_process():
                print("Using CLIP/Text weights restored from checkpoint for generation.")
        except Exception as e:
            if is_main_process():
                print(f"Warning: could not restore CLIP/text from checkpoint: {e}")

    # load mapper checkpoint
    if not os.path.exists(mapper_path):
        raise FileNotFoundError(f"Mapper checkpoint not found: {mapper_path}")
    mapper, cfg = _load_mapper(mapper_path, device=torch.device("cpu"))
    mapper.to(device).eval()

    # load pipeline
    pipe = StableDiffusionPipeline.from_pretrained('ali-vilab/text-to-video-ms-1.7b', torch_dtype=pipe_dtype).to(device)
    text_model = pipe.text_encoder.to(device).eval()
    for p in list(clip_model.parameters()) + list(text_model.parameters()):
        p.requires_grad = False
    pipe.safety_checker = None
    vae, unet, scheduler = pipe.vae, pipe.unet, pipe.scheduler

    # prepare image tensors
    pil_img, img_tensor = _prepare_image(input_image_path)
    # img_tensor = img_tensor.to(device=device, dtype=torch.float32)  # for CLIP/mapper

    # compute mapped conditioning and uncond embedding
    with torch.no_grad():
        # Use same preprocessing as training (resize/center-crop/normalize)
        # across transformers versions: some image processors don't accept extra kwargs
        try:
            clip_inputs = processor(images=[pil_img], return_tensors="pt", do_normalize=True, do_resize=True, do_center_crop=True)
        except Exception:
            try:
                clip_inputs = processor(images=[pil_img], return_tensors="pt")
            except Exception:
                # Fall back to calling image_processor / feature_extractor directly
                img_proc = getattr(processor, "image_processor", None) or getattr(processor, "feature_extractor", None)
                if img_proc is None:
                    raise
                clip_inputs = img_proc(images=[pil_img], return_tensors="pt")
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
        # safe call in case clip_model was wrapped in DDP
        img_feats = clip_get_image_features(clip_model, **clip_inputs)  # float32
        mapped = mapper(img_feats)  # [1, L, H] float32

        uncond_tokens = tokenizer([""], padding="max_length", truncation=True,
                                  max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_tokens = {k: v.to(device) for k, v in uncond_tokens.items()}
        uncond_emb = text_model(**uncond_tokens).last_hidden_state  # [1, L, H] float32

    # align dtype/device for pipeline UNet
    target_dtype = unet.dtype
    # Ensure contiguous, correct device and dtype expected by the pipeline
    mapped = mapped.contiguous().to(device=device, dtype=target_dtype)
    uncond_emb = uncond_emb.contiguous().to(device=device, dtype=target_dtype)

    return mapped, uncond_emb


if __name__ == "__main__":


    mapped, negative_embeds = generate_variation(
        mapper_path=str(MAPPER_PATH),
        clip_model_name=CLIP_NAME,
        sd_model_name=SD_NAME,
        input_image_path="/home/subhankar/koustav/Image_to_Image_Diffusion/wind.webp",
        out_dir=str(OUT_DIR),
        device=torch.device("cuda:3"),
        num_inference_steps=50,
        guidance_scale=3.0,
        strength=0.7,
        seed=42,
        use_source_latents=False,  # keep default for backward compat
        variations=3,               # <-- added parameter
        use_checkpoint_clip=False,
    )

    pipe = TextToVideoSDPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16"
    )
    # pipe.enable_model_cpu_offload()
    pipe.to("cuda:3")

    # # Load the text-to-video model
    # model_id = "cerspense/zeroscope_v2_576w"  # Text-to-video model
    # pipe = DiffusionPipeline.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.float16,
    # )
    # pipe.to("cuda:3")

    # # Optional: Use faster scheduler
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # print(f"Custom embedding shape: {mapped.shape}")

    # # For classifier-free guidance, also convert negative embedding
    # # negative_embedding = torch.randn(1, 77, 1024, dtype=torch.float16, device="cuda:2")
    # # negative_embedding = torch.cat([negative_1024_embedding], dim=-1)

    # # Generate video using custom embeddings
    video_frames = pipe(
        # prompt="Spiderman is surfing",
        prompt_embeds=mapped,
        # negative_prompt_embeds=negative_embeds,
        # num_inference_steps=25,
        # num_frames=24,
        # height=256,
        # width=256,
        # guidance_scale=9.0,
    ).frames[0]

    # Save video
    export_to_video(video_frames, "wind1.mp4")

    print("Video generated successfully!")
    print(f"Video shape: {len(video_frames)} frames")