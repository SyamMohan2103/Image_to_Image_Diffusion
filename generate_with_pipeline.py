import os
import argparse
from pathlib import Path
from typing import List

from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline

from mapper_model import ImageToTextMapper

def _load_mapper(mapper_path: str, device: torch.device):
    data = torch.load(mapper_path, map_location="cpu")
    cfg = data["config"]
    mapper = ImageToTextMapper(in_dim=cfg["in_dim"], out_seq_len=cfg["out_seq_len"], out_dim=cfg["out_dim"])
    state = data["state_dict"]
    # strip DataParallel prefix if present
    if any(k.startswith("module.") for k in list(state.keys())):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    mapper.load_state_dict(state)
    mapper.to(device).eval()
    return mapper, cfg

def _prepare_image(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((512, 512))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    t = preprocess(img).unsqueeze(0)  # [1,C,H,W]
    return img, t

def generate_with_pipeline(
    mapper_path: str,
    clip_model_name: str,
    sd_model_name: str,
    input_image_path: str,
    out_dir: str,
    device: torch.device,
    num_inference_steps: int = 50,
    guidance_scale: float = 3.0,
    strength: float = 0.7,
    variations: int = 3,
    seed: int = 42,
    use_source_latents: bool = False,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    # dtype choices: keep CLIP/text/mapper in float32 and cast embeddings to pipe.unet.dtype later
    pipe_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # load models
    processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
    clip = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_model = CLIPTextModel.from_pretrained(clip_model_name).to(device).eval()
    for p in list(clip.parameters()) + list(text_model.parameters()):
        p.requires_grad = False

    mapper, cfg = _load_mapper(mapper_path, device=torch.device("cpu"))  # load to cpu then move below
    mapper.to(device).eval()

    pipe = StableDiffusionPipeline.from_pretrained(sd_model_name).to(device)
    pipe.safety_checker = None
    vae, unet, scheduler = pipe.vae, pipe.unet, pipe.scheduler

    # prepare image and latents
    pil_img, img_tensor = _prepare_image(input_image_path)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)  # CLIP/mapper float32

    with torch.no_grad():
        # CLIP image features -> mapper conditioning
        clip_inputs = processor(images=pil_img, return_tensors="pt")
        clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
        img_feats = clip.get_image_features(**clip_inputs)  # float32 on device
        mapped = mapper(img_feats)  # [1, L, H] float32

        # unconditional text embedding (empty prompt)
        uncond_tokens = tokenizer([""], padding="max_length", truncation=True,
                                  max_length=tokenizer.model_max_length, return_tensors="pt")
        uncond_tokens = {k: v.to(device) for k, v in uncond_tokens.items()}
        uncond_emb = text_model(**uncond_tokens).last_hidden_state  # [1, L, H] float32

    # align dtype/device for pipeline UNet
    target_dtype = unet.dtype
    mapped = mapped.to(device=device, dtype=target_dtype)
    uncond_emb = uncond_emb.to(device=device, dtype=target_dtype)

    # prepare latents: either encode source or random latents of same shape
    with torch.no_grad():
        # encode using VAE expects pipe dtype (vae may want float16)
        enc_in = img_tensor.to(device=device, dtype=pipe_dtype)
        latents_orig = vae.encode(enc_in).latent_dist.sample() * vae.config.scaling_factor  # [1, C, H/8, W/8]

    out_paths = []
    for i in range(variations):
        gen_seed = seed + i
        generator = torch.Generator(device=device).manual_seed(gen_seed)

        if use_source_latents:
            latents = latents_orig.clone().to(device=device, dtype=latents_orig.dtype)
        else:
            latents = torch.randn_like(latents_orig, device=device, dtype=latents_orig.dtype)

        # call pipeline using prompt_embeds and negative_prompt_embeds; pipeline will handle guidance
        images = pipe(
            prompt_embeds=mapped,
            negative_prompt_embeds=uncond_emb,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=latents,
            generator=generator,
            output_type="pil",
        ).images  # list of PIL images

        out_name = f"variation_seed{gen_seed}.png"
        out_path = os.path.join(out_dir, out_name)
        images[0].save(out_path)
        out_paths.append(out_path)

    return out_paths

def _cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapper", required=True)
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--input_image", required=True)
    p.add_argument("--out_dir", default="./gen_out")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--strength", type=float, default=0.7)  # currently unused (kept for parity)
    p.add_argument("--variations", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_source_latents", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_paths = generate_with_pipeline(
        mapper_path=args.mapper,
        clip_model_name=args.clip_model,
        sd_model_name=args.sd_model,
        input_image_path=args.input_image,
        out_dir=args.out_dir,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance,
        strength=args.strength,
        variations=args.variations,
        seed=args.seed,
        use_source_latents=args.use_source_latents,
    )
    for p in out_paths:
        print("Saved:", p)

if __name__ == "__main__":
    _cli_main()