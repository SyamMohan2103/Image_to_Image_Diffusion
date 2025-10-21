import math
from pathlib import Path
from typing import List

import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline

from mapper_model import ImageToTextMapper
from run_mapper import collate_batch  # reuse dtype helpers if needed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

# --- model bootstrap ---------------------------------------------------------
CLIP_NAME = "openai/clip-vit-large-patch14"
SD_NAME = "runwayml/stable-diffusion-v1-5"
MAPPER_PATH = Path("/home1/koustav/Image_to_Image_Diffusion/mapper_ckpt/mapper_final.pth") 

processor = CLIPProcessor.from_pretrained(CLIP_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_NAME).to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME)
text_model = CLIPTextModel.from_pretrained(CLIP_NAME).to(DEVICE)

clip_model.eval()
text_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
for p in text_model.parameters():
    p.requires_grad = False

mapper_ckpt = torch.load(MAPPER_PATH, map_location=DEVICE)
mapper_cfg = mapper_ckpt["config"]
mapper = ImageToTextMapper(
    in_dim=mapper_cfg["in_dim"],
    out_seq_len=mapper_cfg["out_seq_len"],
    out_dim=mapper_cfg["out_dim"],
    hidden_dim=mapper_cfg.get("hidden_dim", 4096),
    num_layers=mapper_cfg.get("num_layers", 3),
).to(DEVICE)
state_dict = mapper_ckpt["state_dict"]
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
mapper.load_state_dict(state_dict)
mapper.eval()

pipe = StableDiffusionPipeline.from_pretrained(
    SD_NAME,
    torch_dtype=TORCH_DTYPE,
).to(DEVICE)
pipe.safety_checker = None  # optional

to_latents = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# --- generation core ---------------------------------------------------------
@torch.inference_mode()
def generate_variants(
    image: Image.Image,
    prompt: str,
    guidance: float,
    strength: float,
    steps: int,
    variations: int,
    seed: int,
) -> List[Image.Image]:
    if image is None:
        return []

    image = image.convert("RGB")
    init_tensor = to_latents(image).unsqueeze(0).to(device=DEVICE, dtype=TORCH_DTYPE)

    latents = pipe.vae.encode(init_tensor).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor

    # mapper conditioning
    processed = processor(images=image, return_tensors="pt")
    processed = {k: v.to(DEVICE) for k, v in processed.items()}
    img_feats = clip_model.get_image_features(**processed)
    mapped = mapper(img_feats)  # [1, L, H]

    cond_tokens = tokenizer(
        [prompt or ""], padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt",
    )
    cond_emb = text_model(
        input_ids=cond_tokens.input_ids.to(DEVICE),
        attention_mask=cond_tokens.attention_mask.to(DEVICE),
    ).last_hidden_state

    uncond_tokens = tokenizer(
        [""], padding="max_length", truncation=True,
        max_length=tokenizer.model_max_length, return_tensors="pt",
    )
    uncond_emb = text_model(
        input_ids=uncond_tokens.input_ids.to(DEVICE),
        attention_mask=uncond_tokens.attention_mask.to(DEVICE),
    ).last_hidden_state

    target_dtype = pipe.unet.dtype
    latents = latents.to(dtype=target_dtype)
    mapped = mapped.to(dtype=target_dtype)
    cond_emb = cond_emb.to(dtype=target_dtype)
    uncond_emb = uncond_emb.to(dtype=target_dtype)

    pipe.scheduler.set_timesteps(steps)
    init_timestep = int(strength * (steps - 1))
    init_timestep = min(max(init_timestep, 0), steps - 1)
    t_init = pipe.scheduler.timesteps[init_timestep]

    outputs: List[Image.Image] = []
    for i in range(variations):
        torch.manual_seed(seed + i)
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, t_init)
        latent = noisy_latents

        for t in pipe.scheduler.timesteps[init_timestep:]:
            latent_input = torch.cat([latent] * 2)
            model_out = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=torch.cat([uncond_emb, mapped], dim=0),
            ).sample
            eps_uncond, eps_cond = model_out.chunk(2)
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)
            latent = pipe.scheduler.step(eps, t, latent).prev_sample

        decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        img = decoded.cpu().permute(0, 2, 3, 1)[0].numpy()
        outputs.append(Image.fromarray((img * 255).astype("uint8")))

    return outputs


# --- gradio UI ---------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Image-to-Image Variants with Mapper")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input image")
        gallery = gr.Gallery(label="Generated variants", columns=[2], height=512)

    prompt = gr.Textbox(label="Optional text prompt", value="")
    with gr.Row():
        guidance = gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Guidance scale")
        strength = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Strength")
        steps = gr.Slider(10, 75, value=50, step=1, label="Inference steps")
    with gr.Row():
        variations = gr.Slider(1, 6, value=3, step=1, label="Number of variants")
        seed = gr.Number(value=42, precision=0, label="Seed")

    run_btn = gr.Button("Generate")

    run_btn.click(
        fn=generate_variants,
        inputs=[image_input, prompt, guidance, strength, steps, variations, seed],
        outputs=gallery,
    )

if __name__ == "__main__":
    demo.launch(share=True)