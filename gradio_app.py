import tempfile
from pathlib import Path
from typing import List

import gradio as gr
import torch
from PIL import Image

from run_mapper import generate_variation as run_generate_variation
# import internal helpers to warm/load checkpoint weights
from run_mapper import _load_mapper, _restore_clip_text_from_ckpt
from transformers import CLIPModel, CLIPTextModel

# configuration
CLIP_NAME = "openai/clip-vit-large-patch14"
SD_NAME = "runwayml/stable-diffusion-v1-5"
MAPPER_PATH = Path("/home1/koustav/Image_to_Image_Diffusion/mapper_ckpt_new/mapper_epoch22.pth")
OUT_DIR = Path("/home1/koustav/Image_to_Image_Diffusion/gradio_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print("Warming models: loading mapper checkpoint and CLIP/text (cpu)...")
    if MAPPER_PATH.exists():
        try:
            _load_mapper(str(MAPPER_PATH), device=torch.device("cpu"))
        except Exception as e:
            print(f"Warning: failed to load mapper for warm-up: {e}")

        # instantiate CLIP models to populate HF cache and optionally restore finetuned weights
        try:
            clip_warm = CLIPModel.from_pretrained(CLIP_NAME)
            text_warm = CLIPTextModel.from_pretrained(CLIP_NAME)
            try:
                _restore_clip_text_from_ckpt(str(MAPPER_PATH), clip_warm, text_warm, device=torch.device("cpu"))
                print("Restored CLIP/text weights from mapper checkpoint (warm).")
            except Exception:
                # not fatal — checkpoint may not contain clip/text
                pass
        except Exception as e:
            print(f"Warning: could not instantiate CLIP models for warm-up: {e}")
    else:
        print(f"Warning: mapper checkpoint not found at {MAPPER_PATH}; first request will load models.")
except Exception as e:
    print(f"Model warm-up skipped: {e}")


def _call_pipeline_and_load(
    image: Image.Image,
    # prompt: str,
    guidance: float,
    strength: float,
    steps: int,
    variations: int,
    seed: int,
    use_source_latents: bool = False,
    use_checkpoint_clip: bool = True,
) -> List[Image.Image]:
    if image is None:
        return []

    # save uploaded PIL to a temp file for the pipeline wrapper
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
        tmp_path = tmpf.name
        image.convert("RGB").save(tmp_path)

    out_paths = run_generate_variation(
        mapper_path=str(MAPPER_PATH),
        clip_model_name=CLIP_NAME,
        sd_model_name=SD_NAME,
        input_image_path=tmp_path,
        out_dir=str(OUT_DIR),
        device=DEVICE,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        seed=int(seed),
        use_source_latents=bool(use_source_latents),
        variations=int(variations),
        use_checkpoint_clip=bool(use_checkpoint_clip),
    )

    # open generated images and return as list of PIL images
    outputs: List[Image.Image] = []
    for p in out_paths:
        try:
            img = Image.open(p).convert("RGB")
            outputs.append(img)
        except Exception:
            continue

    return outputs


# --- gradio UI ---------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Image-to-Image Diffusion")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Input image")
        gallery = gr.Gallery(label="Generated variants", columns=2, height=512)

    # prompt = gr.Textbox(label="Optional text prompt", value="")
    # prompt = None
    with gr.Row():
        guidance = gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Guidance scale")
        strength = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="Strength")
        steps = gr.Slider(10, 75, value=50, step=1, label="Inference steps")
    with gr.Row():
        variations = gr.Slider(1, 6, value=3, step=1, label="Number of variants")
        seed = gr.Number(value=42, precision=0, label="Seed")

    use_src_latents = gr.Checkbox(
        label="Use source latents (img2img)",
        value=False,
        info="If enabled, encode the input image to latents (img2img). Otherwise start from random latents (more diverse outputs).",
    )

    use_ckpt_clip = gr.Checkbox(
        label="Use CLIP/Text weights from mapper checkpoint",
        value=True,
        info="If enabled, attempt to load CLIP/text weights stored inside the mapper checkpoint for generation.",
    )

    run_btn = gr.Button("Generate")
    run_btn.click(
        fn=_call_pipeline_and_load,
        inputs=[image_input, guidance, strength, steps, variations, seed, use_src_latents, use_ckpt_clip],
        outputs=gallery,
    )

if __name__ == "__main__":
    demo.launch(share=True)
