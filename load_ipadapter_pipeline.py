#!/usr/bin/env python3
"""
Load Stable Diffusion pipeline and IP-Adapter weights, then compute IP-Adapter image embeddings.

This script follows the pattern used in the `ip_adapter.ipynb` notebook and the
`diffusers` IP-Adapter utilities.

Usage:
  python load_ipadapter_pipeline.py --image /path/to/image.png --adapter h94/IP-Adapter

Outputs:
  - Prints the shape of the generated IP-Adapter embeddings
  - Optionally saves the embeddings to a .pt file using --out

Note: Requires `diffusers` with IP-Adapter support (recent versions). Run in the same
conda env where you run your pipeline (ie643) to ensure compatibility.
"""

import argparse
from pathlib import Path
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image


def load_pipeline_and_ipadapter(sd_model: str, adapter_id: str, device: torch.device, torch_dtype=None, subfolder: str = "models", weight_name: str = "ip-adapter_sd15.bin"):
    """Load a Stable Diffusion text2image pipeline and attach an IP-Adapter.

    Returns: pipeline
    """
    print(f"Loading pipeline: {sd_model} (dtype={torch_dtype})")
    pipe = AutoPipelineForText2Image.from_pretrained(sd_model, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    print(f"Loading IP-Adapter: {adapter_id} (subfolder={subfolder}, weight_name={weight_name})")
    # The AutoPipelineForText2Image exposes a convenience method to load IP-Adapter weights
    pipe.load_ip_adapter(adapter_id, subfolder=subfolder, weight_name=weight_name)

    return pipe


def compute_ipadapter_embeds(pipe, image_path: str, device: torch.device, num_images_per_prompt: int = 1, do_classifier_free_guidance: bool = True):
    """Compute IP-Adapter image embeddings for a PIL image path using the pipeline helper.

    Returns: list of tensors (one per prompt / image slot)
    """
    pil_img = load_image(image_path)
    print(f"Loaded image: {image_path}, size={pil_img.size}")

    # The pipeline helper returns a list/tuple of embeds suitable for passing into the denoiser
    image_embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=pil_img,
        ip_adapter_image_embeds=None,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    return image_embeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--adapter", default="h94/IP-Adapter", help="Hugging Face id or local path of the IP-Adapter package")
    parser.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model id")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu). If not set, auto-select GPU if available")
    parser.add_argument("--out", default=None, help="Optional path to save embeddings (.pt)")
    parser.add_argument("--subfolder", default="models", help="subfolder inside the adapter repo where weights are stored")
    parser.add_argument("--weight_name", default="ip-adapter_sd15.bin", help="Filename of the IP-Adapter binary weights")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    pipe = load_pipeline_and_ipadapter(args.sd_model, args.adapter, device=device, torch_dtype=torch_dtype, subfolder=args.subfolder, weight_name=args.weight_name)

    image_embeds = compute_ipadapter_embeds(pipe, args.image, device=device)

    print("Got IP-Adapter embeddings. Length:", len(image_embeds))
    for i, e in enumerate(image_embeds):
        try:
            print(f"embed[{i}] shape: {tuple(e.shape)}, dtype={e.dtype}, device={e.device}")
        except Exception:
            print(f"embed[{i}] type: {type(e)}")

    if args.out:
        out_path = Path(args.out)
        torch.save(image_embeds, out_path)
        print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
