#!/usr/bin/env python3
"""
Compare mapper outputs vs CLIP image features and (optionally) IPAdapter embeddings
for the same input image.

Saves a JSON summary with cosine and L2 distances and prints shapes for inspection.

Usage examples:
  python compare_mapper_ipadapter.py --mapper mapper_ckpt/mapper_best.pth --input_image input.jpg
  python compare_mapper_ipadapter.py --mapper mapper_ckpt/mapper_best.pth --input_image input.jpg --ipadapter_path /path/to/ipadapter

Notes:
- This script uses helpers from `run_mapper.py` if available (`_load_mapper`, `clip_get_image_features`).
- IPAdapter loading is best-effort: it will try a few common import paths. If your IPAdapter has a
  custom wrapper, you can adapt the small `load_ipadapter()` helper below.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# try to reuse helpers from run_mapper if present
try:
    from run_mapper import _load_mapper, clip_get_image_features
except Exception:
    _load_mapper = None
    clip_get_image_features = None


def _prepare_image(img_path: str, size=(512, 512)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    return img


def load_clip_and_processor(clip_name: str, device: torch.device):
    processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(clip_name).to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    return processor, clip_model


def try_load_ipadapter(ipadapter_path: str, device: torch.device):
    """Attempt to load an IPAdapter model from common entrypoints.

    Returns (model, loader_desc) or (None, None) if not available.
    """
    if ipadapter_path is None:
        return None, None
    # Try import paths
    try:
        # ip-adapter standalone package
        import ip_adapter as _ipa
        if hasattr(_ipa, "IPAdapterModel"):
            ModelCls = getattr(_ipa, "IPAdapterModel")
            model = ModelCls.from_pretrained(ipadapter_path).to(device).eval()
            return model, "ip_adapter.IPAdapterModel"
    except Exception:
        pass
    try:
        # Diffusers-flavored IPAdapter (if exists in this environment)
        from diffusers import IPAdapterModel as DiffIP
        model = DiffIP.from_pretrained(ipadapter_path).to(device).eval()
        return model, "diffusers.IPAdapterModel"
    except Exception:
        pass

    # Last resort: try to load a state dict and expose it raw (user must know how to interpret)
    if os.path.exists(ipadapter_path):
        try:
            sd = torch.load(ipadapter_path, map_location="cpu")
            return sd, "state_dict"
        except Exception:
            pass

    return None, None


def embed_with_ipadapter(adapter, adapter_desc, pil_img: Image.Image, device: torch.device):
    """Given an instantiated adapter or state_dict, return a tensor embedding or raise informative error.
    """
    if adapter is None:
        raise RuntimeError("No adapter provided")
    if isinstance(adapter, dict):
        raise RuntimeError("Adapter is a raw state_dict; please load it into your adapter class before using this script.")
    # Common API: try get_image_features or forward
    try:
        # Some adapters expect preprocessed pixels, some expect PIL. We'll try multiple common calls.
        if hasattr(adapter, "get_image_features"):
            with torch.no_grad():
                emb = adapter.get_image_features(images=pil_img) if not isinstance(pil_img, torch.Tensor) else adapter.get_image_features(images=pil_img)
            return emb
    except Exception:
        pass
    try:
        # some implementations use encode_image or encode
        if hasattr(adapter, "encode_image"):
            with torch.no_grad():
                emb = adapter.encode_image(pil_img)
            return emb
    except Exception:
        pass
    # last try: call adapter(pil_img)
    try:
        with torch.no_grad():
            out = adapter(pil_img)
        # if output is tuple/dict, try to extract common keys
        if isinstance(out, dict) and "last_hidden_state" in out:
            return out["last_hidden_state"]
        if isinstance(out, (list, tuple)):
            return out[0]
        return out
    except Exception as e:
        raise RuntimeError(f"Could not run adapter model; please adapt this script for your adapter. Error: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mapper", required=True, help="Path to mapper checkpoint (.pth)")
    p.add_argument("--input_image", required=True, help="Path to input image (512x512 preferred)")
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5", help="Stable Diffusion model id used when loading IP-Adapter via pipeline")
    p.add_argument("--ipadapter_path", default=None, help="Optional path/identifier for IPAdapter model (HF or local)")
    p.add_argument("--use_ipadapter_via_pipeline", action="store_true", help="If set, load IPAdapter via the Stable Diffusion pipeline helper (load_ipadapter_pipeline.py)")
    p.add_argument("--device", default=None)
    p.add_argument("--out_json", default="compare_results.json")
    p.add_argument("--fit_linear_proj", action="store_true", help="If set and dims differ, fit a ridge linear projection from mapper->clip using the current sample")
    p.add_argument("--proj_out", default=None, help="Optional path to save the learned linear projection (.pt)")
    p.add_argument("--ridge_lambda", type=float, default=1e-3, help="Regularization for ridge regression when fitting projection")
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # load mapper (use run_mapper helper if available)
    if _load_mapper is None:
        raise RuntimeError("run_mapper helper _load_mapper not found in this environment. Please run this script from the project root where run_mapper.py is available.")

    print("Loading mapper checkpoint (to CPU then moving to device)...")
    mapper, cfg = _load_mapper(args.mapper, device=torch.device("cpu"))
    mapper.to(device).eval()

    print("Loading CLIP model and processor...")
    processor, clip_model = load_clip_and_processor(args.clip_model, device)

    pil_img = _prepare_image(args.input_image)

    # CLIP preprocessing
    clip_inputs = processor(images=[pil_img], return_tensors="pt")
    clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}

    with torch.no_grad():
        # CLIP image features used as mapper input
        if clip_get_image_features is not None:
            img_feats = clip_get_image_features(clip_model, **clip_inputs)
        else:
            img_feats = clip_model.get_image_features(**clip_inputs)
        # Mapper output (sequence)
        mapped = mapper(img_feats.to(next(mapper.parameters()).device))

    # Reduce mapped sequence to a single vector for comparison (mean over tokens)
    mapped_mean = mapped.mean(dim=1)  # [1, hidden]
    # Ensure shapes
    img_feats_vec = img_feats
    if img_feats_vec.ndim > 2:
        img_feats_vec = img_feats_vec.reshape(img_feats_vec.shape[0], -1).mean(dim=1, keepdim=True)

    # Align dims: if mapped_mean has different hidden dim, try a linear projection? For now we assert equal dims.
    mapped_dim = mapped_mean.shape[-1]
    img_dim = img_feats_vec.shape[-1]

    results = {
        "mapper_checkpoint": str(args.mapper),
        "input_image": str(args.input_image),
        "clip_model": args.clip_model,
        "mapped_shape": list(mapped.shape),
        "img_feats_shape": list(img_feats.shape),
    }

    # compute simple metrics where possible
    try:
        # If dims differ, compute cosine between mapped_mean projected and img_feats via mean pooling
        if mapped_dim == img_dim:
            mm = F.normalize(mapped_mean, p=2, dim=-1)
            im = F.normalize(img_feats_vec, p=2, dim=-1)
            cosine = (mm @ im.T).cpu().numpy().tolist()
            l2 = (mapped_mean - img_feats_vec).norm(p=2, dim=-1).cpu().numpy().tolist()
            results.update({"cosine_mapped_img_feats": cosine, "l2_mapped_img_feats": l2})
        else:
            # dims differ — optionally fit a ridge linear projection from mapper -> clip using this sample
            if args.fit_linear_proj:
                try:
                    # Use CPU double precision for the small linear solve for stability
                    X = mapped_mean.detach().cpu().to(dtype=torch.float64)  # [N, d_m]
                    Y = img_feats_vec.detach().cpu().to(dtype=torch.float64)  # [N, d_i]
                    # Ridge solution: W = (X^T X + lambda I)^{-1} X^T Y
                    lambda_reg = float(args.ridge_lambda)
                    XtX = X.T @ X  # [d_m, d_m]
                    reg = lambda_reg * torch.eye(XtX.shape[0], dtype=XtX.dtype)
                    W = torch.linalg.solve(XtX + reg, X.T @ Y)  # [d_m, d_i]

                    # project mapped -> clip space
                    proj = (X @ W).to(dtype=img_feats_vec.dtype, device=device)
                    # compute metrics between projected mapped and img_feats_vec
                    proj_norm = F.normalize(proj, p=2, dim=-1)
                    im_norm = F.normalize(img_feats_vec, p=2, dim=-1)
                    cosine_proj = (proj_norm @ im_norm.T).cpu().numpy().tolist()
                    l2_proj = (proj.to(device) - img_feats_vec).norm(p=2, dim=-1).cpu().numpy().tolist()

                    results.update({
                        "cosine_mapped_img_feats": None,
                        "l2_mapped_img_feats": None,
                        "note": "mapped_dim != img_feats_dim; fitted linear projection from mapper->clip using current sample",
                        "cosine_proj_mapped_img_feats": cosine_proj,
                        "l2_proj_mapped_img_feats": l2_proj,
                        "proj_shape": list(W.shape),
                    })

                    # save projection if requested
                    if args.proj_out:
                        torch.save(W.to(dtype=torch.float32), args.proj_out)
                        results["proj_saved_to"] = args.proj_out
                except Exception as e:
                    results.update({"cosine_mapped_img_feats": None, "l2_mapped_img_feats": None, "proj_error": str(e)})
            else:
                results.update({"cosine_mapped_img_feats": None, "l2_mapped_img_feats": None, "note": "mapped_dim != img_feats_dim; consider projecting to same space before comparison"})
    except Exception as e:
        results["error_compute_metrics"] = str(e)

    # Optional: load ipadapter and compare
    if args.ipadapter_path:
        print(f"Attempting to load IPAdapter from: {args.ipadapter_path}")
        if args.use_ipadapter_via_pipeline:
            # Use the pipeline loader which attaches the IP-Adapter and provides helper to compute embeds
            try:
                from load_ipadapter_pipeline import load_pipeline_and_ipadapter, compute_ipadapter_embeds

                pipe = load_pipeline_and_ipadapter(args.sd_model, args.ipadapter_path, device=device, torch_dtype=(torch.float16 if device.type == 'cuda' else torch.float32), subfolder="models")
                image_embeds = compute_ipadapter_embeds(pipe, args.input_image, device=device)
                # image_embeds is a list of tensors
                if len(image_embeds) == 0:
                    results.update({"ipadapter_loaded": False, "ipadapter_error": "no_embeds_returned"})
                else:
                    ip_emb = image_embeds[0]
                    if not isinstance(ip_emb, torch.Tensor):
                        ip_emb = torch.tensor(ip_emb).to(device)
                    if ip_emb.ndim == 3:
                        ip_mean = ip_emb.mean(dim=1)
                    else:
                        ip_mean = ip_emb
                    if ip_mean.shape[-1] == mapped_mean.shape[-1]:
                        cos_m = (F.normalize(ip_mean, p=2, dim=-1) @ F.normalize(mapped_mean, p=2, dim=-1).T).cpu().numpy().tolist()
                        l2_m = (ip_mean - mapped_mean).norm(p=2, dim=-1).cpu().numpy().tolist()
                        results.update({"ipadapter_loaded": True, "ipadapter_desc": f"pipeline({args.ipadapter_path})", "cosine_ipadapter_mapped": cos_m, "l2_ipadapter_mapped": l2_m})
                    else:
                        results.update({"ipadapter_loaded": True, "ipadapter_desc": f"pipeline({args.ipadapter_path})", "note_ipadapter_mapped": "dim_mismatch"})
            except Exception as e:
                results.update({"ipadapter_loaded": False, "ipadapter_error": str(e)})
        else:
            adapter, desc = try_load_ipadapter(args.ipadapter_path, device)
            if adapter is None:
                print("Could not load IPAdapter automatically; skipping IPAdapter comparison.")
                results["ipadapter_loaded"] = False
            else:
                print(f"Loaded IPAdapter via {desc}; computing embeddings...")
                try:
                    ip_emb = embed_with_ipadapter(adapter, desc, pil_img, device)
                    # ensure ip_emb is tensor on device
                    if not isinstance(ip_emb, torch.Tensor):
                        ip_emb = torch.tensor(ip_emb).to(device)
                    # pool if sequence
                    if ip_emb.ndim == 3:
                        ip_mean = ip_emb.mean(dim=1)
                    else:
                        ip_mean = ip_emb
                    # align dims and compute metrics where possible
                    if ip_mean.shape[-1] == mapped_mean.shape[-1]:
                        cos_m = (F.normalize(ip_mean, p=2, dim=-1) @ F.normalize(mapped_mean, p=2, dim=-1).T).cpu().numpy().tolist()
                        l2_m = (ip_mean - mapped_mean).norm(p=2, dim=-1).cpu().numpy().tolist()
                        results.update({"ipadapter_loaded": True, "ipadapter_desc": desc, "cosine_ipadapter_mapped": cos_m, "l2_ipadapter_mapped": l2_m})
                    else:
                        results.update({"ipadapter_loaded": True, "ipadapter_desc": desc, "note_ipadapter_mapped": "dim_mismatch"})
                except Exception as e:
                    results.update({"ipadapter_loaded": False, "ipadapter_error": str(e)})

    # Save results
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved comparison results to {out_json}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
