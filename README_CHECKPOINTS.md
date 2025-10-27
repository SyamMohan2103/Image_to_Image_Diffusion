Checkpoint format and how to reload CLIP/Text weights

This project stores combined checkpoints when training may have fine-tuned both the mapper and parts of CLIP.

Files produced
- mapper_epoch{N}.pth, mapper_best.pth, mapper_final.pth

Checkpoint dict keys
- "mapper_state_dict": state_dict for the mapper network (required)
- "config": dict with {"in_dim", "out_seq_len", "out_dim"}
- "clip_state_dict": (optional) state_dict for the CLIP vision encoder if it was unfrozen during training
- "text_state_dict": (optional) state_dict for the CLIP text encoder if it was unfrozen during training

Notes
- Keys "clip_state_dict" and "text_state_dict" are present ONLY when those modules had trainable parameters during training.
- Saved state dicts may contain a "module." prefix if models were wrapped in DDP; the loader strips that prefix automatically.

How to generate using a checkpoint's CLIP/Text weights
- Use the CLI flag `--use_checkpoint_clip` when running in `--mode gen`. Example:

```bash
python run_mapper.py --mode gen --mapper ./mapper_ckpt/mapper_best.pth --input_image input.png \
    --out_dir ./output --use_checkpoint_clip
```

This will attempt to load `clip_state_dict` and/or `text_state_dict` from the provided mapper checkpoint and apply them to the CLIP models used for generation. If those keys are not present, the generation will fall back to the HF pretrained CLIP weights.

Programmatic usage
- The helper `_restore_clip_text_from_ckpt(mapper_path, clip_model, text_model, device)` is provided and will return True if any state was loaded.

Best practices
- If you fine-tune CLIP, prefer staged unfreeze with `--unfreeze_after_epoch` to avoid instability.
- Keep backups of the HF pretrained model names used for generation. If you restore fine-tuned CLIP weights, ensure they were trained from the same HF model version for compatibility.

If you'd like, I can add a small utility that extracts the checkpoint into a folder with separate files (`mapper.pt`, `clip.pt`, `text.pt`) for easier inspection or re-use.
