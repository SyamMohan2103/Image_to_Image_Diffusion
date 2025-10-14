
# Image → Image Diffusion (Mapper)

This repository implements a small "mapper" model that converts an input image into a latent vector compatible with the text-conditioning input of a frozen Stable Diffusion UNet. The mapper is trained to produce embeddings with the same dimension as the text encoder output so that images can be passed as conditioning signals into the diffusion model instead of (or in addition to) text.

## Motivation

- Many image-to-image tasks benefit from supplying a learned conditioning vector rather than raw text.
- Freezing the diffusion model and training a small mapper drastically reduces training cost and lets you leverage a pretrained Stable Diffusion model.

## Contents

- `data_download.py` — streaming LAION downloader (saves images + captions into Excel batches).
- `run_mapper.py` — training and generation entrypoint (train/generate modes).
- `laion_images/`, `laion_images_filtered/`, `laion_subset_batch*.xlsx` — sample data and saved batches.
- `mapper_ckpt/` — where mapper checkpoints are stored.
- `output/` — generation outputs.

## High-level design

1. Dataset: pairs of (image, caption). Images are preprocessed and converted to feature representations.
2. Mapper: a lightweight model (e.g. MLP / CNN → projection) trained to output a vector matching the text encoder dimension used by Stable Diffusion's conditioning.
3. Training: Stable Diffusion (UNet + text encoder) is frozen. The mapper outputs a conditioning vector which replaces or augments the text embeddings fed into the UNet. Loss can be L2 against text embeddings and/or diffusion-guided objectives.
4. Generation: Given an input image, the mapper produces a conditioning vector which is fed to the frozen UNet to produce an output image.

## Preparing data

1. Run `data_download.py` to stream LAION and save image batches. The script writes Excel files named `laion_subset_batch{n}.xlsx` and saves images to `laion_images/`.
2. You can optionally filter or inspect batches manually. The training script expects a dataset directory or a list of batch files.

## Training

Example command:

```bash
python run_mapper.py --mode train --dataset /path/to/batches --prefix laion_subset --out_dir ./mapper_ckpt --epochs 10 --batch_size 64
```

Key flags explained
- `--mode train|gen` — training or generation mode.
- `--dataset` — path to the folder containing batch Excel files (or dataset manifest).
- `--prefix` — file prefix used by `data_download.py` (default `laion_subset`).
- `--out_dir` — directory to save mapper checkpoints.
- `--mapper` — path to a saved mapper checkpoint (used in `gen` mode).
- `--input_image` — path to input image for generation in `gen` mode.
- `--epochs`, `--batch_size` — training hyperparameters.

## Generation

Example command:

```bash
python run_mapper.py --mode gen --mapper ./mapper_ckpt/mapper.pth --input_image input.png --out_dir ./output
```
```bash
python run_mapper.py --mode train --dataset /home1/koustav/Image_to_Image_Diffusion/filtered_metadata_parallel.csv --prefix laion_subset --image_path_prefix /home1/koustav/Image_to_Image_Diffusion/laion_images_kp/ --out_dir /home1/koustav/Image_to_Image_Diffusion/mapper_ckpt --epochs 3 --batch_size 8 --lr 1e-4 --hidden_dim 4096 --num_layers 3 --clip_model openai/clip-vit-large-patch14 --gpus "" --device_ids 2,3,4 --device cuda:2
```

This will run the mapper on `input.png`, produce the conditioning vector, and run the frozen Stable Diffusion model to produce an output image in `./output`.

## Implementation notes
- The diffusion model and its tokenizer/text-encoder are kept frozen. Only the mapper parameters are updated.
- The mapper is trained to match the dimension and distribution of text embeddings produced by the diffusion model's text encoder.
- You can design the mapper architecture to suit your needs: small MLP, CNN+proj, or ViT-based projector.

## Next Steps

Extending the mapper
- Try multi-scale conditioning: map different image scales to different segments of the text embedding and concatenate.
- Train a conditional mapper that takes both image and a short prompt to guide style.
- Fine-tune the mapper with additional perceptual or adversarial losses for higher visual fidelity.


## Monitoring

1. Train with TensorBoard (compound):
In VS Code Run view, choose "Train + TensorBoard" and start. It will activate the env, run TensorBoard, and launch training.
2. Or start TensorBoard manually:
python run_tensorboard.py --logdir ./mapper_ckpt/tb_logs --port 6006
3. Or simply: tensorboard --logdir ./mapper_ckpt/tb_logs
Open http://localhost:6006 to view:
Graph (model)
Scalars (loss)
Histograms (input features, predictions)
Images (generated variation)