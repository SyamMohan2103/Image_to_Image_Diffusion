GitHub Copilot

Short list of useful losses (with pros/cons) and small code snippets you can drop into run_mapper.py.

Masked MSE (you already use)

Pros: stable, simple regression of vectors.
Cons: sensitive to scale, ignores angular alignment.
Masked L1

More robust to outliers than MSE.
Cosine (or 1 - cosine_similarity)

Pros: aligns direction (useful when magnitude is unimportant).
Use per-token or pooled tokens.
Combined MSE + Cosine

Best of both worlds: preserve magnitude and direction.
Contrastive / InfoNCE

Pros: pushes correct image-text pair together while pushing other pairs apart (improves retrieval/discrimination).
Requires batch negatives (works well with larger batch sizes).
Triplet / Margin loss

Use explicit positive/negative triplets; needs mining negatives or semi-hard negative selection.
KL divergence (if targets are distributions)

Use if your target is a probability distribution (e.g., soft targets).
Distillation-style losses (attention/pairwise)

E.g., FitNet: match intermediate activations or pairwise similarities.
Which to try first

Pooled cosine loss (simple, often effective).
Combined (cosine + MSE) if you want magnitude too.
Contrastive InfoNCE if you can increase batch size and want better discriminative alignment.
Example code — add near top of run_mapper.py (loss helpers):


```
# ...existing code...
import torch.nn.functional as F
# ...existing code...

def pooled_from_token_embeddings(token_embeds, attention_mask):
    # token_embeds: [B, L, H], attention_mask: [B, L]
    att = attention_mask.unsqueeze(-1).to(token_embeds.dtype)  # [B, L, 1]
    summed = (token_embeds * att).sum(dim=1)  # [B, H]
    lengths = att.sum(dim=1).clamp(min=1.0)  # [B, 1]
    return summed / lengths

def masked_mse_loss(pred, target, attention_mask):
    # token-wise masked MSE (your current loss)
    diff2 = (pred - target).pow(2).mean(dim=-1)  # [B, L]
    masked = diff2 * attention_mask
    return masked.sum() / (attention_mask.sum() + 1e-8)

def masked_cosine_loss(pred, target, attention_mask, eps=1e-8):
    # token-wise cosine (1 - cos_sim), averaged over valid tokens
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1, eps=eps)  # [B, L]
    masked = cos * attention_mask
    return masked.sum() / (attention_mask.sum() + 1e-8)

def pooled_cosine_loss(pred_tokens, target_tokens, attention_mask):
    # pool tokens then compute cosine loss on pooled vectors
    p = pooled_from_token_embeddings(pred_tokens, attention_mask)  # [B, H]
    t = pooled_from_token_embeddings(target_tokens, attention_mask)  # [B, H]
    return 1.0 - F.cosine_similarity(p, t, dim=-1).mean()

def info_nce_loss_from_pooled(pred_tokens, target_tokens, attention_mask, temperature=0.07):
    # Simple InfoNCE using pooled representations; assumes batch negatives
    p = pooled_from_token_embeddings(pred_tokens, attention_mask)  # [B, H]
    t = pooled_from_token_embeddings(target_tokens, attention_mask)  # [B, H]
    p = F.normalize(p, dim=-1)
    t = F.normalize(t, dim=-1)
    logits = (p @ t.t()) / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
# ...existing code...
```

Practical tips

If you unfreeze CLIP layers, use a much smaller LR for CLIP params (e.g., 1e-6) and larger LR for mapper. Use optimizer param groups.
Monitor both loss and retrieval metrics: average cosine similarity on a validation set, top-1/top-5 retrieval (image->text) to see real impact.
Normalize embeddings (L2) before computing cosine/contrastive losses.
Start with pooled cosine (cheap) before trying contrastive (needs more batch size/negatives).
If training becomes unstable, add gradient clipping and a small weight decay.
If you want, I can:

Add the helper loss functions into run_mapper.py and wire a CLI flag to pick loss type.
Provide optimizer param-group snippet to train mapper + small-unfrozen CLIP blocks.
