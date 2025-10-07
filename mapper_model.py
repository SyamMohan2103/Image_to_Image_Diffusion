from typing import Optional
import torch
import torch.nn as nn


class ImageToTextMapper(nn.Module):
    """MLP that maps CLIP pooled image features (B, I) to a target
    token embedding sequence (B, L, H) where L is text seq length (e.g. 77)
    and H is the text hidden dim (e.g. 768).

    The final linear layer outputs L*H values which are reshaped.
    """

    def __init__(self, in_dim: int, out_seq_len: int, out_dim: int, hidden_dim: int = 4096, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        cur = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        # final projection to sequence*dim
        layers.append(nn.Linear(cur, out_seq_len * out_dim))
        self.net = nn.Sequential(*layers)
        self.out_seq_len = out_seq_len
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim]
        b = x.shape[0]
        out = self.net(x)
        out = out.view(b, self.out_seq_len, self.out_dim)
        return out