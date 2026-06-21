import torch
import torch.nn as nn


class DirectTextForecastFusion(nn.Module):
    """Inject a bounded text residual into a forecasting CLS embedding."""

    def __init__(
        self,
        d_model: int,
        text_emb_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
        text_residual_alpha: float = 0.05,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.text_residual_alpha = text_residual_alpha
        self.text_projection = nn.Linear(text_emb_dim, d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ts_emb, text_emb, time_emb, text_mask):
        if ts_emb.ndim != 2 or time_emb.shape != ts_emb.shape:
            raise ValueError(
                "ts_emb and time_emb must both be [B, d_model], "
                f"got {ts_emb.shape} and {time_emb.shape}"
            )
        if text_emb.ndim != 2 or text_emb.shape[0] != ts_emb.shape[0]:
            raise ValueError(
                f"text_emb must be [B, text_emb_dim], got {text_emb.shape}"
            )

        text_proj = self.text_projection(text_emb.to(ts_emb.dtype))
        gate_input = torch.cat([ts_emb, text_proj, time_emb], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        text_mask = text_mask.to(device=ts_emb.device, dtype=ts_emb.dtype).view(-1, 1)
        gate = gate * text_mask
        fused_emb = ts_emb + self.text_residual_alpha * gate * text_proj
        return fused_emb, gate
