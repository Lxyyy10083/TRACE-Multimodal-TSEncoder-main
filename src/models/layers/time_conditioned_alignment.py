import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers.time_prior import TemporalPriorEncoder


class TimeConditionedCompatibilityScorer(nn.Module):
    """Lightweight time-conditioned text/time-series compatibility scorer."""

    def __init__(
        self,
        d_model: int,
        prior_input_dim: int,
        num_domains: int = 10,
        domain_emb_dim: int = 32,
        hidden_dim: int = None,
        dropout: float = 0.1,
        temperature: float = 0.07,
        prior_alpha: float = 0.1,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        hidden_dim = hidden_dim or d_model
        self.temperature = temperature
        self.prior_alpha = prior_alpha
        self.temporal_prior_encoder = TemporalPriorEncoder(
            prior_input_dim=prior_input_dim,
            num_domains=num_domains,
            domain_emb_dim=domain_emb_dim,
            d_model=d_model,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
        self.ts_projection = nn.Linear(d_model, d_model, bias=False)
        self.text_projection = nn.Linear(d_model, d_model, bias=False)
        self.prior_mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        ts_emb,
        text_emb,
        time_feat,
        time_feat_weight,
        domain_id,
    ):
        if ts_emb.ndim != 2 or text_emb.ndim != 2:
            raise ValueError(
                "ts_emb and text_emb must be [B, D], "
                f"got {ts_emb.shape} and {text_emb.shape}"
            )
        if ts_emb.shape != text_emb.shape:
            raise ValueError(
                "Paired ts/text embeddings must have identical shapes, "
                f"got {ts_emb.shape} and {text_emb.shape}"
            )

        z_time = self.temporal_prior_encoder(
            time_feat=time_feat,
            time_feat_weight=time_feat_weight,
            domain_id=domain_id,
        )
        ts_projected = F.normalize(self.ts_projection(ts_emb), dim=-1)
        text_projected = F.normalize(self.text_projection(text_emb), dim=-1)
        base_logits = torch.matmul(ts_projected, text_projected.T) / self.temperature

        batch_size, hidden_size = z_time.shape
        z_time_i = z_time[:, None, :].expand(batch_size, batch_size, hidden_size)
        text_j = text_projected[None, :, :].expand(
            batch_size,
            batch_size,
            hidden_size,
        )
        prior_features = torch.cat(
            [z_time_i, text_j, z_time_i * text_j],
            dim=-1,
        )
        prior_logits = self.prior_mlp(prior_features).squeeze(-1)
        compatibility_logits = base_logits + self.prior_alpha * prior_logits

        paired_gate_features = torch.cat(
            [z_time, text_projected, z_time * text_projected],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_mlp(paired_gate_features))
        return compatibility_logits, prior_logits, gate
