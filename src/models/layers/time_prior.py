import torch
import torch.nn as nn


class TemporalPriorEncoder(nn.Module):
    """
    Encode domain-aware temporal priors into the TRACE hidden space.

    This module is not a plain concatenation of raw calendar features. It first
    calibrates time features with domain-specific feature weights, then fuses
    the calibrated temporal signal with a learnable domain embedding. The output
    prior_emb can later be used to calibrate time-series/text matching scores or
    other multimodal alignment modules without changing the raw data pipeline.
    """

    def __init__(
        self,
        prior_input_dim: int,
        num_domains: int,
        domain_emb_dim: int,
        d_model: int,
        dropout: float = 0.1,
        hidden_dim: int = None,
        activation: str = "gelu",
    ):
        super().__init__()
        self.prior_input_dim = prior_input_dim
        self.num_domains = num_domains
        self.domain_emb_dim = domain_emb_dim
        self.d_model = d_model

        hidden_dim = hidden_dim or max(d_model, prior_input_dim + domain_emb_dim)
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.domain_embedding = nn.Embedding(num_domains, domain_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(prior_input_dim + domain_emb_dim, hidden_dim),
            activation_layer,
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        time_feat: torch.Tensor,
        time_feat_weight: torch.Tensor,
        domain_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            time_feat: [B, prior_input_dim] temporal feature vectors. If a
                temporal window [B, T, prior_input_dim] is provided, it is
                averaged over T before encoding.
            time_feat_weight: same shape as time_feat, with domain-specific
                feature weights.
            domain_id: [B] integer domain ids.

        Returns:
            prior_emb: [B, d_model] domain-aware temporal prior embedding.
        """
        if time_feat.shape != time_feat_weight.shape:
            raise ValueError(
                "time_feat and time_feat_weight must have identical shapes, "
                f"got {time_feat.shape} and {time_feat_weight.shape}"
            )

        if time_feat.dim() == 3:
            # Current data loaders may provide one feature vector per target
            # timestamp. Aggregate the window into a sample-level prior vector.
            time_feat = time_feat.mean(dim=1)
            time_feat_weight = time_feat_weight.mean(dim=1)
        elif time_feat.dim() != 2:
            raise ValueError(
                "time_feat must be [B, prior_input_dim] or "
                f"[B, T, prior_input_dim], got {time_feat.shape}"
            )

        if time_feat.shape[-1] != self.prior_input_dim:
            raise ValueError(
                f"Expected prior_input_dim={self.prior_input_dim}, "
                f"got {time_feat.shape[-1]}"
            )

        domain_id = domain_id.long().view(-1)
        if domain_id.shape[0] != time_feat.shape[0]:
            raise ValueError(
                f"domain_id batch size {domain_id.shape[0]} does not match "
                f"time_feat batch size {time_feat.shape[0]}"
            )

        # Domain weights emphasize different calendar granularities per domain:
        # Agriculture can focus on season/month, Traffic on weekday/week/hour,
        # Energy on month/season/hour, etc.
        weighted_time_feat = time_feat * time_feat_weight
        domain_emb = self.domain_embedding(domain_id)
        prior_input = torch.cat([weighted_time_feat, domain_emb], dim=-1)
        prior_emb = self.output_norm(self.mlp(prior_input))
        return prior_emb


def _shape_test():
    """Minimal shape check: prior_emb should be [B, d_model]."""
    batch_size = 4
    prior_input_dim = 12
    num_domains = 10
    domain_emb_dim = 16
    d_model = 384

    encoder = TemporalPriorEncoder(
        prior_input_dim=prior_input_dim,
        num_domains=num_domains,
        domain_emb_dim=domain_emb_dim,
        d_model=d_model,
        dropout=0.1,
    )
    time_feat = torch.randn(batch_size, prior_input_dim)
    time_feat_weight = torch.ones(batch_size, prior_input_dim)
    domain_id = torch.tensor([0, 3, 8, 9])
    prior_emb = encoder(time_feat, time_feat_weight, domain_id)
    assert prior_emb.shape == (batch_size, d_model)


if __name__ == "__main__":
    _shape_test()
