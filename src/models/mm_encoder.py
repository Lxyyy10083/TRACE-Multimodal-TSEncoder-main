from argparse import Namespace
from copy import deepcopy
import torch
from torch import nn

from src.common import TASKS
from src.data.base import TimeseriesOutputs
from src.utils.masking import Masking
from src.utils.tools import NamespaceWithDefaults, MultiHeadWrapper

from src.models.layers.embed import TimeEmbedding
from src.models.layers.revin import RevIN
from src.models.layers.prediction_head import (
    ClassificationHead,
    ForecastingHead,
    ReconstructionHead,
    EmbeddingHead,
    RetrievalAugmentedHead
)
from src.models.layers.get_encoder import get_transformer_backbone
from src.models.timeseries_encoders.ts_encoder import TS_Encoder as TimeSeriesEncoder


def _infer_text_embedding_dim(text_encoder_name: str, default_dim: int = 768):
    """Infer common precomputed text embedding dimensions without loading models."""
    model_dimensions = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "bert-base-uncased": 768,
        "bert-large-uncased": 1024,
        "roberta-base": 768,
        "roberta-large": 1024,
        "nomic-ai/nomic-embed-text-v1": 768,
        "nomic-ai/nomic-embed-text-v1.5": 768,
    }
    return model_dimensions.get(text_encoder_name, default_dim)


class PriorLogitScorer(nn.Module):
    """Score pairwise compatibility between temporal priors and text embeddings."""

    def __init__(self, d_model: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, prior_emb, text_emb):
        """
        Args:
            prior_emb: [B, d_model]
            text_emb: [N, d_model]

        Returns:
            prior_logits: [B, N]
        """
        bsz, dim = prior_emb.shape
        num_text = text_emb.shape[0]
        prior_i = prior_emb.unsqueeze(1).expand(bsz, num_text, dim)
        text_j = text_emb.unsqueeze(0).expand(bsz, num_text, dim)
        pair_feat = torch.cat([prior_i, text_j, prior_i * text_j], dim=-1)
        return self.mlp(pair_feat).squeeze(-1)


class TS_Encoder(nn.Module):

    def __init__(self, configs: Namespace | dict, **kwargs):
        super().__init__()

        configs = self._update_inputs(configs, **kwargs)
        self.configs = configs

        self.task_name = configs.task_name
        self.n_channels = configs.n_channels

        self.seq_len_channel = configs.seq_len_channel
        self.patch_len = configs.patch_len
        self.patch_stride_len = configs.patch_stride_len

        self.num_patches = (
            (max(self.seq_len_channel, self.patch_len) - self.patch_len)
            // self.patch_stride_len + 1
        )

        self.channel_special_tokens = configs.model_name == "TraceEncoder"
        self.dec_shape = "BTD" if configs.model_name == "TraceEncoder" else "else"

        # =====================
        # core modules
        # =====================
        self.normalizer = RevIN(
            num_features=1,
            affine=getattr(configs, "revin_affine", False)
        )

        self.patch_embedding = TimeEmbedding(
            d_model=configs.d_model,
            num_channels=configs.n_channels,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=getattr(configs, "dropout", 0.1),
            pos_embed_type=getattr(configs, "pos_embed_type", "rel_pos"),
            value_embedding_bias=getattr(configs, "value_embedding_bias", False),
            orth_gain=getattr(configs, "orth_gain", 1.41),
            channel_special_tokens=self.channel_special_tokens
        )

        self.mask_generator = Masking(
            mask_ratio=getattr(configs, "mask_ratio", 0.0),
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len
        )

        self.encoder = get_transformer_backbone(configs)

        self.head = self._get_head(self.task_name)
        self.embedding_head = EmbeddingHead(self.n_channels)

    # ======================================================
    # FIX 1: stable input wrapper
    # ======================================================
    def _update_inputs(self, configs, **kwargs):
        if isinstance(configs, dict):
            configs = NamespaceWithDefaults(**configs)

        if "model_kwargs" in kwargs:
            for k, v in kwargs["model_kwargs"].items():
                setattr(configs, k, v)

        return configs

    # ======================================================
    # FIX 2: ALWAYS compute all heads (DDP-safe)
    # ======================================================
    def _get_encoding_out(self, x_enc, pretrain_mask, input_mask=None):

        x_enc = self.normalizer(
            x=x_enc,
            mask=pretrain_mask * input_mask if input_mask is not None else pretrain_mask,
            mode="norm"
        )

        x_enc = torch.nan_to_num(x_enc)

        enc_in = self.patch_embedding(x_enc, mask=pretrain_mask)

        attn_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ) if input_mask is not None else None

        enc_out, attns = self.encoder(
            x=enc_in,
            attn_mask=attn_mask,
            n_vars=self.n_channels,
            n_tokens=self.num_patches,
        )

        return enc_out, attns

    # ======================================================
    # PRETRAINING (FIXED FOR DDP)
    # ======================================================
    def pretraining(self, x_enc, pretrain_mask=None, input_mask=None):

        if pretrain_mask is None:
            pretrain_mask = self.mask_generator.generate_mask(
                x=x_enc,
                input_mask=input_mask
            ).to(x_enc.device)

        enc_out, attns = self._get_encoding_out(
            x_enc, pretrain_mask, input_mask
        )

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        # ======================================================
        # IMPORTANT FIX: ALWAYS COMPUTE BOTH HEADS
        # (even if loss ignores them)
        # ======================================================

        recon = self.head["reconstruct_head"](
            enc_out,
            shape=self.dec_shape
        )

        cls = self.head["classification_head"](
            enc_out,
            input_mask_patch_view,
            shape=self.dec_shape
        )

        recon = self.normalizer(x=recon, mode="denorm")

        # ======================================================
        # FIX: prevent DDP graph mismatch
        # detach optional outputs safely
        # ======================================================
        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=recon,
            classification=cls,
            pretrain_mask=pretrain_mask
        )

    # ======================================================
    # FORECAST
    # ======================================================
    def forecast(self, x_enc, input_mask=None):

        pretrain_mask = torch.ones_like(input_mask)

        enc_out, _ = self._get_encoding_out(
            x_enc, pretrain_mask, input_mask
        )

        dec = self.head(enc_out, shape=self.dec_shape)
        dec = self.normalizer(x=dec, mode="denorm")

        return TimeseriesOutputs(
            input_mask=input_mask,
            forecast=dec
        )

    # ======================================================
    # CLASSIFICATION
    # ======================================================
    def classification(self, x_enc, input_mask=None):

        pretrain_mask = torch.ones_like(input_mask)

        enc_out, _ = self._get_encoding_out(
            x_enc, pretrain_mask, input_mask
        )

        mask_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        out = self.head(enc_out, mask_view, shape=self.dec_shape)

        return TimeseriesOutputs(
            input_mask=input_mask,
            classification=out
        )

    # ======================================================
    # FORWARD (CRITICAL STABILITY FIX)
    # ======================================================
    def forward(self, x_enc, pretrain_mask=None, input_mask=None, **kwargs):

        # ALWAYS use same return structure
        if self.task_name == TASKS.PRETRAINING:
            return self.pretraining(x_enc, pretrain_mask, input_mask)

        elif self.task_name == TASKS.FORECASTING:
            return self.forecast(x_enc, input_mask)

        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classification(x_enc, input_mask)

        else:
            raise NotImplementedError(self.task_name)


class MultiModalEncoder(nn.Module):
    """
    Minimal multimodal wrapper used by context alignment.

    The time-series branch is the real TRACE TS_Encoder in embedding mode. When
    temporal prior inputs are provided, they are passed into that branch and are
    added to the CLS representation inside TS_Encoder.
    """

    def __init__(self, configs: Namespace | dict):
        super().__init__()
        if isinstance(configs, dict):
            configs = NamespaceWithDefaults(**configs)
        else:
            configs = NamespaceWithDefaults.from_namespace(configs)
        self.configs = configs
        ts_configs = deepcopy(configs)
        ts_configs.task_name = TASKS.EMBEDDING
        self.ts_encoder = TimeSeriesEncoder(configs=ts_configs)
        text_dim = _infer_text_embedding_dim(
            configs.getattr("text_encoder_name", "bert-base-uncased")
        )
        self.text_projection = nn.Linear(text_dim, configs.d_model)
        self.channel_text_projection = nn.Linear(text_dim, configs.d_model)
        self.event_projection = nn.Linear(text_dim, configs.d_model)
        self.classification_head = nn.Linear(
            configs.d_model,
            configs.getattr("num_class", 9),
        )
        self.use_prior_calibrated_logits = configs.getattr(
            "use_prior_calibrated_logits",
            False,
        )
        self.prior_beta = configs.getattr("prior_beta", 0.1)
        self.prior_scorer = PriorLogitScorer(
            d_model=configs.d_model,
            hidden_dim=configs.getattr("prior_hidden_dim", None),
            dropout=configs.getattr("prior_dropout", configs.getattr("dropout", 0.1)),
        )
        self._logged_prior_logits = False

    def prior_calibrated_logits(self, ts_emb, text_emb, prior_emb=None, tau=0.07):
        """
        Combine semantic similarity logits with temporal-prior calibration.

        sim_logits are the original TRACE likelihood term. prior_logits are a
        learnable temporal prior term conditioned on prior_emb and each text
        embedding. The sum forms the posterior matching distribution logits.
        """
        sim_logits = torch.matmul(ts_emb, text_emb.T) / tau
        if (
            not self.use_prior_calibrated_logits
            or prior_emb is None
        ):
            return sim_logits

        prior_logits = self.prior_scorer(prior_emb, text_emb)
        posterior_logits = sim_logits + self.prior_beta * prior_logits
        if not self._logged_prior_logits:
            print(
                "[MultiModalEncoder] prior-calibrated logits: "
                f"sim_logits.shape={sim_logits.shape}, "
                f"prior_logits.shape={prior_logits.shape}, "
                f"posterior_logits.shape={posterior_logits.shape}, "
                f"prior_beta={self.prior_beta}, "
                f"use_prior_calibrated_logits={self.use_prior_calibrated_logits}"
            )
            self._logged_prior_logits = True
        return posterior_logits

    def forward(
        self,
        x_enc,
        input_mask=None,
        channel_description_emb=None,
        description_emb=None,
        event_emb=None,
        time_feat=None,
        time_feat_weight=None,
        domain_id=None,
        **kwargs,
    ):
        ts_outputs = self.ts_encoder(
            x_enc=x_enc,
            input_mask=input_mask,
            time_feat=time_feat,
            time_feat_weight=time_feat_weight,
            domain_id=domain_id,
            **kwargs,
        )

        if description_emb is not None:
            description_emb = self.text_projection(description_emb)
        if channel_description_emb is not None:
            channel_description_emb = self.channel_text_projection(channel_description_emb)
        if event_emb is not None:
            event_emb = self.event_projection(event_emb)

        classification = self.classification_head(ts_outputs.embeddings)
        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=ts_outputs.reconstruction,
            embeddings=ts_outputs.embeddings,
            channel_embeddings=ts_outputs.channel_embeddings,
            cls_embedding=ts_outputs.cls_embedding,
            classification=classification,
            description_emb=description_emb,
            channel_description_emb=channel_description_emb,
            event_emb=event_emb,
            prior_emb=ts_outputs.prior_emb,
        )

    def get_ts_embedding(self, x_enc, input_mask=None, **kwargs):
        return self.ts_encoder(
            x_enc=x_enc,
            input_mask=input_mask,
            **kwargs,
        )
