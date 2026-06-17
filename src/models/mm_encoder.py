from argparse import Namespace
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