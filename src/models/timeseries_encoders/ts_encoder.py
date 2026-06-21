from argparse import Namespace
from pdb import set_trace
import torch
from torch import nn

from src.common import TASKS
from src.data.base import TimeseriesOutputs
from src.utils.masking import Masking
from src.utils.tools import NamespaceWithDefaults, MultiHeadWrapper

from src.models.layers.embed import TimeEmbedding
from src.models.layers.revin import RevIN
from src.models.layers.prediction_head import ForecastingHead, ReconstructionHead, EmbeddingHead, RetrievalAugmentedHead
from src.models.layers.get_encoder import get_transformer_backbone
from src.models.layers.time_prior import TemporalPriorEncoder
from src.models.layers.text_forecast_fusion import DirectTextForecastFusion
from src.data.time_prior_features import TIME_FEATURE_NAMES

class TS_Encoder(nn.Module):
    def __init__(self, configs: Namespace | dict, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        self.configs = configs
        self.task_name = configs.task_name
        self.n_channels = configs.n_channels  # number of channels
        self.output_attention = configs.output_attention

        ## Patching parameters
        self.seq_len_channel = configs.seq_len_channel  # length of per channel time-series
        self.patch_len = configs.patch_len  # length of each patch
        self.patch_stride_len = configs.patch_stride_len  # stride length of each patch
        self.num_patches = (max(self.seq_len_channel, self.patch_len) - self.patch_len) // self.patch_stride_len + 1
        # self.total_len = self.seq_len_channel * self.n_channels + self.n_channels + 1

        self.channel_special_tokens = configs.model_name == "TraceEncoder"
        self.dec_shape = "BTD" if configs.model_name == "TraceEncoder" else "else"
        # Normalization, patching and embedding
        self.normalizer = RevIN(
            num_features=1, affine=configs.getattr("revin_affine", False)
        )
        self.patch_embedding = TimeEmbedding(
            d_model=configs.d_model,
            num_channels=configs.n_channels,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            pos_embed_type=configs.getattr("pos_embed_type", "rel_pos"),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
            channel_special_tokens=self.channel_special_tokens
        )
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0),
                                      patch_len=configs.patch_len,
                                      stride=configs.patch_stride_len)

        # Transformer backbone
        self.d_model = configs.d_model
        self.encoder = get_transformer_backbone(configs)
        self.use_temporal_prior = configs.getattr("use_temporal_prior", False)
        self.prior_alpha = configs.getattr("prior_alpha", 0.1)
        self._logged_temporal_prior = False
        self._logged_forecast_temporal_prior = False
        if self.use_temporal_prior:
            prior_input_dim = configs.getattr("prior_input_dim", None)
            if prior_input_dim is None:
                prior_input_dim = len(TIME_FEATURE_NAMES)
                print(
                    "[TS_Encoder] prior_input_dim was null; "
                    f"inferred {prior_input_dim} from TIME_FEATURE_NAMES."
                )
            self.temporal_prior_encoder = TemporalPriorEncoder(
                prior_input_dim=prior_input_dim,
                num_domains=configs.getattr("num_domains", 10),
                domain_emb_dim=configs.getattr("domain_emb_dim", 32),
                d_model=configs.d_model,
                dropout=configs.getattr("prior_dropout", configs.getattr("dropout", 0.1)),
                hidden_dim=configs.getattr("prior_hidden_dim", None),
            )
        else:
            self.temporal_prior_encoder = None

        self.use_direct_text_forecast = (
            configs.getattr("use_direct_text_forecast", False)
            and not configs.getattr("ts_only", False)
            and self.task_name == TASKS.FORECASTING
        )
        if self.use_direct_text_forecast:
            fusion_type = configs.getattr("text_fusion_type", "gated_residual")
            if fusion_type != "gated_residual":
                raise ValueError(
                    f"Unsupported text_fusion_type={fusion_type}; "
                    "expected gated_residual"
                )
            self.direct_text_fusion = DirectTextForecastFusion(
                d_model=configs.d_model,
                text_emb_dim=configs.getattr("text_emb_dim", 768),
                hidden_dim=configs.getattr("prior_hidden_dim", None),
                dropout=configs.getattr("prior_dropout", configs.getattr("dropout", 0.1)),
                text_residual_alpha=configs.getattr("text_residual_alpha", 0.05),
            )
        else:
            self.direct_text_fusion = None

        # Prediction Head
        self.head = self._get_head(self.task_name)
        self.embedding_head = EmbeddingHead(self.n_channels)


    def set_retriever(self, device):
        from src.models.trace_retriever import RetrievalAugmentedWrapper
        self.retriever = RetrievalAugmentedWrapper(device)
        for param in self.retriever.parameters():
            param.requires_grad = False
        self.top_k = self.configs.top_k

    def _update_inputs(
        self, configs: Namespace | dict, **kwargs
    ) -> NamespaceWithDefaults:
        if isinstance(configs, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**configs, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(configs)


    def _get_head(self, task_name: str) -> nn.Module:
        if hasattr(self.configs, "data_name") and self.configs.data_name in ["health", "env", "energy"]:
            return MultiHeadWrapper({
                "reconstruct_head": ReconstructionHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.configs.patch_len,
                    self.configs.getattr("dropout", 0.1),
                    self.configs.getattr("orth_gain", 1.41),
                ),
                "forecasting_head": ForecastingHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.num_patches,
                    self.configs.forecast_horizon,
                    self.configs.getattr("head_dropout", 0.1),
                )
            })
        else:
            if task_name == TASKS.PRETRAINING:
                return MultiHeadWrapper({
                    "reconstruct_head": ReconstructionHead(
                        self.configs.n_channels,
                        self.configs.d_model,
                        self.configs.patch_len,
                        self.configs.getattr("dropout", 0.1),
                        self.configs.getattr("orth_gain", 1.41),
                    )
                })
            elif task_name == TASKS.RECONSTRUCTION:
                return ReconstructionHead(
                    self.configs.d_model,
                    self.configs.patch_len,
                    self.configs.getattr("dropout", 0.1),
                    self.configs.getattr("orth_gain", 1.41),
                )
            elif task_name == TASKS.FORECASTING:
                return ForecastingHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.num_patches,
                    self.configs.forecast_horizon,
                    self.configs.getattr("head_dropout", 0.1),
                )
            elif task_name == TASKS.EMBEDDING:
                return EmbeddingHead(
                    self.configs.n_channels
                )
            elif task_name == TASKS.RAG:
                return RetrievalAugmentedHead(
                    self.configs.n_channels,
                    self.configs.d_model,
                    self.num_patches,
                    self.configs.forecast_horizon,
                    self.configs.getattr("head_dropout", 0.1),
                    self.configs.top_k,
                    self.configs.ts_only
                )
            else:
                raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_encoding_out(self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [B, C, L] Time-series data
        pretrain_mask  : [B, C, L] Data that is masked but still attended to via mask-tokens
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        output:
            [B, total_len, d_model] for TraceEncoder, [B, C, N, d_model] for other encoders
        """
        B, C, L = x_enc.shape
        # Normalization
        x_enc = self.normalizer(x=x_enc, mask=pretrain_mask * input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        # Some time-series are too short, so masking them out results in NaNs.


        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=pretrain_mask)
        # [B, total_len, d_model] or [B, C, N, d_model]

        # Encoder
        # In forecasting, if input_mask is all ones, there is no padding/missing value to mask.
        # Skip passing it into attention to avoid special-token length mismatch.
        if input_mask is not None and torch.all(input_mask == 1):
            attention_mask = None
        else:
            attention_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)  #[B, C, N]

        enc_out, attns = self.encoder(
            x=enc_in,
            attn_mask=attention_mask,
            **{
                "n_vars": self.n_channels,
                "n_tokens": self.num_patches,
            }
        )
        return enc_out, attns

    def _apply_temporal_prior_to_cls(
        self,
        h_cls,
        time_feat=None,
        time_feat_weight=None,
        domain_id=None,
    ):
        """
        Add domain-aware temporal prior to the CLS representation only.

        This keeps patch tokens, channel tokens, attention masks, and positional
        encodings unchanged. If the prior is disabled or any input is missing,
        TRACE falls back to the original CLS behavior.
        """
        if (
            not self.use_temporal_prior
            or self.temporal_prior_encoder is None
            or h_cls is None
            or time_feat is None
            or time_feat_weight is None
            or domain_id is None
        ):
            return h_cls, None

        prior_emb = self.temporal_prior_encoder(
            time_feat=time_feat.to(h_cls.device),
            time_feat_weight=time_feat_weight.to(h_cls.device),
            domain_id=domain_id.to(h_cls.device),
        )
        h_cls = h_cls + self.prior_alpha * prior_emb
        if not self._logged_temporal_prior:
            print(
                "[TS_Encoder] temporal prior applied: "
                f"h_cls.shape={h_cls.shape}, "
                f"prior_emb.shape={prior_emb.shape}, "
                f"prior_alpha={self.prior_alpha}, "
                f"use_temporal_prior={self.use_temporal_prior}"
            )
            self._logged_temporal_prior = True
        return h_cls, prior_emb


    def embed(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        """
        x_enc : [B, C, L] Time-series data
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        if input_mask is None:
            pretrain_mask = torch.ones_like(x_enc)
        else:
            if input_mask is None:
                pretrain_mask = torch.ones_like(x_enc)
            else:
                pretrain_mask = torch.ones_like(input_mask)
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)

        # Decoder
        input_mask_patch_view = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        emb_dict= self.head(enc_out, input_mask_patch_view, shape=self.dec_shape)
        h_cls, prior_emb = self._apply_temporal_prior_to_cls(
            emb_dict["cls"],
            time_feat=kwargs.get("time_feat", None),
            time_feat_weight=kwargs.get("time_feat_weight", None),
            domain_id=kwargs.get("domain_id", None),
        )
        if prior_emb is not None:
            emb_dict["cls"] = h_cls
            emb_dict["global"] = h_cls


        return TimeseriesOutputs(
            input_mask=input_mask,
            embeddings=emb_dict["global"], # [B, d_model]
            channel_embeddings=emb_dict["channels"], # [B, C, d_model]
            cls_embedding=emb_dict["cls"], # [B, d_model]
            prior_emb=prior_emb,
        )

    def pretraining(
        self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [B, C, L] Time-series data
        pretrain_mask  : [B, C, L] Data that is masked but still attended to via mask-tokens
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        if pretrain_mask is None:
            pretrain_mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            pretrain_mask = pretrain_mask.to(x_enc.device)  # mask: [B, C, L]

        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)

        # Decoder
        dec_out = self.head["reconstruct_head"](enc_out, shape=self.dec_shape)  # [B, C, L]
        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")
        illegal_output = (
            self._check_model_weights_for_illegal_values()
            if self.configs.debug
            else None
        )
        if self.output_attention:
            return TimeseriesOutputs(
                input_mask=input_mask,  # [B, C, L]
                reconstruction=dec_out,  # [B, C, L]
                pretrain_mask=pretrain_mask,  # [B, C, L]
                illegal_output=illegal_output  # None or True
            ), attns
        else:
            return TimeseriesOutputs(
                input_mask=input_mask,  # [B, C, L]
                reconstruction=dec_out,  # [B, C, L]
                pretrain_mask=pretrain_mask,  # [B, C, L]
                illegal_output=illegal_output  # None or True
            )

    def timemmd_pretraining(
        self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [B, C, L] Time-series data
        pretrain_mask  : [B, C, L] Data that is masked but still attended to via mask-tokens
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        if pretrain_mask is None:
            pretrain_mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            pretrain_mask = pretrain_mask.to(x_enc.device)  # mask: [B, C, L]

        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)

        # Decoder
        reconstruction = self.head["reconstruct_head"](enc_out, shape=self.dec_shape)  # [B, C, L]
        prior_emb = None
        use_cls_context = False
        forecast_enc_out = enc_out
        if self.dec_shape == "BTD":
            h_cls, prior_emb = self._apply_temporal_prior_to_cls(
                enc_out[:, 0, :],
                time_feat=kwargs.get("time_feat", None),
                time_feat_weight=kwargs.get("time_feat_weight", None),
                domain_id=kwargs.get("domain_id", None),
            )
            if prior_emb is not None:
                forecast_enc_out = enc_out.clone()
                forecast_enc_out[:, 0, :] = h_cls
                use_cls_context = True
        forecasting = self.head["forecasting_head"](
            forecast_enc_out,
            shape=self.dec_shape,
            use_cls_context=use_cls_context,
        )  # z: [B, C, H]

        # De-Normalization
        reconstruction = self.normalizer(x=reconstruction, mode="denorm")  #[B, C, L]
        forecasting = self.normalizer(x=forecasting, mode="denorm")  #[B, C, H]
        if prior_emb is not None and not self._logged_forecast_temporal_prior:
            print(
                "[TS_Encoder] forecasting temporal prior: "
                f"hidden.shape={forecast_enc_out.shape}, "
                f"prior_emb.shape={prior_emb.shape}, "
                f"pred.shape={forecasting.shape}, "
                f"prior_alpha={self.prior_alpha}, "
                f"use_temporal_prior={self.use_temporal_prior}"
            )
            self._logged_forecast_temporal_prior = True

        return TimeseriesOutputs(
            input_mask=input_mask,  # [B, C, L]
            reconstruction=reconstruction,  # [B, C, L]
            pretrain_mask=pretrain_mask,  # [B, C, L]
            forecast=forecasting,  # [B, C, H]
            prior_emb=prior_emb,
        )



    def forecast(
        self, x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        **kwargs
    ):
        """
        x_enc : [B, C, L] Time-series data
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        if input_mask is None:
            input_mask = torch.ones_like(x_enc)
        pretrain_mask = torch.ones_like(input_mask)
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)

        # Decoder
        prior_emb = None
        time_emb = None
        ts_emb = None
        fused_emb = None
        direct_text_gate = None
        use_cls_context = False
        if self.dec_shape == "BTD":
            ts_emb = enc_out[:, 0, :]
            h_cls, prior_emb = self._apply_temporal_prior_to_cls(
                ts_emb,
                time_feat=kwargs.get("time_feat", None),
                time_feat_weight=kwargs.get("time_feat_weight", None),
                domain_id=kwargs.get("domain_id", None),
            )
            time_emb = (
                prior_emb
                if prior_emb is not None
                else torch.zeros_like(h_cls)
            )
            fused_emb = h_cls
            text_residual = None
            if (
                self.direct_text_fusion is not None
                and kwargs.get("text_emb", None) is not None
                and kwargs.get("text_mask", None) is not None
            ):
                fused_emb, direct_text_gate = self.direct_text_fusion(
                    ts_emb=h_cls,
                    text_emb=kwargs["text_emb"].to(h_cls.device),
                    time_emb=time_emb,
                    text_mask=kwargs["text_mask"].to(h_cls.device),
                )
                text_residual = fused_emb - h_cls

            if prior_emb is not None:
                enc_out = enc_out.clone()
                enc_out[:, 0, :] = (
                    fused_emb if text_residual is not None else h_cls
                )
                use_cls_context = True
            elif text_residual is not None:
                # Inject only the bounded residual. For text_mask=0 this is
                # exactly zero, preserving the original TS-only forecast.
                enc_out = enc_out.clone()
                enc_out[:, 0, :] = text_residual
                use_cls_context = True

        dec_out = self.head(
            enc_out,
            shape=self.dec_shape,
            use_cls_context=use_cls_context,
        )  # z: [B, C, H]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")  #[B, C, H]
        if prior_emb is not None and not self._logged_forecast_temporal_prior:
            print(
                "[TS_Encoder] forecasting temporal prior: "
                f"hidden.shape={enc_out.shape}, "
                f"prior_emb.shape={prior_emb.shape}, "
                f"pred.shape={dec_out.shape}, "
                f"prior_alpha={self.prior_alpha}, "
                f"use_temporal_prior={self.use_temporal_prior}"
            )
            self._logged_forecast_temporal_prior = True

        return TimeseriesOutputs(
            input_mask=input_mask,
            forecast=dec_out,
            prior_emb=prior_emb,
            cls_embedding=ts_emb,
            fused_emb=fused_emb,
            direct_text_gate=direct_text_gate,
            time_emb=time_emb,
        )

    def rag_forecasting(
        self, x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        **kwargs
    ):
        """
        x_enc : [B, C, L] Time-series data
        input_mask : [B, C, L]
            Input mask for the time-series data that is unobserved.
            This is typically padded data, that is not attended to.
        """
        if input_mask is None:
            input_mask = torch.ones_like(x_enc)
        pretrain_mask = torch.ones_like(input_mask)
        enc_out, attns = self._get_encoding_out(x_enc, pretrain_mask, input_mask)
        soft_prompt = self.retriever(x_enc, input_mask, top_k=self.top_k)
        dec_out = self.head(enc_out,soft_prompt, shape=self.dec_shape)  # z: [B, C, H]
        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")  #[B, C, H]

        return TimeseriesOutputs(
            input_mask=input_mask,
            forecast=dec_out)


    def forward(
        self,
        x_enc: torch.Tensor,
        pretrain_mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        '''
        Input: (L is the length of per-channel time series)
            x_enc: [B, C, L]
            pretrain_mask: [B, C, L]
            input_mask: [B, C, L]
        '''
        if hasattr(self.configs, "data_name") and self.configs.data_name in ["health", "env", "energy"]:
            return self.timemmd_pretraining(x_enc=x_enc, pretrain_mask=pretrain_mask, input_mask=input_mask, **kwargs)
        else:
            if self.task_name == TASKS.PRETRAINING:
                return self.pretraining(x_enc=x_enc, pretrain_mask=pretrain_mask, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.FORECASTING:
                return self.forecast(x_enc=x_enc, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.EMBEDDING:
                return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
            elif self.task_name == TASKS.RAG:
                return self.rag_forecasting(x_enc=x_enc, input_mask=input_mask, **kwargs)
            else:
                raise NotImplementedError(f"Task {self.task_name} not implemented.")

    def _check_model_weights_for_illegal_values(self):
        illegal_encoder_weights = (
            torch.stack([torch.isnan(p).any() for p in self.encoder.parameters()])
            .any()
            .item()
        )
        illegal_head_weights = (
            torch.stack([torch.isnan(p).any() for p in self.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_weights = (
            torch.stack(
                [torch.isnan(p).any() for p in self.patch_embedding.parameters()]
            )
            .any()
            .item()
        )

        return (
            illegal_encoder_weights
            or illegal_head_weights
            or illegal_patch_embedding_weights
        )
