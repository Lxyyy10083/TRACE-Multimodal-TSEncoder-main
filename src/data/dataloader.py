import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from .base import TimeseriesData
from .dataset import (
    PretrainingDataset,
    ForecastingDataset,
    ClassificationDataset,
    RetrievalDataset,
    MMDataset,
    TIMEMMD_DATASETS,
)
from torch.utils.data.distributed import DistributedSampler

MMD_DATA_NAMES = {"health", "env", "energy", *TIMEMMD_DATASETS.keys()}


def _collate_fn_basic(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples] # [C, L]
    input_masks = [torch.from_numpy(example.input_mask) for example in examples] # [C, L]
    timeseries = torch.stack(timeseries)  # [B, C, L]
    input_masks = torch.stack(input_masks)  # [B, C, L]
    labels = [example.labels for example in examples]
    labels = np.asarray(labels)  # [B]
    return TimeseriesData(timeseries=timeseries, input_mask=input_masks, labels=labels)

def _collate_fn_forecasting(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples] # [C, L]
    input_masks = [torch.from_numpy(example.input_mask) for example in examples] # [C, L]
    timeseries = torch.stack(timeseries)  # [B, C, L]
    input_masks = torch.stack(input_masks)  # [B, C, L]
    forecast = [torch.from_numpy(example.forecast) for example in examples] # [C, H]
    forecast = torch.stack(forecast)  # [B, C, H]
    batch_kwargs = {
        "timeseries": timeseries,
        "input_mask": input_masks,
        "forecast": forecast,
    }

    if examples[0].time_feat is not None:
        time_feat = [torch.from_numpy(example.time_feat) for example in examples]
        batch_kwargs["time_feat"] = torch.stack(time_feat)  # [B, T, D]
        assert batch_kwargs["time_feat"].ndim == 3, (
            f"batched time_feat must be [B, T, D], got {batch_kwargs['time_feat'].shape}"
        )

    if examples[0].time_feat_weight is not None:
        time_feat_weight = [
            torch.from_numpy(example.time_feat_weight) for example in examples
        ]
        batch_kwargs["time_feat_weight"] = torch.stack(time_feat_weight)  # [B, T, D]
        assert batch_kwargs["time_feat_weight"].shape == batch_kwargs["time_feat"].shape, (
            "batched time_feat_weight must match time_feat shape, "
            f"got {batch_kwargs['time_feat_weight'].shape} vs {batch_kwargs['time_feat'].shape}"
        )

    if examples[0].domain_id is not None:
        domain_id = [int(np.asarray(example.domain_id).item()) for example in examples]
        batch_kwargs["domain_id"] = torch.tensor(domain_id, dtype=torch.long)  # [B]
        assert batch_kwargs["domain_id"].dtype == torch.long, "domain_id must be int64"

    if examples[0].text_emb is not None:
        batch_kwargs["text_emb"] = torch.stack(
            [torch.from_numpy(example.text_emb) for example in examples]
        )
        batch_kwargs["text_mask"] = torch.tensor(
            [float(example.text_mask) for example in examples],
            dtype=torch.float32,
        )
        batch_kwargs["text_time"] = [example.text_time for example in examples]
        batch_kwargs["forecast_origin_time"] = [
            example.forecast_origin_time for example in examples
        ]
    
    if examples[0].prior_y is not None:
        prior_y = [torch.from_numpy(example.prior_y) for example in examples] # [C, H]
        prior_y = torch.stack(prior_y)  # [B, C, H]
        batch_kwargs["prior_y"] = prior_y
        return TimeseriesData(**batch_kwargs)
    else:
        return TimeseriesData(**batch_kwargs)

def _collate_fn_classification(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples] # [C, L]
    timeseries = torch.stack(timeseries)  # [B, C, L]
    labels = [example.labels for example in examples]
    labels = np.asarray(labels)  # [B]
    return TimeseriesData(timeseries=timeseries, labels=labels)


def _collate_fn_retrieval(examples):
    examples = list(filter(lambda x: x is not None, examples))
    timeseries = [torch.from_numpy(example.timeseries) for example in examples] # [C, L]
    timeseries = torch.stack(timeseries)  # [B, C, L]
    
    input_masks = [torch.from_numpy(example.input_mask) for example in examples] # [C, L]
    input_masks = torch.stack(input_masks)  # [B, C, L]
    
    labels = [example.labels for example in examples]
    labels = np.asarray(labels)  # [B]
    
    channel_description_emb = [example.channel_description_emb for example in examples]
    channel_description_emb = torch.stack(channel_description_emb)  # [B, C, d]
    
    description_emb = [example.description_emb for example in examples]
    description_emb = torch.stack(description_emb)  # [B, d]
    
    event_emb = [example.event_emb for example in examples]
    event_emb = torch.stack(event_emb)  # [B, d]
    
    prior_kwargs = {}
    if examples[0].time_feat is not None:
        prior_kwargs["time_feat"] = torch.stack([
            torch.from_numpy(example.time_feat) for example in examples
        ])
        prior_kwargs["time_feat_weight"] = torch.stack([
            torch.from_numpy(example.time_feat_weight) for example in examples
        ])
        prior_kwargs["domain_id"] = torch.tensor([
            int(np.asarray(example.domain_id).item()) for example in examples
        ], dtype=torch.long)

    if examples[0].descriptions is not None: #in test set
        descriptions = [example.descriptions for example in examples]
        events = [example.events for example in examples]
        return TimeseriesData(timeseries=timeseries,
                              input_mask=input_masks, 
                              labels=labels, 
                              channel_description_emb=channel_description_emb, 
                              description_emb=description_emb, 
                              event_emb=event_emb, 
                              descriptions=descriptions, 
                              events=events,
                              **prior_kwargs)
    else:
        return TimeseriesData(timeseries=timeseries,
                              input_mask=input_masks, 
                              labels=labels, 
                              channel_description_emb=channel_description_emb, 
                              description_emb=description_emb, 
                              event_emb=event_emb,
                              **prior_kwargs)

def get_dataloader(args):
    if hasattr(args, "data_name") and str(args.data_name).lower() in MMD_DATA_NAMES:
        return get_mmd_dataloader(args)
    else:
        if args.task_name == "pretraining":
            dataset = PretrainingDataset(
                seq_len_channel=args.seq_len_channel,
                data_split=args.data_split,
                scale=args.scale,
                upsampling_pad_direction=args.upsampling_pad_direction,
                upsampling_type=args.upsampling_type,
                downsampling_type=args.downsampling_type,
                pad_mode=args.pad_mode,
            )
        elif args.task_name == "forecasting":
            dataset = ForecastingDataset(
                seq_len_channel=args.seq_len_channel,
                forecast_horizon=args.forecast_horizon,
                data_split=args.data_split,
                scale=args.scale,
                upsampling_pad_direction=args.upsampling_pad_direction,
                upsampling_type=args.upsampling_type,
                downsampling_type=args.downsampling_type,
                pad_mode=args.pad_mode,
            )
        elif args.task_name == "classification":
            dataset = ClassificationDataset(
                seq_len_channel=args.seq_len_channel,
                data_split=args.data_split,
                scale=args.scale,
            )
        elif args.task_name == "retrieval":
            dataset = RetrievalDataset(
                seq_len_channel=args.seq_len_channel,
                data_split=args.data_split,
                scale=args.scale,
                text_encoder_name=args.text_encoder_name,
                domain_name=getattr(args, "domain_name", None),
                n_channels=args.n_channels,
            )
        else:
            raise ValueError(f"Invalid task name: {args.task_name}")
        
        collate_fn_map = {
            "pretraining": _collate_fn_basic,
            "forecasting": _collate_fn_forecasting,
            "classification": _collate_fn_classification,
            "retrieval": _collate_fn_retrieval,
        }
        if getattr(args, "distributed", False):
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                collate_fn=collate_fn_map[args.task_name],
                sampler=DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.shuffle),
            )
        else:
            shuffle = True if args.data_split == "train" else False
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                collate_fn=collate_fn_map[args.task_name],
            )
        return dataloader



def get_mmd_dataloader(args):
    dataset = MMDataset(
        seq_len_channel=args.seq_len_channel,
        forecast_len=args.forecast_horizon,
        data_name=args.data_name,
        data_split=args.data_split,
        scale=args.scale,
        task_name=args.task_name,
        use_direct_text_forecast=getattr(args, "use_direct_text_forecast", False),
        text_data_path=getattr(args, "text_data_path", None),
        text_date_col=getattr(args, "text_date_col", "date"),
        text_col=getattr(args, "text_col", "text"),
        text_encoder_type=getattr(args, "text_encoder_type", "offline_embedding"),
        text_embedding_path=getattr(args, "text_embedding_path", None),
        text_emb_dim=getattr(args, "text_emb_dim", 768),
        lookback_text_window=getattr(args, "lookback_text_window", None),
        use_text_leakage_check=getattr(args, "use_text_leakage_check", True),
    )
    
    if args.world_size > 1:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=_collate_fn_forecasting,
            sampler=DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=args.shuffle),
            )
    else:
        shuffle = True if args.data_split == "train" else False
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=_collate_fn_forecasting
        )
    return dataloader
