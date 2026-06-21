import logging
import os
import warnings
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.common import PATHS, TASKS
from src.data.load_data import (
    load_timeseries_from_json,
    load_npy_timeseries,
    load_forecasting_from_json,
    load_retrieval_from_parquet,
    load_retrieval_from_timemmd_csv,
    normalize_csv_date_column,
    build_time_prior_features,
)
from src.utils.data import (
    interpolate_timeseries,
    upsample_timeseries,
)
from .base import TaskDataset, TimeseriesData
import torch
warnings.filterwarnings("ignore")

TIMEMMD_DATASETS = {
    "agriculture": "Agriculture",
    "climate": "Climate",
    "economy": "Economy",
    "energy": "Energy",
    "environment": "Environment",
    "health": "Health",
    "security": "Security",
    "socialgood": "SocialGood",
    "traffic": "Traffic",
    "weather": "Weather",
}


class PretrainingDataset(TaskDataset):
    def __init__(
        self,
        seq_len_channel: int = 180,
        root_path: str = PATHS.DATA_DIR + "pretrain/",
        data_split: str = "train",
        scale: bool = True,
        task_name: str = TASKS.PRETRAINING,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        downsampling_type="interpolate",
        pad_mode="constant",
        pad_constant_values=0,
        return_meta_data=False,
        **kwargs,
    ):
        super(PretrainingDataset, self).__init__()
        self.seq_len_channel = seq_len_channel
        self.root_path = root_path

        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values
        self.return_meta_data = return_meta_data
        # Input checking
        self._check_inputs()

        # Read data
        self._read_data()

    def _check_inputs(self):
        # Input checking
        assert self.data_split in [
            "train",
            "test",
            "val",
        ], "data_split must be one of 'train', 'test' or 'val'"


    def _transform_labels(self, train_labels: np.ndarray, test_labels: np.ndarray):
        # Move the labels to {0, ..., L-1}
        labels = np.unique(train_labels)
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        train_labels = np.vectorize(transform.get)(train_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, test_labels

    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()

        # output = load_timeseries_from_json(self.data_split, self.root_path, return_meta_data=self.return_meta_data)
        output = load_npy_timeseries(self.data_split, self.root_path, return_meta_data=self.return_meta_data)
        
        if self.return_meta_data:
            self.data, self.labels, self.meta_data = output
        else:
            self.data, self.labels = output 
        
        # meta_data: list of dicts
        # Check if time-series have equal lengths. If not, left pad with zeros
        # self._check_if_equal_length()

        # Check and remove NaNs
        self._check_and_remove_nans()
        self.n_timeseries = len(self.data)
        if self.scale:
            for i, ts in enumerate(self.data):
                ts = ts.T  # Now shape is [L, C]
                ts_scaled = self.scaler.fit_transform(ts)  
                self.data[i] = ts_scaled.T
        # self.data: list of [C, L], L varies across time
        # self.data = self.data.T
        # self.input_mask = self.input_mask.T

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[index] # [C, L]
        timeseries_len = timeseries.shape[1]
        labels = self.labels[index,].astype(int)
        
        ## padding to the same length
        if timeseries_len <= self.seq_len_channel:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len_channel,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )

        meta_data = self.meta_data[index] if self.return_meta_data else None
        
        return TimeseriesData(
            timeseries=timeseries,  # [C, L]
            labels=labels,
            input_mask=input_mask,  # [C,L]
            metadata=meta_data
        )

    def __len__(self):
        return self.n_timeseries

    def _check_and_remove_nans(self):
        for i, ts in enumerate(self.data):
            if np.isnan(ts).any():
                ts = interpolate_timeseries(
                    timeseries=ts, interp_length=ts.shape[-1]
                )
                ts = np.nan_to_num(ts)
                self.data[i] = ts

    def _check_if_equal_length(self):
        if isinstance(self.data, list):
            n_timeseries = len(self.data)
            self.n_channels = self.data[0].shape[0]
            # Assume all time-series have the same number of channels
            # Then we have time-series of unequal lengths
            max_len = max([ts.shape[-1] for ts in self.data])
            for i, ts in enumerate(self.data):
                self.data[i] = interpolate_timeseries(
                    timeseries=ts, interp_length=max_len
                )
            self.data = np.asarray(self.data)
            logging.info(
                f"Time-series have unequal lengths. Reshaping to {self.data.shape}"
            )

    def plot(self, idx, channel=0):
        timeseries_data = self.__getitem__(idx)
        label = timeseries_data.labels
        timeseries = timeseries_data.timeseries[0, channel, :]

        plt.title(f"idx={idx}, label={label}", fontsize=18)
        plt.plot(
            np.arange(self.seq_len_channel),
            timeseries,
            label="Time-series",
            c="darkblue",
        )
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=18)
        plt.show()


class ForecastingDataset(TaskDataset):
    def __init__(
        self,
        seq_len_channel: int = 180,
        forecast_horizon: int = 7,
        root_path: str = PATHS.DATA_DIR + "forecasting/",
        data_split: str = "train",
        scale: bool = True,
        task_name: str = TASKS.FORECASTING,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        downsampling_type="interpolate",
        pad_mode="constant",
        pad_constant_values=0,
        return_meta_data=False,
        **kwargs,
    ):
        super(ForecastingDataset, self).__init__()
        self.seq_len_channel = seq_len_channel
        self.forecast_len = forecast_horizon
        self.root_path = root_path
        self.data_split = data_split
        self.scale = scale
        self.task_name = task_name
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.downsampling_type = downsampling_type
        self.pad_mode = pad_mode
        self.pad_constant_values = pad_constant_values
        self.return_meta_data = return_meta_data
        # Input checking
        self._check_inputs()

        # Read data
        self._read_data()

    def _check_inputs(self):
        # Input checking
        assert self.data_split in [
            "train",
            "test",
            "val",
        ], "data_split must be one of 'train', 'test' or 'val'"



    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()

        raw_data= load_forecasting_from_json(self.data_split, self.root_path)
        data = []
        forecast_data = []
        # Check and remove NaNs
        raw_data = self._check_and_remove_nans(raw_data)
        self.n_timeseries = len(raw_data)
        if self.scale:
            for i, ts in enumerate(raw_data):
                ts_scale = self.scaler.fit_transform(ts.T)
                ts_scale = ts_scale.T  #[C, L]
                data.append(ts_scale[:, :-self.forecast_len])
                forecast_data.append(ts_scale[:, -self.forecast_len:])
        self.data = data
        self.forecast_data = forecast_data

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[index] # [C, L]
        forecast = self.forecast_data[index] # [C, H]
        assert forecast.shape[-1] == self.forecast_len  
        timeseries_len = timeseries.shape[1]
        
        ## padding to the same length
        if timeseries_len <= self.seq_len_channel:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len_channel,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )

        return TimeseriesData(
            timeseries=timeseries,  # [C, L]
            forecast=forecast,  # [C, H]
            input_mask=input_mask,  # [C,L]
        )

    def __len__(self):
        return self.n_timeseries

    def _check_and_remove_nans(self, data):
        for i, ts in enumerate(data):
            if np.isnan(ts).any():
                ts = interpolate_timeseries(
                    timeseries=ts, interp_length=ts.shape[-1]
                )
                ts = np.nan_to_num(ts)
                data[i] = ts
        return data



class ClassificationDataset(TaskDataset):
    def __init__(self, 
        seq_len_channel: int = 180,
        root_path: str = PATHS.DATA_DIR + "classification/",
        data_split: str = "train",
        scale: bool = True,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        pad_mode="constant"):
        super(ClassificationDataset, self).__init__()
        self.seq_len_channel = seq_len_channel
        self.root_path = root_path
        self.data_split = data_split
        self.scale = scale
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.pad_mode = pad_mode
        self._read_data()
    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        self.data, self.labels = load_npy_timeseries(self.data_split, self.root_path)
        self._check_and_remove_nans()
        self.n_timeseries = len(self.data)
        if self.scale:
            for i, ts in enumerate(self.data):
                ts = ts.T  # Now shape is [L, C]
                ts_scaled = self.scaler.fit_transform(ts)  
                self.data[i] = ts_scaled.T
        
        
    def __getitem__(self, index):
        timeseries = self.data[index] # [C, L]
        timeseries_len = timeseries.shape[1]
        labels = self.labels[index,].astype(int)
        
        ## padding to the same length
        if timeseries_len <= self.seq_len_channel:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len_channel,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )
        
        return TimeseriesData(
            timeseries=timeseries,  # [C, L]
            labels=labels,
            # input_mask=input_mask,  # [C,L]
        )

    def _check_and_remove_nans(self):
        for i, ts in enumerate(self.data):
            if np.isnan(ts).any():
                ts = interpolate_timeseries(
                    timeseries=ts, interp_length=ts.shape[-1]
                )
                ts = np.nan_to_num(ts)
                self.data[i] = ts
                
    def __len__(self):
        return self.n_timeseries
    
    
class RetrievalDataset(TaskDataset):
    def __init__(self, 
        seq_len_channel: int = 180,
        root_path: str = PATHS.DATA_DIR + "retrieval/",
        data_split: str = "train",
        scale: bool = True,
        text_encoder_name: str = "bert-base-uncased",
        domain_name: str = None,
        n_channels: int = 7,
        upsampling_pad_direction="backward",
        upsampling_type="pad",
        pad_mode="constant"):
        super(RetrievalDataset, self).__init__()
        self.seq_len_channel = seq_len_channel
        self.root_path = root_path
        self.data_split = data_split
        self.scale = scale
        self.upsampling_pad_direction = upsampling_pad_direction
        self.upsampling_type = upsampling_type
        self.pad_mode = pad_mode
        self.text_encoder_name = text_encoder_name
        self.domain_name = domain_name
        self.n_channels = n_channels
        self._read_data()

    def _read_data(self) -> TimeseriesData:
        self.scaler = StandardScaler()
        parquet_path = os.path.join(
            self.root_path,
            self.data_split,
            f"{self.data_split}.parquet",
        )
        if os.path.exists(parquet_path):
            payload, self.retrieval_metadata = load_retrieval_from_parquet(
                self.data_split,
                self.root_path,
                self.text_encoder_name,
                return_metadata=True,
            )
        else:
            if not self.domain_name:
                raise ValueError(
                    "domain_name is required for TimeMMD CSV retrieval fallback"
                )
            payload, self.retrieval_metadata = load_retrieval_from_timemmd_csv(
                self.data_split,
                self.root_path,
                self.domain_name,
                self.text_encoder_name,
                seq_len_channel=self.seq_len_channel,
                n_channels=self.n_channels,
            )
        if self.data_split == "train":
            self.data, self.descriptions_emb, self.channel_descriptions_emb, self.events_emb, self.labels = payload
        else:
            self.data, self.descriptions_emb, self.channel_descriptions_emb, self.events_emb, self.labels, self.descriptions, self.channel_descriptions, self.events = payload
        self._check_and_remove_nans()
        self.n_timeseries = len(self.data)
        if self.scale:
            for i, ts in enumerate(self.data):
                ts = ts.T  # Now shape is [L, C]
                ts_scaled = self.scaler.fit_transform(ts)  
                self.data[i] = ts_scaled.T
        
        
    def __getitem__(self, index):
        timeseries = self.data[index] # [C, L]
        timeseries_len = timeseries.shape[1]
        labels = self.labels[index,].astype(int)
        ## padding to the same length
        if timeseries_len <= self.seq_len_channel:
            timeseries, input_mask = upsample_timeseries(
                timeseries,
                self.seq_len_channel,
                direction=self.upsampling_pad_direction,
                sampling_type=self.upsampling_type,
                mode=self.pad_mode,
            )
        else:
            timeseries = timeseries[:, -self.seq_len_channel:]
            input_mask = np.ones_like(timeseries)

        prior_kwargs = {}
        if self.retrieval_metadata is not None:
            prior_kwargs = {
                "time_feat": self.retrieval_metadata["time_feat"][index],
                "time_feat_weight": self.retrieval_metadata["time_feat_weight"][index],
                "domain_id": np.array(self.retrieval_metadata["domain_id"], dtype=np.int64),
                "metadata": {
                    "date": self.retrieval_metadata["dates"][index],
                    "source_path": self.retrieval_metadata["source_path"],
                },
            }
        
        if self.data_split == "train":
            return TimeseriesData(
                timeseries=timeseries,  # [C, L]
                labels=labels,
                description_emb=self.descriptions_emb[index],
                channel_description_emb=self.channel_descriptions_emb[index], #[C, d]
                event_emb=self.events_emb[index],
                input_mask=input_mask,  # [C,L]
                **prior_kwargs,
            )
        else:
            return TimeseriesData(
                timeseries=timeseries,  # [C, L]
                labels=labels,
                description_emb=self.descriptions_emb[index],
                channel_description_emb=self.channel_descriptions_emb[index], #[C, d]
                event_emb=self.events_emb[index],
                descriptions=self.descriptions[index],
                # channel_descriptions=self.channel_descriptions[index],
                events=self.events[index],
                input_mask=input_mask,  # [C,L]
                **prior_kwargs,
            )

    def _check_and_remove_nans(self):
        for i, ts in enumerate(self.data):
            if np.isnan(ts).any():
                ts = interpolate_timeseries(
                    timeseries=ts, interp_length=ts.shape[-1]
                )
                ts = np.nan_to_num(ts)
                self.data[i] = ts
                
    def __len__(self):
        return self.n_timeseries





class MMDataset(TaskDataset):
    def __init__(
        self,
        seq_len_channel: int = 32,
        forecast_len: int = 12,
        root_path: str = PATHS.DATA_DIR,
        data_name: str = "env",
        data_split: str = "train",
        scale: bool = True,
        task_name: str = TASKS.PRETRAINING,
        use_direct_text_forecast: bool = False,
        text_data_path: str = None,
        text_date_col: str = "date",
        text_col: str = "text",
        text_encoder_type: str = "offline_embedding",
        text_embedding_path: str = None,
        text_emb_dim: int = 768,
        lookback_text_window=None,
        use_text_leakage_check: bool = True,
        **kwargs,
    ):
        super(MMDataset, self).__init__()
        self.seq_len = seq_len_channel
        self.label_len = forecast_len
        self.pred_len = forecast_len
        self.task_name = task_name
        self.use_direct_text_forecast = (
            bool(use_direct_text_forecast) and task_name == "forecasting"
        )
        self.text_data_path = text_data_path
        self.text_date_col = text_date_col
        self.text_col = text_col
        self.text_encoder_type = text_encoder_type
        self.text_embedding_path = text_embedding_path
        self.text_emb_dim = int(text_emb_dim)
        self.lookback_text_window = lookback_text_window
        self.use_text_leakage_check = bool(use_text_leakage_check)
        self._text_dates = None
        self._text_embeddings = None
        assert data_split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[data_split]
        self.text_name='Final_Search_6'
        self.features = "S"
        self.target = "OT"
        self.scale = scale
        self.data_name = data_name
        self.root_path, self.data_path = self._resolve_csv_path(root_path, data_name)
        self._logged_time_prior_sample = False
        self.__read_data__()
        self._load_direct_text_data()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def _load_embedding_array(self, embedding_path):
        extension = os.path.splitext(embedding_path)[1].lower()
        if extension == ".npy":
            return np.load(embedding_path)
        if extension == ".npz":
            archive = np.load(embedding_path)
            key = "embeddings" if "embeddings" in archive.files else archive.files[0]
            return archive[key]
        if extension in {".pt", ".pth"}:
            value = torch.load(embedding_path, map_location="cpu")
            if isinstance(value, dict):
                for key in ("embeddings", "text_emb", "text_embeddings"):
                    if key in value:
                        value = value[key]
                        break
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            return np.asarray(value)
        if extension == ".csv":
            embedding_df = pd.read_csv(embedding_path)
            numeric_df = embedding_df.select_dtypes(include=[np.number])
            return numeric_df.to_numpy()
        raise ValueError(f"Unsupported text embedding format: {embedding_path}")

    def _load_direct_text_data(self):
        if not self.use_direct_text_forecast:
            return
        if self.text_encoder_type != "offline_embedding":
            print(
                "[Direct text][warning] only offline_embedding is supported; "
                "falling back to time-series-only samples."
            )
            return
        if not self.text_data_path:
            print(
                "[Direct text][warning] text_data_path is null; "
                "all samples will use text_mask=0."
            )
            return

        try:
            text_df = pd.read_csv(self.text_data_path)
            if self.text_date_col not in text_df.columns:
                raise ValueError(
                    f"Missing text date column '{self.text_date_col}'"
                )
            resolved_text_col = self.text_col
            if resolved_text_col not in text_df.columns and "fact" in text_df.columns:
                resolved_text_col = "fact"
            if resolved_text_col not in text_df.columns:
                raise ValueError(
                    f"Missing text column '{self.text_col}' and fallback 'fact'"
                )
            text_dates = pd.to_datetime(
                text_df[self.text_date_col],
                errors="coerce",
            )
            text_available = (
                text_df[resolved_text_col].notna()
                & text_df[resolved_text_col].astype(str).str.strip().ne("")
            )
            valid = (text_dates.notna() & text_available).to_numpy()
            text_df = text_df.loc[valid].reset_index(drop=True)
            text_dates = text_dates.loc[valid].reset_index(drop=True)

            if self.text_embedding_path:
                embeddings = self._load_embedding_array(self.text_embedding_path)
                if len(embeddings) != len(valid):
                    raise ValueError(
                        "text embedding rows must match the original text CSV rows: "
                        f"{len(embeddings)} != {len(valid)}"
                    )
                embeddings = np.asarray(embeddings)[valid]
            elif "text_emb" in text_df.columns:
                embeddings = np.stack(
                    [
                        np.asarray(ast.literal_eval(str(value)), dtype=np.float32)
                        for value in text_df["text_emb"]
                    ]
                )
            else:
                embedding_cols = [
                    column
                    for column in text_df.columns
                    if column.startswith("emb_") or column.startswith("embedding_")
                ]
                if not embedding_cols:
                    print(
                        "[Direct text][warning] no offline embeddings found; "
                        "raw text is not encoded during training and text_mask will be 0."
                    )
                    return
                embeddings = text_df[embedding_cols].to_numpy(dtype=np.float32)

            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[1] != self.text_emb_dim:
                raise ValueError(
                    f"Expected text embeddings [N, {self.text_emb_dim}], "
                    f"got {embeddings.shape}"
                )

            order = np.argsort(text_dates.to_numpy(dtype="datetime64[ns]"))
            self._text_dates = text_dates.iloc[order].reset_index(drop=True)
            self._text_embeddings = embeddings[order]
            print(
                "[Direct text] loaded offline embeddings: "
                f"rows={len(self._text_dates)}, dim={self.text_emb_dim}, "
                f"text_data_path={self.text_data_path}, "
                f"text_embedding_path={self.text_embedding_path}"
            )
        except Exception as error:
            self._text_dates = None
            self._text_embeddings = None
            print(
                "[Direct text][warning] failed to load text data; "
                f"falling back to text_mask=0. error={error}"
            )

    def _select_text_for_origin(self, forecast_origin_time):
        zero_embedding = np.zeros(self.text_emb_dim, dtype=np.float32)
        if self._text_dates is None or self._text_embeddings is None:
            return zero_embedding, np.float32(0.0), None

        origin = pd.Timestamp(forecast_origin_time)
        text_dates_np = self._text_dates.to_numpy(dtype="datetime64[ns]")
        right = np.searchsorted(
            text_dates_np,
            origin.to_datetime64(),
            side="right",
        )
        if right == 0:
            return zero_embedding, np.float32(0.0), None

        selected_indices = np.array([right - 1])
        if self.lookback_text_window not in {None, "", 0, "0"}:
            window = self.lookback_text_window
            if isinstance(window, (int, float)):
                window = pd.Timedelta(days=float(window))
            else:
                window = pd.Timedelta(window)
            left_time = (origin - window).to_datetime64()
            left = np.searchsorted(text_dates_np, left_time, side="left")
            selected_indices = np.arange(left, right)
            if len(selected_indices) == 0:
                return zero_embedding, np.float32(0.0), None

        selected_time = self._text_dates.iloc[selected_indices[-1]]
        if self.use_text_leakage_check and selected_time > origin:
            raise RuntimeError(
                "Text leakage detected: "
                f"text_time={selected_time} > forecast_origin_time={origin}"
            )
        embedding = self._text_embeddings[selected_indices].mean(axis=0)
        return embedding.astype(np.float32), np.float32(1.0), selected_time

    def _resolve_csv_path(self, root_path, data_name):
        root_path = root_path or ""
        legacy_names = {"env", "health", "energy"}
        if data_name in legacy_names:
            legacy_root = os.path.join(root_path, data_name)
            legacy_path = os.path.join(legacy_root, f"{data_name}.csv")
            if os.path.exists(legacy_path):
                return legacy_root, f"{data_name}.csv"

        canonical_name = TIMEMMD_DATASETS.get(str(data_name).lower())
        if canonical_name is not None:
            timemmd_root = os.path.join(root_path, "TimeMMD")
            timemmd_path = os.path.join(timemmd_root, f"{canonical_name}.csv")
            if os.path.exists(timemmd_path):
                return timemmd_root, f"{canonical_name}.csv"

        raise ValueError(
            f"Unsupported or missing MMDataset CSV for data_name={data_name}. "
            f"Expected legacy datasets {sorted(legacy_names)} or TimeMMD datasets "
            f"{sorted(TIMEMMD_DATASETS.values())} under {root_path}."
        )

    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(file_path)
        df_raw, self.date_col = normalize_csv_date_column(df_raw, file_path)
        time_prior = build_time_prior_features(
            df_raw,
            data_name=self.data_name,
            file_path=file_path,
        )
        self.domain_id = time_prior["domain_id"]
        self.domain_name = time_prior["domain_name"]
        self.time_granularity = time_prior["granularity"]
        self.time_feat_names = time_prior["feature_names"]
        self.time_focus_features = time_prior["focus_features"]

        required_cols = [self.target, "date", "prior_history_avg"]
        missing_cols = [col for col in required_cols if col not in df_raw.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {file_path}: {missing_cols}")

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_prior = self.scaler.transform(df_data_prior.values[:,-1].reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_prior = data_prior[border1:border2]
        self.time_feat = time_prior["time_feat"][border1:border2]
        self.time_feat_weight = time_prior["time_feat_weight"][border1:border2]

        self.date=df_raw[['date']][border1:border2].values
        self.start_date=(
            df_raw[['start_date']][border1:border2].values
            if 'start_date' in df_raw.columns else None
        )
        self.end_date=(
            df_raw[['end_date']][border1:border2].values
            if 'end_date' in df_raw.columns else None
        )
        self.fact=(
            df_raw[['fact']][border1:border2].values
            if 'fact' in df_raw.columns else None
        )
        self.preds=(
            df_raw[['preds']][border1:border2].values
            if 'preds' in df_raw.columns else None
        )
        text_col = self.text_name if self.text_name in df_raw.columns else None
        if text_col is None and 'preds' in df_raw.columns:
            text_col = 'preds'
        elif text_col is None and 'fact' in df_raw.columns:
            text_col = 'fact'
        self.text=(
            df_raw[[text_col]][border1:border2].values
            if text_col is not None else None
        )

    def _get_window_with_front_padding(self, data, begin, end):
        """Return a row window, padding zeros at the front when begin < 0."""
        if begin >= 0:
            return data[begin:end]

        valid_part = data[0:end]
        pad_shape = (abs(begin),) + data.shape[1:]
        pad_part = np.zeros(pad_shape, dtype=data.dtype)
        return np.concatenate([pad_part, valid_part], axis=0)
        
    def get_prior_y(self, indices):
        # If indices is a single integer index
        if isinstance(indices, (int, np.integer)):
            s_begin = indices % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len
            prior_y = self.data_prior[r_begin:r_end]
            return prior_y
        
        # If indices is a tensor or array
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()
            
        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends
        r_ends = r_begins + self.pred_len
        prior_y = np.array([self.data_prior[r_beg:r_end] for r_beg, r_end in zip(r_begins, r_ends)])
        return prior_y
    
    ## TODO: no need for pretraining => revise for retrieval task
    # def get_text(self, indices):
    #     if isinstance(indices, torch.Tensor):
    #         indices = indices.numpy()

    #     s_begins = indices % self.tot_len
    #     s_ends = s_begins + self.seq_len
    #     print(s_ends)
    #     text=np.array([self.text[s_end] for s_end in s_ends])[:,0]
    #     return text
    
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        if self.task_name == "forecasting":
            # Forecast fine-tuning consumes only the future prediction range.
            r_begin = s_end
            r_end = s_end + self.pred_len
            expected_time_len = self.pred_len
        else:
            # Preserve the existing context + future layout for other MMD tasks.
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            expected_time_len = self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1].reshape(1, -1)

        # Fix for r_begin < 0
        if r_begin >= 0:
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1].reshape(1, -1)
        else:
            valid_part = self.data_y[0:r_end, feat_id:feat_id + 1]  # shape: [r_end, 1]
            pad_len = abs(r_begin)
            pad_part = np.zeros((pad_len, 1))  # shape: [pad_len, 1]
            seq_y = np.concatenate([pad_part, valid_part], axis=0).reshape(1, -1)

        input_mask = np.ones((1, self.seq_len))
        prior_y = self.get_prior_y(index).reshape(1, -1)
        time_feat = self._get_window_with_front_padding(
            self.time_feat,
            r_begin,
            r_end,
        )
        time_feat_weight = self._get_window_with_front_padding(
            self.time_feat_weight,
            r_begin,
            r_end,
        )
        expected_target_len = (
            self.pred_len if self.task_name == "forecasting" else expected_time_len
        )
        assert seq_y.shape[-1] == expected_target_len, (
            f"forecast target length must be {expected_time_len}, got {seq_y.shape[-1]}"
        )
        assert time_feat.ndim == 2, f"time_feat must be [T, D], got {time_feat.shape}"
        assert time_feat_weight.shape == time_feat.shape, (
            "time_feat_weight must match time_feat shape, "
            f"got {time_feat_weight.shape} vs {time_feat.shape}"
        )
        assert time_feat.shape[0] == expected_time_len, (
            f"time_feat length must be forecast range={expected_time_len}, "
            f"got {time_feat.shape[0]}"
        )
        assert isinstance(self.domain_id, (int, np.integer)), (
            f"domain_id must be an integer, got {type(self.domain_id)}"
        )
        if not self._logged_time_prior_sample:
            if self.task_name == "forecasting":
                print(
                    "[Forecast dataset] "
                    f"history_len={seq_x.shape[-1]}, "
                    f"forecast_horizon={self.pred_len}, "
                    f"target_shape={seq_y.shape}, "
                    f"time_feat_shape={time_feat.shape}"
                )
            else:
                print(
                    "[MMDataset] time prior sample ready: "
                    f"time_feat_shape={time_feat.shape}, "
                    f"domain_id={int(self.domain_id)}"
                )
            self._logged_time_prior_sample = True

        forecast_origin_time = pd.Timestamp(self.date[s_end - 1, 0])
        if self.use_direct_text_forecast:
            text_emb, text_mask, text_time = self._select_text_for_origin(
                forecast_origin_time
            )
        else:
            text_emb, text_mask, text_time = None, None, None

        return TimeseriesData(
            timeseries=seq_x,
            input_mask=input_mask,
            forecast=seq_y,
            prior_y=prior_y,
            time_feat=time_feat,
            time_feat_weight=time_feat_weight,
            domain_id=np.array(self.domain_id, dtype=np.int64),
            text_emb=text_emb,
            text_mask=text_mask,
            text_time=text_time,
            forecast_origin_time=forecast_origin_time,
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
