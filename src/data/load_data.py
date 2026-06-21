

import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
import torch
from src.common import EVENT_MAP
from src.data.time_prior_features import build_time_prior_features
from tqdm import tqdm
keys_to_save = ['temperature', 'precipitation', 'relative_humidity', 'visibility', 'wind_u', 'wind_v', 'sky_code']
DATE_COLUMN_PRIORITY = ["date", "Date", "Month", "data", "start_date"]


def normalize_csv_date_column(df: pd.DataFrame, file_path: str = None):
    """Find the dataset date column and standardize it to df["date"]."""
    date_col = next((col for col in DATE_COLUMN_PRIORITY if col in df.columns), None)
    path_msg = file_path if file_path is not None else "<unknown>"

    print(f"[CSV date] file_path: {path_msg}")
    if date_col is None:
        print(
            "[CSV date][warning] no date column found. "
            f"Checked priority: {DATE_COLUMN_PRIORITY}"
        )
        return df, None

    df = df.copy()
    print(f"[CSV date] detected date column: {date_col}")
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")

    invalid_count = int(df["date"].isna().sum())
    if invalid_count > 0:
        print(
            "[CSV date][warning] "
            f"{invalid_count} rows have invalid dates and will be skipped."
        )
        df = df.loc[df["date"].notna()].reset_index(drop=True)

    if len(df) > 0:
        date_min = df["date"].min()
        date_max = df["date"].max()
        print(f"[CSV date] date range: {date_min} -> {date_max}")
    else:
        print("[CSV date][warning] no rows remain after dropping invalid dates.")

    print(f"[CSV date] invalid date rows: {invalid_count}")
    return df, date_col

def load_timeseries_from_json(split: str, dir_path: str, return_meta_data=False):
    file_path = os.path.join(dir_path, f'{split}.json')
    ts_data = []
    labels = []
    meta_data = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i, (k, v) in enumerate(list(data.items())):
        ts_sample = []
        for key in keys_to_save:
            if len(v[key]) >= 100:
                ts_sample.append(v[key])
            
        if len(ts_sample) > 0:
            ts_sample = np.array(ts_sample)
            ts_data.append(ts_sample)
            labels.append(v['event_type'])
            meta_data.append({"id": k, "station_id": v["station_id"], "mode": v['mode'], "location": v['location']})
                
    labels = np.array(labels).reshape(-1, 1)
    if return_meta_data:
        return ts_data, labels, meta_data
    else:
        return ts_data, labels


def load_npy_timeseries(split: str, dir_path: str, return_meta_data=False):
    file_path = os.path.join(dir_path, f'{split}_data')
    ts_data = []
    # Load all numbered npy files (timeseries data)
    npy_files = [f for f in os.listdir(file_path) if f.endswith('.npy') and f != 'labels.npy']
    npy_files.sort()  # Ensure consistent ordering
    
    for npy_file in npy_files:
        ts = np.load(os.path.join(file_path, npy_file))
        ts_data.append(ts)
        
    # Load labels
    labels = np.load(os.path.join(file_path, 'labels.npy'))
    
    return ts_data, labels

def load_forecasting_from_json(split: str, dir_path: str):
    file_path = os.path.join(dir_path, f'{split}.json')
    ts_data = []
    with open(file_path, 'r') as f:
        data = json.load(f)
    for i, (k, v) in enumerate(list(data.items())[:]):
        ts_sample = []
        for key in keys_to_save:
            if len(v[key]) >= 100:
                ts_sample.append(v[key])
            
        if len(ts_sample) > 0:
            ts_sample = np.array(ts_sample)
            ts_data.append(ts_sample)

    return ts_data


def generate_dsp(description):
    keys_to_save = ['temperature', 'precipitation', 'relative_humidity', 'visibility', 'wind_u', 'wind_v', 'sky_code']
    date = description["DATE"]
    location = description["location"]
    labels = description["labels"]
    prompt = f"Weather time series location: {location} Time range: {date} The weather is {labels}. {description[keys_to_save[0]]} \n {description[keys_to_save[1]]} \n {description[keys_to_save[2]]} \n {description[keys_to_save[3]]} \n {description[keys_to_save[4]]} \n {description[keys_to_save[5]]} \n {description[keys_to_save[6]]}"
    return prompt

def generate_channel_description(description):
    keys_to_save = ['temperature', 'precipitation', 'relative_humidity', 'visibility', 'wind_u', 'wind_v', 'sky_code']
    channel_description = [description[key] for key in keys_to_save]
    return channel_description
    
    
    
def generate_er(event):
    event_idx = int(event["event_type"])
    event_type = list(EVENT_MAP.keys())[event_idx]
    event_description = event["narrative"]
    prompt = f"The weather event is {event_type}. {event_description}"
    return prompt

def _encode_or_load_texts(texts, cache_path, text_encoder_name):
    """Encode retrieval text once and cache it next to the TimeMMD CSV."""
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu")

    from sentence_transformers import SentenceTransformer

    print(f"Generating text embeddings with {text_encoder_name}...")
    model = SentenceTransformer(text_encoder_name, trust_remote_code=True)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
    ).cpu()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings


def _nonempty_text(value):
    return "" if pd.isna(value) else str(value).strip()


def load_retrieval_from_timemmd_csv(
    split: str,
    root_path: str,
    domain_name: str,
    text_encoder_name: str,
    seq_len_channel: int = 180,
    n_channels: int = 7,
):
    """Build retrieval samples from a TimeMMD row-oriented CSV.

    Each row is paired with a causal numeric history ending at that row.  The
    returned payload has the same 5-item (train) / 8-item (test) contract as
    the original parquet loader; temporal metadata is returned separately for
    RetrievalDataset to attach to TimeseriesData.
    """
    normalized_root = os.path.normpath(root_path)
    if os.path.basename(normalized_root).lower() == "retrieval":
        dataset_root = os.path.dirname(normalized_root)
    elif os.path.basename(normalized_root).lower() == "dataset":
        dataset_root = normalized_root
    else:
        raise ValueError(
            "TimeMMD retrieval root must be dataset/retrieval or dataset; "
            f"got: {root_path}"
        )
    csv_path = os.path.join(dataset_root, "TimeMMD", f"{domain_name}.csv")
    display_path = csv_path.replace("\\", "/")
    print(f"[TimeMMD CSV Retrieval] Loading {display_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"TimeMMD CSV not found: {display_path}")

    df = pd.read_csv(csv_path)
    df, _ = normalize_csv_date_column(df, csv_path)
    if "OT" not in df.columns:
        raise ValueError(f"TimeMMD retrieval CSV must contain OT: {csv_path}")
    if len(df) < 2:
        raise ValueError(f"TimeMMD retrieval CSV needs at least 2 valid rows: {csv_path}")

    # OT is always channel zero. Other genuinely numeric columns are retained;
    # date/text columns are never accidentally converted into model channels.
    excluded = {"fact", "preds", "date", "Date", "Month", "start_date", "end_date"}
    numeric_cols = ["OT"]
    for col in df.columns:
        if col == "OT" or col in excluded:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            numeric_cols.append(col)
    numeric_cols = numeric_cols[:n_channels]

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.interpolate(limit_direction="both")
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    values = numeric.to_numpy(dtype=np.float32)
    if values.shape[1] < n_channels:
        pad_count = n_channels - values.shape[1]
        values = np.pad(values, ((0, 0), (0, pad_count)), mode="constant")
        channel_names = numeric_cols + [f"padding_channel_{i + 1}" for i in range(pad_count)]
    else:
        channel_names = numeric_cols

    raw_texts = []
    for idx, row in df.iterrows():
        fact = _nonempty_text(row.get("fact"))
        preds = _nonempty_text(row.get("preds"))
        date_text = row["date"].strftime("%Y-%m-%d")
        raw_texts.append(fact or preds or f"{domain_name} observation on {date_text}.")

    time_prior = build_time_prior_features(df, data_name=domain_name, file_path=csv_path)
    split_at = max(1, min(len(df) - 1, int(len(df) * 0.8)))
    if split == "train":
        row_indices = range(0, split_at)
    elif split in {"test", "val"}:
        row_indices = range(split_at, len(df))
    else:
        raise ValueError(f"Unsupported retrieval split: {split}")

    timeseries, descriptions, dates = [], [], []
    time_feat_windows, time_weight_windows = [], []
    for idx in row_indices:
        begin = max(0, idx - seq_len_channel + 1)
        timeseries.append(values[begin:idx + 1].T.copy())
        descriptions.append(raw_texts[idx])
        dates.append(df.iloc[idx]["date"])

        feat = time_prior["time_feat"][begin:idx + 1]
        weight = time_prior["time_feat_weight"][begin:idx + 1]
        pad_len = seq_len_channel - len(feat)
        time_feat_windows.append(np.pad(feat, ((pad_len, 0), (0, 0))))
        time_weight_windows.append(np.pad(weight, ((pad_len, 0), (0, 0))))

    channel_descriptions_per_sample = [
        [f"{domain_name} numeric channel: {name}" for name in channel_names]
        for _ in descriptions
    ]
    flat_channel_descriptions = [
        text for sample_texts in channel_descriptions_per_sample for text in sample_texts
    ]
    events = list(descriptions)
    labels = np.full((len(descriptions), 1), time_prior["domain_id"], dtype=np.int64)

    encoder_short_name = text_encoder_name.split("/")[-1]
    cache_dir = os.path.join(os.path.dirname(csv_path), ".retrieval_cache")
    cache_key = f"{domain_name}_{split}_{encoder_short_name}_l{seq_len_channel}_c{n_channels}"
    description_emb = _encode_or_load_texts(
        descriptions, os.path.join(cache_dir, f"description_{cache_key}.pt"), text_encoder_name
    )
    channel_description_emb = _encode_or_load_texts(
        flat_channel_descriptions,
        os.path.join(cache_dir, f"channels_{cache_key}.pt"),
        text_encoder_name,
    ).reshape(len(descriptions), n_channels, -1)
    event_emb = _encode_or_load_texts(
        events, os.path.join(cache_dir, f"event_{cache_key}.pt"), text_encoder_name
    )

    metadata = {
        "source": "timemmd_csv",
        "source_path": csv_path,
        "dates": dates,
        "time_feat": time_feat_windows,
        "time_feat_weight": time_weight_windows,
        "domain_id": time_prior["domain_id"],
        "channel_names": channel_names,
    }
    payload = (timeseries, description_emb, channel_description_emb, event_emb, labels)
    if split != "train":
        payload += (descriptions, flat_channel_descriptions, events)
    return payload, metadata


def load_retrieval_from_parquet(
    split: str,
    file_path: str,
    text_encoder_name: str,
    device="cuda:0",
    domain_name=None,
    seq_len_channel=180,
    n_channels=7,
    return_metadata=False,
):
    file_path_pq = os.path.join(file_path, split, f'{split}.parquet')
    if not os.path.exists(file_path_pq):
        if not domain_name:
            raise FileNotFoundError(
                f"{file_path_pq} not found and domain_name is not configured for TimeMMD fallback"
            )
        data_root = os.path.dirname(os.path.normpath(file_path))
        csv_path = os.path.join(data_root, "TimeMMD", f"{domain_name}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Neither retrieval parquet ({file_path_pq}) nor TimeMMD CSV ({csv_path}) exists"
            )
        payload, metadata = load_retrieval_from_timemmd_csv(
            split=split,
            root_path=file_path,
            domain_name=domain_name,
            text_encoder_name=text_encoder_name,
            seq_len_channel=seq_len_channel,
            n_channels=n_channels,
        )
        return (payload, metadata) if return_metadata else payload

    print(f"[Retrieval] loading parquet: {file_path_pq}")
    df = pd.read_parquet(file_path_pq)
    timeseries = []
    descriptions = []
    channel_descriptions = []
    events = []
    labels = []
    for ts_bytes in df["timeseries"]:
        ts = np.load(io.BytesIO(ts_bytes))
        timeseries.append(ts)
    for description in df["description"]:
        context = generate_dsp(description)
        channel_description = generate_channel_description(description)
        descriptions.append(context)
        channel_descriptions.extend(channel_description)
    for event in df["events"]:
        if event is not None:
            er = generate_er(event)
            events.append(er)
            labels.append(int(event["event_type"]))
        else:
            events.append("No severe weather event.")
            labels.append(-100)
    labels = np.array(labels).reshape(-1, 1)
    assert len(descriptions)*len(keys_to_save) == len(channel_descriptions)
    
    encoder_short_name = text_encoder_name.split("/")[-1]
    
    channel_description_emb_path = os.path.join(file_path+split, f'channel_description_emb_{encoder_short_name}.pt')
    if os.path.exists(channel_description_emb_path):
        channel_description_emb = torch.load(channel_description_emb_path,map_location="cpu")
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Generating channel description embeddings with {text_encoder_name}...")
        model = SentenceTransformer(text_encoder_name, trust_remote_code=True)
        channel_description_emb = model.encode(
            channel_descriptions, 
            batch_size=64,         
            show_progress_bar=True,
            convert_to_tensor=True   
        )
        torch.save(channel_description_emb, channel_description_emb_path)
    
    description_emb_path = os.path.join(file_path+split, f'description_emb_{encoder_short_name}.pt')
    if os.path.exists(description_emb_path):
        description_emb = torch.load(description_emb_path,map_location="cpu")
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Generating description embeddings with {text_encoder_name}...")
        model = SentenceTransformer(text_encoder_name, trust_remote_code=True)
        description_emb = model.encode(
            descriptions, 
            batch_size=64,         
            show_progress_bar=True,
            convert_to_tensor=True 
        )
        torch.save(description_emb, description_emb_path)
        
    
    event_emb_path = os.path.join(file_path+split, f'event_emb_{encoder_short_name}.pt')
    if os.path.exists(event_emb_path):
        event_emb = torch.load(event_emb_path,map_location="cpu")
    else:
        from sentence_transformers import SentenceTransformer
        print(f"Generating event embeddings with {text_encoder_name}...")
        model = SentenceTransformer(text_encoder_name, trust_remote_code=True)
        event_emb = model.encode(
            events, 
            batch_size=64,         
            show_progress_bar=True,
            convert_to_tensor=True   
        )
        torch.save(event_emb, event_emb_path)
    emb_dim = event_emb.shape[1]
    channel_description_emb = channel_description_emb.reshape(-1, len(keys_to_save), emb_dim)
    assert channel_description_emb.shape[0] == description_emb.shape[0] == event_emb.shape[0]
    if split == "train":
        payload = timeseries, description_emb, channel_description_emb, event_emb, labels
    else:
        payload = timeseries, description_emb, channel_description_emb, event_emb, labels, descriptions, channel_descriptions, events
    return (payload, None) if return_metadata else payload
