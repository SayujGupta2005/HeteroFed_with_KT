"""Dataset loading, label harmonization, and partitioning module for Federated Emotion Distillation.

This module provides:
1. CANONICAL_LABELS taxonomy (6 classes matching dair-ai/emotion).
2. Public dataset loader for Knowledge Distillation (KD) pool and evaluation holdout.
3. Explicit label mappings (LABEL_MAPS) harmonizing 10 heterogeneous emotion datasets.
4. Client dataset loader (load_private_dataset) with error tolerance, filtering,
   and configurable max dataset size capping.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_pkg_dir = _current_dir.parent
_parent_dir = _pkg_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from datasets import Dataset, concatenate_datasets, load_dataset

from federated_emotion.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Canonical Label Taxonomy
# ---------------------------------------------------------------------------
CANONICAL_LABELS: List[str] = [
    "sadness",   # 0
    "joy",       # 1
    "love",      # 2
    "anger",     # 3
    "fear",      # 4
    "surprise",  # 5
]

LABEL_TO_ID: Dict[str, int] = {label: i for i, label in enumerate(CANONICAL_LABELS)}
ID_TO_LABEL: Dict[int, str] = {i: label for i, label in enumerate(CANONICAL_LABELS)}
NUM_CLASSES: int = len(CANONICAL_LABELS)  # 6


# ---------------------------------------------------------------------------
# 2. Client Datasets Registry (Client IDs 1-10)
# ---------------------------------------------------------------------------
CLIENT_DATASETS: Dict[int, Tuple[str, Optional[str]]] = {
    1: ("go_emotions", "simplified"),
    2: ("tweet_eval", "emotion"),
    3: ("sem_eval_2018_task1", "subtask5.english"),
    4: ("silicone", "dyda_e"),
    5: ("silicone", "meld_e"),
    6: ("silicone", "iemocap"),
    7: ("empathetic_dialogues", None),
    8: ("emo", None),
    9: ("xed_en_fi", "en_annotated"),
    10: ("poem_sentiment", None),
}

# Candidate Hugging Face Hub repository paths for each dataset to handle
# modern namespaces as well as legacy top-level dataset names across hub versions.
DATASET_ALIASES: Dict[str, List[str]] = {
    "go_emotions": ["google-research-datasets/go_emotions", "go_emotions"],
    "tweet_eval": ["cardiffnlp/tweet_eval", "tweet_eval"],
    "sem_eval_2018_task1": ["sem_eval_2018_task1", "SetFit/sem_eval_2018_task1"],
    "silicone": ["silicone", "g-ronimo/silicone"],
    "empathetic_dialogues": ["facebook/empathetic_dialogues", "empathetic_dialogues"],
    "emo": ["emo", "monologg/emo"],
    "xed_en_fi": ["xed_en_fi", "TartuNLP/xed_en_fi"],
    "poem_sentiment": ["google-research-datasets/poem_sentiment", "poem_sentiment"],
}


# ---------------------------------------------------------------------------
# 3. Explicit Label Mappings for Heterogeneous Datasets
# ---------------------------------------------------------------------------
# None indicates the example should be discarded due to lack of a clean
# correspondence with the 6 canonical emotion classes.

LABEL_MAPS: Dict[str, Dict[Union[str, int], Optional[int]]] = {
    # 1. go_emotions: 27 emotions + neutral.
    # Map Ekman-adjacent / canonical emotions, drop the rest.
    "go_emotions": {
        # String mappings
        "admiration": None,
        "amusement": None,
        "anger": LABEL_TO_ID["anger"],          # 3
        "annoyance": None,
        "approval": None,
        "caring": None,
        "confusion": None,
        "curiosity": None,
        "desire": None,
        "disappointment": None,
        "disapproval": None,
        "disgust": None,
        "embarrassment": None,
        "excitement": None,
        "fear": LABEL_TO_ID["fear"],            # 4
        "gratitude": None,
        "grief": None,
        "joy": LABEL_TO_ID["joy"],              # 1
        "love": LABEL_TO_ID["love"],            # 2
        "nervousness": None,
        "optimism": None,
        "pride": None,
        "realization": None,
        "relief": None,
        "remorse": None,
        "sadness": LABEL_TO_ID["sadness"],      # 0
        "surprise": LABEL_TO_ID["surprise"],    # 5
        "neutral": None,
        # Integer IDs in go_emotions simplified
        0: None,                            # admiration
        1: None,                            # amusement
        2: LABEL_TO_ID["anger"],            # anger -> 3
        3: None,                            # annoyance
        4: None,                            # approval
        5: None,                            # caring
        6: None,                            # confusion
        7: None,                            # curiosity
        8: None,                            # desire
        9: None,                            # disappointment
        10: None,                           # disapproval
        11: None,                           # disgust
        12: None,                           # embarrassment
        13: None,                           # excitement
        14: LABEL_TO_ID["fear"],            # fear -> 4
        15: None,                           # gratitude
        16: None,                           # grief
        17: LABEL_TO_ID["joy"],             # joy -> 1
        18: LABEL_TO_ID["love"],            # love -> 2
        19: None,                           # nervousness
        20: None,                           # optimism
        21: None,                           # pride
        22: None,                           # realization
        23: None,                           # relief
        24: None,                           # remorse
        25: LABEL_TO_ID["sadness"],         # sadness -> 0
        26: LABEL_TO_ID["surprise"],        # surprise -> 5
        27: None,                           # neutral
    },

    # 2. tweet_eval (config: emotion): anger, joy, optimism, sadness
    "tweet_eval": {
        0: LABEL_TO_ID["anger"],            # 3
        1: LABEL_TO_ID["joy"],              # 1
        2: None,                            # optimism (dropped)
        3: LABEL_TO_ID["sadness"],          # 0
        "anger": LABEL_TO_ID["anger"],
        "joy": LABEL_TO_ID["joy"],
        "optimism": None,
        "sadness": LABEL_TO_ID["sadness"],
    },

    # 3. sem_eval_2018_task1: Multi-label binary columns handled in custom extractor below.
    "sem_eval_2018_task1": {
        "anger": LABEL_TO_ID["anger"],
        "fear": LABEL_TO_ID["fear"],
        "joy": LABEL_TO_ID["joy"],
        "love": LABEL_TO_ID["love"],
        "sadness": LABEL_TO_ID["sadness"],
        "surprise": LABEL_TO_ID["surprise"],
        "anticipation": None,
        "disgust": None,
        "optimism": None,
        "pessimism": None,
        "trust": None,
    },

    # 4. silicone (config: dyda_e - DailyDialog emotion)
    # 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise
    "silicone_dyda_e": {
        0: None,                            # no emotion
        1: LABEL_TO_ID["anger"],            # 3
        2: None,                            # disgust
        3: LABEL_TO_ID["fear"],             # 4
        4: LABEL_TO_ID["joy"],              # happiness -> joy (1)
        5: LABEL_TO_ID["sadness"],          # 0
        6: LABEL_TO_ID["surprise"],         # 5
        "no emotion": None,
        "anger": LABEL_TO_ID["anger"],
        "disgust": None,
        "fear": LABEL_TO_ID["fear"],
        "happiness": LABEL_TO_ID["joy"],
        "sadness": LABEL_TO_ID["sadness"],
        "surprise": LABEL_TO_ID["surprise"],
    },

    # 5. silicone (config: meld_e - MELD emotion)
    # 0: neutral, 1: surprise, 2: fear, 3: sadness, 4: joy, 5: disgust, 6: anger
    "silicone_meld_e": {
        0: None,                            # neutral
        1: LABEL_TO_ID["surprise"],         # 5
        2: LABEL_TO_ID["fear"],             # 4
        3: LABEL_TO_ID["sadness"],          # 0
        4: LABEL_TO_ID["joy"],              # 1
        5: None,                            # disgust
        6: LABEL_TO_ID["anger"],            # 3
        "neutral": None,
        "surprise": LABEL_TO_ID["surprise"],
        "fear": LABEL_TO_ID["fear"],
        "sadness": LABEL_TO_ID["sadness"],
        "joy": LABEL_TO_ID["joy"],
        "disgust": None,
        "anger": LABEL_TO_ID["anger"],
    },

    # 6. silicone (config: iemocap)
    # 0: neutral, 1: frustrated, 2: angry, 3: sad, 4: happy, 5: excited
    "silicone_iemocap": {
        0: None,                            # neutral
        1: None,                            # frustrated
        2: LABEL_TO_ID["anger"],            # 3
        3: LABEL_TO_ID["sadness"],          # 0
        4: LABEL_TO_ID["joy"],              # happy -> joy (1)
        5: None,                            # excited
        "neutral": None,
        "frustrated": None,
        "angry": LABEL_TO_ID["anger"],
        "sad": LABEL_TO_ID["sadness"],
        "happy": LABEL_TO_ID["joy"],
        "excited": None,
    },

    # 7. empathetic_dialogues (32 emotions)
    # Map clearly matching ones, drop the remaining 26+ labels.
    "empathetic_dialogues": {
        "joyful": LABEL_TO_ID["joy"],              # 1
        "sad": LABEL_TO_ID["sadness"],             # 0
        "afraid": LABEL_TO_ID["fear"],             # 4
        "terrified": LABEL_TO_ID["fear"],          # 4
        "angry": LABEL_TO_ID["anger"],             # 3
        "furious": LABEL_TO_ID["anger"],           # 3
        "surprised": LABEL_TO_ID["surprise"],      # 5
        # Dropped non-canonical emotions
        "caring": None,
        "annoyed": None,
        "guilty": None,
        "lonely": None,
        "grateful": None,
        "hopeful": None,
        "excited": None,
        "disgusted": None,
        "anxious": None,
        "confident": None,
        "jealous": None,
        "proud": None,
        "embarrassed": None,
        "content": None,
        "devastated": None,
        "impressed": None,
        "nostalgic": None,
        "sentimental": None,
        "disappointed": None,
        "ashamed": None,
        "prepared": None,
        "anticipating": None,
        "apprehensive": None,
        "trusting": None,
        "faithful": None,
        "neutral": None,
    },

    # 8. emo: 0: others, 1: happy, 2: sad, 3: angry
    "emo": {
        0: None,                            # others
        1: LABEL_TO_ID["joy"],              # happy -> joy (1)
        2: LABEL_TO_ID["sadness"],          # 0
        3: LABEL_TO_ID["anger"],            # 3
        "others": None,
        "happy": LABEL_TO_ID["joy"],
        "sad": LABEL_TO_ID["sadness"],
        "angry": LABEL_TO_ID["anger"],
    },

    # 9. xed_en_fi (Plutchik 8 emotions: 1: anger, 2: anticipation, 3: disgust,
    # 4: fear, 5: joy, 6: sadness, 7: surprise, 8: trust)
    "xed_en_fi": {
        1: LABEL_TO_ID["anger"],            # 3
        2: None,                            # anticipation
        3: None,                            # disgust
        4: LABEL_TO_ID["fear"],             # 4
        5: LABEL_TO_ID["joy"],              # 1
        6: LABEL_TO_ID["sadness"],          # 0
        7: LABEL_TO_ID["surprise"],         # 5
        8: None,                            # trust
        "anger": LABEL_TO_ID["anger"],
        "anticipation": None,
        "disgust": None,
        "fear": LABEL_TO_ID["fear"],
        "joy": LABEL_TO_ID["joy"],
        "sadness": LABEL_TO_ID["sadness"],
        "surprise": LABEL_TO_ID["surprise"],
        "trust": None,
    },

    # 10. poem_sentiment: 0: negative, 1: positive, 2: no_impact, 3: mixed
    # NOTE: poem_sentiment is a sentiment dataset without fine-grained emotion labels.
    # We apply a rough proxy mapping: positive -> joy (1), negative -> sadness (0),
    # dropping ambiguous categories (no_impact, mixed).
    "poem_sentiment": {
        0: LABEL_TO_ID["sadness"],          # negative -> sadness proxy (0)
        1: LABEL_TO_ID["joy"],              # positive -> joy proxy (1)
        2: None,                            # no_impact
        3: None,                            # mixed
        "negative": LABEL_TO_ID["sadness"],
        "positive": LABEL_TO_ID["joy"],
        "no_impact": None,
        "mixed": None,
    },
}


# ---------------------------------------------------------------------------
# 4. Public Dataset Loader
# ---------------------------------------------------------------------------
def load_public_dataset(config: Config) -> Tuple[Dataset, Dataset]:
    """Load, unify, and partition the public dataset (dair-ai/emotion) for KD and eval holdout.

    Loads the dataset from Hugging Face, shuffles it with config.seed_base,
    and slices it into disjoint subsets:
    - kd_pool: config.public_kd_pool_size examples
    - eval_holdout: config.public_eval_holdout_size examples

    Args:
        config: Populated Config dataclass.

    Returns:
        Tuple of (kd_pool, eval_holdout) HF Dataset objects with columns ["text", "label"].
    """
    token = config.hf_token
    total_required = config.public_kd_pool_size + config.public_eval_holdout_size

    try:
        raw_ds = load_dataset("dair-ai/emotion", token=token)
    except Exception as e:
        logger.warning(f"Could not load 'dair-ai/emotion' with token; retrying without token: {e}")
        raw_ds = load_dataset("dair-ai/emotion")

    # Combine all available splits to form a comprehensive pool
    split_keys = [k for k in ["train", "validation", "test"] if k in raw_ds]
    if split_keys:
        full_dataset = concatenate_datasets([raw_ds[k] for k in split_keys])
    else:
        full_dataset = raw_ds[list(raw_ds.keys())[0]]

    # Ensure canonical columns
    if "text" not in full_dataset.column_names:
        text_candidates = ["Tweet", "utterance", "sentence", "content"]
        for cand in text_candidates:
            if cand in full_dataset.column_names:
                full_dataset = full_dataset.rename_column(cand, "text")
                break

    # Retain only text and label
    columns_to_keep = {"text", "label"}
    cols_to_remove = [col for col in full_dataset.column_names if col not in columns_to_keep]
    if cols_to_remove:
        full_dataset = full_dataset.remove_columns(cols_to_remove)

    # Shuffle with seed_base
    shuffled = full_dataset.shuffle(seed=config.seed_base)

    if len(shuffled) < total_required:
        raise ValueError(
            f"Public dataset has {len(shuffled)} examples, which is fewer than required "
            f"({config.public_kd_pool_size} for KD pool + {config.public_eval_holdout_size} for holdout = {total_required})."
        )

    # Disjoint slices
    kd_pool_indices = list(range(0, config.public_kd_pool_size))
    eval_holdout_indices = list(
        range(config.public_kd_pool_size, config.public_kd_pool_size + config.public_eval_holdout_size)
    )

    kd_pool = shuffled.select(kd_pool_indices)
    eval_holdout = shuffled.select(eval_holdout_indices)

    return kd_pool, eval_holdout


# ---------------------------------------------------------------------------
# 5. Helper Utilities for Extracting Text & Harmonizing Labels
# ---------------------------------------------------------------------------
def _find_text_column(column_names: List[str]) -> Optional[str]:
    """Find the most likely text column in a dataset schema."""
    candidates = [
        "text",
        "Tweet",
        "Utterance",
        "utterance",
        "sentence",
        "verse_text",
        "content",
        "dialogue",
    ]
    for cand in candidates:
        if cand in column_names:
            return cand
    return None


def _map_single_example(
    example: Dict[str, Any],
    dataset_key: str,
    text_col: str,
) -> Optional[Dict[str, Any]]:
    """Harmonize a single dataset row to canonical text and int label 0-5.

    Returns:
        Dict with {"text": str, "label": int} if successfully mapped, else None.
    """
    text_val = example.get(text_col)
    if text_val is None or not isinstance(text_val, str) or not text_val.strip():
        return None

    # Handle sem_eval_2018_task1 multi-binary format
    if dataset_key == "sem_eval_2018_task1":
        active_canonical: List[int] = []
        for canon_label in CANONICAL_LABELS:
            val = example.get(canon_label, 0)
            if val in (1, "1", True):
                active_canonical.append(LABEL_TO_ID[canon_label])
        if len(active_canonical) == 1:
            return {"text": text_val.strip(), "label": active_canonical[0]}
        return None

    mapping_dict = LABEL_MAPS.get(dataset_key, {})

    # Determine raw label value
    raw_label = None
    for l_key in ["label", "labels", "Label", "context", "emotion", "tags", "turn3"]:
        if l_key in example:
            raw_label = example[l_key]
            break

    if raw_label is None:
        return None

    # Multi-label list handling (e.g. go_emotions, xed_en_fi)
    if isinstance(raw_label, (list, tuple)):
        mapped_labels: List[int] = []
        for item in raw_label:
            m = mapping_dict.get(item)
            if m is not None and m not in mapped_labels:
                mapped_labels.append(m)
        # Only accept unambiguous single canonical emotion matches
        if len(mapped_labels) == 1:
            return {"text": text_val.strip(), "label": mapped_labels[0]}
        return None

    # Single label lookup
    if isinstance(raw_label, str):
        cleaned_str = raw_label.strip().lower()
        mapped_id = mapping_dict.get(cleaned_str)
        if mapped_id is not None:
            return {"text": text_val.strip(), "label": int(mapped_id)}
        return None

    if isinstance(raw_label, (int, float)):
        int_key = int(raw_label)
        mapped_id = mapping_dict.get(int_key)
        if mapped_id is not None:
            return {"text": text_val.strip(), "label": int(mapped_id)}
        return None

    return None


# ---------------------------------------------------------------------------
# 6. Private Dataset Loader
# ---------------------------------------------------------------------------
def load_private_dataset(
    client_id: int,
    config: Config,
) -> Optional[Dataset]:
    """Load, harmonize, filter, and cap a private dataset for a federated client.

    Maps client_id (1-10) to the corresponding dataset via CLIENT_DATASETS,
    applies explicit LABEL_MAPS to drop non-canonical examples, and caps dataset size
    at config.private_dataset_max_size.

    Args:
        client_id: Client identifier (supports 1-10 as well as 0-9 index).
        config: Global Config dataclass instance.

    Returns:
        Hugging Face Dataset with columns ["text", "label"] (int 0-5), or None if loading fails.
    """
    # Normalize client_id to 1-10 registry
    reg_id = client_id
    if reg_id not in CLIENT_DATASETS:
        if (client_id + 1) in CLIENT_DATASETS:
            reg_id = client_id + 1
        else:
            logger.warning(f"Client ID {client_id} is not configured in CLIENT_DATASETS registry.")
            return None

    hf_name, hf_config = CLIENT_DATASETS[reg_id]

    # Resolve dataset lookup key for LABEL_MAPS
    if hf_name == "silicone":
        dataset_key = f"silicone_{hf_config}"
    else:
        dataset_key = hf_name

    logger.info(
        f"[Client {client_id}] Loading private dataset '{hf_name}' (config: {hf_config})..."
    )

    # Try candidate repository aliases (e.g. namespaced paths first, then legacy names)
    candidate_names = DATASET_ALIASES.get(hf_name, [hf_name])
    raw_data = None
    last_error = None

    for cand_name in candidate_names:
        try:
            if hf_config is not None:
                raw_data = load_dataset(cand_name, hf_config, token=config.hf_token)
            else:
                raw_data = load_dataset(cand_name, token=config.hf_token)
            break
        except Exception as e_with_token:
            try:
                # Fallback without token in case token was invalid or unneeded
                if hf_config is not None:
                    raw_data = load_dataset(cand_name, hf_config)
                else:
                    raw_data = load_dataset(cand_name)
                break
            except Exception as e_without_token:
                last_error = e_without_token

    if raw_data is None:
        print(
            f"[WARNING] Skipping Client {client_id}: Failed to load dataset '{hf_name}' "
            f"(config: {hf_config}) from candidate Hub paths {candidate_names}: {last_error}"
        )
        return None

    # Concatenate available splits (e.g. train + validation)
    if isinstance(raw_data, dict) or hasattr(raw_data, "keys"):
        split_keys = [k for k in ["train", "validation", "test"] if k in raw_data]
        if not split_keys:
            split_keys = list(raw_data.keys())
        if not split_keys:
            print(f"[WARNING] Client {client_id}: Dataset '{hf_name}' has no usable splits.")
            return None
        combined_ds = concatenate_datasets([raw_data[k] for k in split_keys])
    else:
        combined_ds = raw_data

    # Find text column
    text_col = _find_text_column(combined_ds.column_names)
    if text_col is None:
        print(
            f"[WARNING] Client {client_id}: Could not find a recognized text column in '{hf_name}' "
            f"columns: {combined_ds.column_names}."
        )
        return None

    # Filter and harmonize rows
    valid_texts: List[str] = []
    valid_labels: List[int] = []

    for ex in combined_ds:
        mapped = _map_single_example(ex, dataset_key, text_col)
        if mapped is not None:
            valid_texts.append(mapped["text"])
            valid_labels.append(mapped["label"])

    if not valid_texts:
        print(
            f"[WARNING] Client {client_id}: No valid canonical emotion examples found after "
            f"applying label mapping to '{hf_name}'."
        )
        return None

    # Create new harmonized Dataset
    harmonized_ds = Dataset.from_dict({
        "text": valid_texts,
        "label": valid_labels,
    })

    # Shuffle with client-specific seed
    client_seed = config.get_seed_for_client(client_id)
    shuffled_ds = harmonized_ds.shuffle(seed=client_seed)

    # Cap dataset size
    max_size = min(len(shuffled_ds), config.private_dataset_max_size)
    final_ds = shuffled_ds.select(range(max_size))

    print(
        f"[Client {client_id}] Successfully prepared {len(final_ds)} examples from '{hf_name}' "
        f"(config: {hf_config}) with canonical classes."
    )
    return final_ds


# ---------------------------------------------------------------------------
# 7. Package Exports
# ---------------------------------------------------------------------------
__all__ = [
    "CANONICAL_LABELS",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "NUM_CLASSES",
    "CLIENT_DATASETS",
    "LABEL_MAPS",
    "load_public_dataset",
    "load_private_dataset",
]
