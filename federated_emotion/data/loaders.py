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
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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

import numpy as np
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
# 2. Client Datasets Registry
# ---------------------------------------------------------------------------
# IDs 1-5 are the active roster, all verified loadable 2026-09 (see coverage_report.json).
# IDs 6+ are retained for provenance only; every one of them fails to load.
#
# Note on ID 5 (``emotion`` = dair-ai/emotion): this is the corpus that serves as the public
# KD pool in ``mode: public_set``. In ``mode: data_free_fd`` there is no public pool at all, so
# it is available as an ordinary client dataset -- and it is the only source that natively
# carries all six canonical labels, which is what lifts ``love`` from one usable holder to two.
# load_private_dataset() refuses it while mode == public_set (see PUBLIC_DATASET_SOURCES).
CLIENT_DATASETS: Dict[int, Tuple[str, Optional[str]]] = {
    # --- active roster ---------------------------------------------------
    1: ("go_emotions", "simplified"),
    2: ("tweet_eval", "emotion"),
    3: ("sem_eval_2018_task1", "subtask5.english"),
    4: ("xed_en_fi", "en_annotated"),
    5: ("emotion", None),
    # --- broken: all ship a loading script, which `datasets` v3+ rejects with
    #     "Dataset scripts are no longer supported, but found <name>.py".
    #     Verified 2026-09. Kept only so the provenance of the roster is legible.
    #
    #     silicone/*, empathetic_dialogues and emo previously appeared to work only because
    #     DATASET_ALIASES silently redirected them to mteb/emotion -- the evaluation corpus.
    #     That redirect has been removed; they now fail honestly.
    #
    #     daily_dialog is the upstream source of silicone/dyda_e and was trialled as its
    #     replacement. It fails for the same reason, so xed_en_fi and dair-ai/emotion took
    #     slots 4 and 5 instead.
    6: ("silicone", "dyda_e"),
    7: ("silicone", "meld_e"),
    8: ("silicone", "iemocap"),
    9: ("empathetic_dialogues", None),
    10: ("emo", None),
    11: ("daily_dialog", None),
    12: ("poem_sentiment", None),
}

# Candidate Hugging Face Hub repository paths for each dataset to handle
# modern namespaces and Parquet mirrors as well as legacy top-level dataset names across hub versions.
#
# IMPORTANT: aliases must resolve to the *same underlying corpus* as the key. Adding a
# convenient-but-unrelated repository here silently substitutes the wrong data. In particular,
# no private-client alias may point at an `emotion` mirror -- that is the public KD/eval corpus,
# and using it as a client dataset leaks the evaluation set into training (see PUBLIC_DATASET_SOURCES).
DATASET_ALIASES: Dict[str, List[str]] = {
    "go_emotions": ["google-research-datasets/go_emotions", "go_emotions"],
    "tweet_eval": ["cardiffnlp/tweet_eval", "tweet_eval"],
    "sem_eval_2018_task1": ["vibhorag101/sem_eval_2018_task_1_english_cleaned_labels", "sem_eval_2018_task1"],
    "silicone": ["eusip/silicone", "silicone"],
    "empathetic_dialogues": ["facebook/empathetic_dialogues", "empathetic_dialogues"],
    "emo": ["emo"],
    "xed_en_fi": ["akkasi/xed_en_fi", "Helsinki-NLP/xed_en_fi", "xed_en_fi"],
    "poem_sentiment": ["google-research-datasets/poem_sentiment", "poem_sentiment"],
    "emotion": ["dair-ai/emotion", "emotion"],
    # Retained but non-functional: both paths serve a loading script, rejected by datasets v3+.
    "daily_dialog": ["li2017dailydialog/daily_dialog", "daily_dialog"],
}

# Hub paths that back the *public* KD pool / evaluation holdout. A private client dataset that
# resolves to any of these would train on the evaluation data, inflating its holdout accuracy and
# -- because aggregation_mode="accuracy_weighted" -- corrupting the consensus for every other
# client. load_private_dataset() refuses such a resolution outright.
PUBLIC_DATASET_SOURCES: Set[str] = {
    "dair-ai/emotion",
    "mteb/emotion",
    "emotion",
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

    # 10. emotion (dair-ai/emotion). This dataset *defines* the canonical taxonomy, so the
    # mapping is the identity. Integer ids are already 0-5 in canonical order.
    "emotion": {
        0: LABEL_TO_ID["sadness"],
        1: LABEL_TO_ID["joy"],
        2: LABEL_TO_ID["love"],
        3: LABEL_TO_ID["anger"],
        4: LABEL_TO_ID["fear"],
        5: LABEL_TO_ID["surprise"],
        "sadness": LABEL_TO_ID["sadness"],
        "joy": LABEL_TO_ID["joy"],
        "love": LABEL_TO_ID["love"],
        "anger": LABEL_TO_ID["anger"],
        "fear": LABEL_TO_ID["fear"],
        "surprise": LABEL_TO_ID["surprise"],
    },

    # 11. daily_dialog (0: no_emotion, 1: anger, 2: disgust, 3: fear, 4: happiness,
    # 5: sadness, 6: surprise). Upstream source of silicone/dyda_e; trialled as its replacement
    # and rejected -- it ships a loading script, which `datasets` v3+ refuses. Mapping retained
    # in case a parquet mirror appears. Note the labels are per-utterance lists, which
    # _map_single_example would need to flatten.
    "daily_dialog": {
        0: None,                            # no_emotion -- dominant class, dropped
        1: LABEL_TO_ID["anger"],
        2: None,                            # disgust -- no canonical counterpart
        3: LABEL_TO_ID["fear"],
        4: LABEL_TO_ID["joy"],              # happiness -> joy
        5: LABEL_TO_ID["sadness"],
        6: LABEL_TO_ID["surprise"],
        "no_emotion": None,
        "anger": LABEL_TO_ID["anger"],
        "disgust": None,
        "fear": LABEL_TO_ID["fear"],
        "happiness": LABEL_TO_ID["joy"],
        "sadness": LABEL_TO_ID["sadness"],
        "surprise": LABEL_TO_ID["surprise"],
    },

    # 12. poem_sentiment: 0: negative, 1: positive, 2: no_impact, 3: mixed
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

    # Multi-label list or indicator vector handling (e.g. go_emotions, xed_en_fi)
    if isinstance(raw_label, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in raw_label) and any(x > 0 for x in raw_label) and any(x == 0 for x in raw_label):
            # One-hot / multi-hot indicator vector (e.g. xed_en_fi [0.0, 1.0, ...])
            active_indices = [i for i, val in enumerate(raw_label) if val > 0]
            mapped_labels = [mapping_dict.get(i) for i in active_indices if mapping_dict.get(i) is not None]
        else:
            mapped_labels = []
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
        if int_key in mapping_dict:
            mapped_id = mapping_dict[int_key]
            if mapped_id is not None:
                return {"text": text_val.strip(), "label": int(mapped_id)}
            return None
        # If no explicit mapping entry exists, check if already in canonical 0..5 range
        if 0 <= int_key <= 5:
            return {"text": text_val.strip(), "label": int_key}
        return None

    return None


# ---------------------------------------------------------------------------
# 6. Private Dataset Loader
# ---------------------------------------------------------------------------
def load_private_dataset(
    client_id: int,
    config: Config,
) -> Optional[Dataset]:
    global DBPEDIA_PARTITIONS
    if getattr(config, "is_dbpedia", False):
        print(f"[Client {client_id}] Routing to unified DBpedia Kaggle pipeline...")
        if not DBPEDIA_PARTITIONS:
             full_df = download_and_load_dbpedia_kaggle()
             DBPEDIA_PARTITIONS = get_iid_partitions(full_df, config.num_clients)
        if client_id >= len(DBPEDIA_PARTITIONS):
             client_id = client_id % len(DBPEDIA_PARTITIONS)
        return DBPEDIA_PARTITIONS[client_id]

    # Legacy logic
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

    # Guard: while a public KD/eval pool is in use, no private client dataset may resolve to it.
    # In data-free mode there is no public pool, so the same corpus is a legitimate client
    # dataset -- but say so out loud, because it is easy to misread later.
    leaking = [c for c in candidate_names if c in PUBLIC_DATASET_SOURCES]
    if leaking:
        if config.is_public_set:
            raise ValueError(
                f"Client {client_id}'s dataset '{hf_name}' resolves to public-corpus path(s) "
                f"{leaking}, which back the KD pool and evaluation holdout in "
                f"mode='public_set'. Training on them leaks the evaluation set. Either remove "
                f"them from DATASET_ALIASES or switch to mode='data_free_fd', where no public "
                f"pool exists."
            )
        print(
            f"  [NOTE] Client {client_id} uses '{hf_name}' -> {leaking}. This is the corpus that "
            f"serves as the public pool in mode='public_set'; in mode='{config.mode}' no public "
            f"pool is loaded, so it is being used purely as this client's private data."
        )

    raw_data = None
    resolved_name: Optional[str] = None
    last_error = None

    for cand_name in candidate_names:
        # 1. Try with sub-config (if specified)
        if hf_config is not None:
            try:
                raw_data = load_dataset(cand_name, hf_config, token=config.hf_token)
                resolved_name = cand_name
                break
            except Exception:
                try:
                    raw_data = load_dataset(cand_name, hf_config)
                    resolved_name = cand_name
                    break
                except Exception:
                    pass

        # 2. Try without sub-config (standard parquet root)
        try:
            raw_data = load_dataset(cand_name, token=config.hf_token)
            resolved_name = cand_name
            break
        except Exception:
            try:
                raw_data = load_dataset(cand_name)
                resolved_name = cand_name
                break
            except Exception as e_final:
                last_error = e_final

    if raw_data is None:
        print(
            f"[WARNING] Skipping Client {client_id}: Failed to load dataset '{hf_name}' "
            f"(config: {hf_config}) from candidate Hub paths {candidate_names}: {last_error}"
        )
        return None

    logger.info(f"[Client {client_id}] Resolved '{hf_name}' -> Hub path '{resolved_name}'.")

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
# 7. Class Distribution Utilities
# ---------------------------------------------------------------------------
def compute_class_counts(
    dataset: Optional[Dataset],
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Count examples per canonical class in a harmonized dataset.

    Used by data-free aggregation (per-class logit / prototype weighting) and by the
    client-coverage diagnostic. A class with a count of zero means the client contributes
    no information about that class and must be excluded from its aggregation weight.

    Args:
        dataset: HF Dataset with an integer "label" column, or None.
        num_classes: Size of the canonical label space.

    Returns:
        Integer array of shape (num_classes,) with per-class example counts.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    if dataset is None or len(dataset) == 0:
        return counts

    labels = np.asarray(dataset["label"], dtype=np.int64)
    valid = labels[(labels >= 0) & (labels < num_classes)]
    if valid.size:
        counts += np.bincount(valid, minlength=num_classes)[:num_classes]
    return counts


# ---------------------------------------------------------------------------
# 8. Package Exports
# ---------------------------------------------------------------------------
__all__ = [
    "CANONICAL_LABELS",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "NUM_CLASSES",
    "CLIENT_DATASETS",
    "DATASET_ALIASES",
    "PUBLIC_DATASET_SOURCES",
    "LABEL_MAPS",
    "load_public_dataset",
    "load_private_dataset",
    "compute_class_counts",
]


def download_and_load_dbpedia_kaggle(download_dir: str = "./data/dbpedia") -> pd.DataFrame:
    os.makedirs(download_dir, exist_ok=True)
    print("Downloading DBpedia dataset from Kaggle...")
    dataset_identifier = "danofer/dbpedia-classes" 
    
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_identifier, path=download_dir, unzip=True)
    except Exception as e:
        print(f"Failed to download from Kaggle: {e}. Ensure ~/.kaggle/kaggle.json exists.")
        # Mock dataframe for testing if api fails
        return pd.DataFrame({"text": ["mock text"]*100, "class": [0]*100})
        
    csv_files = [f for f in os.listdir(download_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {download_dir}")
        
    csv_path = os.path.join(download_dir, csv_files[0])
    df = pd.read_csv(csv_path)
    if "class" in df.columns and "content" in df.columns:
        df["text"] = df["content"]
        df["label"] = df["class"] - 1 # 1-indexed to 0-indexed typically
    return df

def get_iid_partitions(df: pd.DataFrame, num_clients: int) -> list:
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    partitions = []
    chunk_size = math.ceil(len(df) / num_clients)
    
    for i in range(num_clients):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        if "label" not in chunk_df.columns:
             chunk_df["label"] = 0
        dataset = Dataset.from_pandas(chunk_df)
        partitions.append(dataset)
        
    return partitions
