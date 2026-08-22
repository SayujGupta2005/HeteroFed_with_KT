"""Standalone verification and audit script for all datasets in the federated emotion pipeline.

Executes a complete health check across:
1. Public reference dataset (dair-ai/emotion) for KD pool and holdout evaluation.
2. All 10 client private datasets (1-10) with canonical 6-class label distribution audits.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

# Ensure package root is discoverable
_current_dir = Path(__file__).resolve().parent
_pkg_dir = _current_dir.parent
_parent_dir = _pkg_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from federated_emotion.config import load_config
from federated_emotion.data.loaders import (
    CANONICAL_LABELS,
    CLIENT_DATASETS,
    ID_TO_LABEL,
    load_private_dataset,
    load_public_dataset,
)


def verify_all_datasets() -> bool:
    """Run comprehensive verification across public and all 10 private datasets."""
    print("\n" + "=" * 90)
    print("        FEDERATED EMOTION PIPELINE — DATASET HEALTH & INTEGRITY AUDIT")
    print("=" * 90)

    config = load_config()
    all_ok = True

    # -----------------------------------------------------------------------
    # 1. Audit Public Dataset
    # -----------------------------------------------------------------------
    print("\n[1/2] Auditing Public Dataset ('dair-ai/emotion')...")
    try:
        kd_pool, eval_holdout = load_public_dataset(config)
        kd_labels = collections.Counter(kd_pool["label"])
        eval_labels = collections.Counter(eval_holdout["label"])

        print(f"  ✅ Public KD Pool       : {len(kd_pool)} instances (Target: {config.public_kd_pool_size})")
        print(f"     Class Distribution  : {dict(sorted(kd_labels.items()))}")
        print(f"  ✅ Public Eval Holdout  : {len(eval_holdout)} instances (Target: {config.public_eval_holdout_size})")
        print(f"     Class Distribution  : {dict(sorted(eval_labels.items()))}")
    except Exception as e:
        print(f"  ❌ Public Dataset FAILED: {e}")
        all_ok = False

    # -----------------------------------------------------------------------
    # 2. Audit All 10 Client Private Datasets
    # -----------------------------------------------------------------------
    print("\n[2/2] Auditing 10 Private Client Datasets...")
    print("-" * 90)
    header = f"{'Client':<8} | {'Dataset (Config)':<35} | {'Status':<10} | {'Samples':<8} | {'Class Breakdown (0-5)'}"
    print(header)
    print("-" * 90)

    for client_id in range(1, 11):
        hf_name, hf_config = CLIENT_DATASETS[client_id]
        ds_name = f"{hf_name}" + (f" ({hf_config})" if hf_config else "")
        ds_name_trunc = (ds_name[:32] + "..") if len(ds_name) > 35 else ds_name

        try:
            ds = load_private_dataset(client_id, config)
            if ds is not None and len(ds) > 0:
                label_counts = collections.Counter(ds["label"])
                counts_str = ", ".join(f"{c}:{label_counts.get(c, 0)}" for c in range(6))
                status = "✅ OK"
                print(f"Client {client_id:02d} | {ds_name_trunc:<35} | {status:<10} | {len(ds):<8} | {counts_str}")
            else:
                status = "❌ EMPTY"
                all_ok = False
                print(f"Client {client_id:02d} | {ds_name_trunc:<35} | {status:<10} | 0        | ---")
        except Exception as e:
            status = "❌ ERROR"
            all_ok = False
            print(f"Client {client_id:02d} | {ds_name_trunc:<35} | {status:<10} | 0        | {e}")

    print("-" * 90)
    print("Canonical Emotion Class Mapping:")
    for idx, name in ID_TO_LABEL.items():
        print(f"  Class {idx}: {name}")
    print("=" * 90 + "\n")

    if all_ok:
        print("🎉 ALL DATASETS VERIFIED SUCCESSFULLY AND READY FOR FEDERATED TRAINING!\n")
    else:
        print("⚠️ Some datasets encountered issues. Review log output above.\n")

    return all_ok


if __name__ == "__main__":
    verify_all_datasets()
