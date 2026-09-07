"""Client / class coverage diagnostic for the federated emotion pipeline.

Loads every configured client's private dataset, harmonizes it onto the 6 canonical emotion
classes, and reports a client x class count matrix. Runs on CPU only -- no models are loaded.

Why this matters
----------------
Both federation modes aggregate *per class*:

- ``public_set``   weights each client once and averages its soft labels over the public pool.
- ``data_free_fd`` averages each client's per-class logit vector across clients.

A client holding zero examples of class ``c`` still contributes a vector for ``c`` under an
unweighted mean, which is pure noise. A client holding three examples contributes an equally
weighted, near-useless estimate. This script surfaces those cases before any GPU time is spent,
so clients can be chosen for coverage rather than by registry ID.

Usage
-----
    python -m federated_emotion.analyze_coverage
    python -m federated_emotion.analyze_coverage --config path/to/config.yaml
    python -m federated_emotion.analyze_coverage --clients 1 2 3 4 5 6 7 8 9 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from typing import Any, Dict, List

import numpy as np

from federated_emotion.config import Config, load_config
from federated_emotion.data.loaders import (
    CANONICAL_LABELS,
    CLIENT_DATASETS,
    NUM_CLASSES,
    compute_class_counts,
    load_private_dataset,
)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def _model_name_for(client_id: int) -> str:
    """Look up a client's backbone name, tolerating a missing torch/peft install.

    This diagnostic is meant to run on a laptop with no GPU, so the heavyweight model
    wrapper is imported lazily and its absence is not fatal.
    """
    try:
        from federated_emotion.models.wrapper import get_model_for_client

        return get_model_for_client(client_id)
    except Exception:
        return "<unavailable>"

#: Below this many examples, a client's estimate for a class is treated as unreliable.
THIN_CLASS_THRESHOLD: int = 20


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Report the client x class coverage matrix for the federation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to the packaged config).",
    )
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        default=None,
        help="Client IDs to inspect. Defaults to config.active_client_ids, else the full registry.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="coverage_report.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--thin",
        type=int,
        default=THIN_CLASS_THRESHOLD,
        help=f"Flag classes with fewer than this many examples (default {THIN_CLASS_THRESHOLD}).",
    )
    return parser.parse_args()


def gather_counts(client_ids: List[int], config: Config) -> Dict[int, Dict[str, Any]]:
    """Load each client's dataset and record its per-class counts.

    Args:
        client_ids: Client IDs to inspect.
        config: Populated Config.

    Returns:
        Mapping of client_id -> {"dataset", "model", "counts", "total"} for clients that loaded.
        Clients whose dataset failed to load are omitted.
    """
    results: Dict[int, Dict[str, Any]] = {}

    for cid in client_ids:
        ds_name, ds_config = CLIENT_DATASETS.get(cid, ("<unregistered>", None))
        label = ds_name + (f"/{ds_config}" if ds_config else "")
        print(f"[{cid:02d}] loading {label} ...", flush=True)

        try:
            ds = load_private_dataset(cid, config)
        except Exception as e:
            print(f"     FAILED: {e}")
            continue

        if ds is None or len(ds) == 0:
            print("     UNAVAILABLE (skipped)")
            continue

        counts = compute_class_counts(ds, config.num_classes)
        results[cid] = {
            "dataset": label,
            "model": _model_name_for(cid),
            "counts": counts.tolist(),
            "total": int(counts.sum()),
        }
        print(f"     ok: {int(counts.sum())} examples")

    return results


def print_matrix(results: Dict[int, Dict[str, Any]], thin: int) -> None:
    """Print the client x class count matrix with per-class and per-client summaries."""
    if not results:
        print("\nNo clients loaded successfully. Nothing to report.")
        return

    labels = CANONICAL_LABELS[:NUM_CLASSES]
    col_w = max(9, max(len(x) for x in labels) + 2)

    header = f"{'client':<8}{'dataset':<34}" + "".join(f"{x:>{col_w}}" for x in labels) + f"{'total':>9}"
    print("\n" + "=" * len(header))
    print("CLIENT x CLASS COVERAGE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    matrix = []
    for cid in sorted(results):
        info = results[cid]
        counts = info["counts"]
        matrix.append(counts)
        cells = ""
        for n in counts:
            mark = "!" if n == 0 else ("~" if n < thin else "")
            cells += f"{str(n) + mark:>{col_w}}"
        print(f"{cid:<8}{info['dataset'][:33]:<34}{cells}{info['total']:>9}")

    arr = np.asarray(matrix, dtype=np.int64)
    print("-" * len(header))
    totals = "".join(f"{int(v):>{col_w}}" for v in arr.sum(axis=0))
    print(f"{'TOTAL':<8}{'':<34}{totals}{int(arr.sum()):>9}")

    holders = (arr > 0).sum(axis=0)
    usable = (arr >= thin).sum(axis=0)
    print(f"{'holders':<8}{'clients with >0':<34}" + "".join(f"{int(v):>{col_w}}" for v in holders))
    print(f"{'usable':<8}{f'clients with >={thin}':<34}" + "".join(f"{int(v):>{col_w}}" for v in usable))
    print("=" * len(header))
    print(f"legend: ! = class absent   ~ = fewer than {thin} examples\n")

    # Actionable warnings ---------------------------------------------------
    n_clients = arr.shape[0]
    for j, name in enumerate(labels):
        if holders[j] == 0:
            print(f"  [FATAL] class '{name}' has NO examples anywhere. It cannot be learned.")
        elif holders[j] == 1:
            print(f"  [WARN ] class '{name}' exists on only 1 of {n_clients} clients -- no consensus possible.")
        elif usable[j] < holders[j]:
            thin_ids = [sorted(results)[i] for i in range(n_clients) if 0 < arr[i, j] < thin]
            print(f"  [WARN ] class '{name}' is thin (<{thin}) on client(s) {thin_ids}; "
                  f"their per-class estimates will be noisy.")

    for i, cid in enumerate(sorted(results)):
        absent = [labels[j] for j in range(len(labels)) if arr[i, j] == 0]
        if absent:
            print(f"  [WARN ] client {cid} holds no examples of: {', '.join(absent)}")

    imbalance = arr.max(axis=1) / np.clip(arr.min(axis=1), 1, None)
    for i, cid in enumerate(sorted(results)):
        if arr[i].min() > 0 and imbalance[i] > 50:
            print(f"  [WARN ] client {cid} is severely imbalanced (max/min class ratio "
                  f"{imbalance[i]:.0f}x).")
    print()


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    config = load_config(args.config)

    if args.clients is not None:
        client_ids = args.clients
    elif config.active_client_ids is not None:
        client_ids = config.active_client_ids
    else:
        client_ids = sorted(CLIENT_DATASETS.keys())

    print(f"mode            : {config.mode}")
    print(f"canonical classes: {config.num_classes} {CANONICAL_LABELS[:config.num_classes]}")
    print(f"clients to check : {client_ids}")
    print(f"cap per client   : {config.private_dataset_max_size}\n")

    results = gather_counts(client_ids, config)
    print_matrix(results, args.thin)

    report = {
        "mode": config.mode,
        "num_classes": config.num_classes,
        "canonical_labels": CANONICAL_LABELS[: config.num_classes],
        "private_dataset_max_size": config.private_dataset_max_size,
        "thin_threshold": args.thin,
        "requested_clients": client_ids,
        "loaded_clients": sorted(results.keys()),
        "unavailable_clients": [c for c in client_ids if c not in results],
        "clients": {str(k): v for k, v in results.items()},
    }
    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {out_path.resolve()}")

    if report["unavailable_clients"]:
        print(f"Unavailable clients: {report['unavailable_clients']}")


if __name__ == "__main__":
    main()
