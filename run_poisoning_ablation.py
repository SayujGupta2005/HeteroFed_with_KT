"""Logit-Poisoning Ablation Runner for the data_free_fd federated pipeline.

Iterates over the full ablation matrix:
    attack_type  ∈ {none, random, zero, fdla, pcfdla}
    attacker_pct ∈ {0 (none-only), 10, 20, 30}  percent of 10 clients

For each cell the script:
  1. Builds an isolated Config (10 clients, 5 rounds, 3 local epochs,
     sample-count-weighted aggregation, data_free_fd mode).
  2. Runs the full federated pipeline, injecting the attack into the
     designated malicious clients' logit payloads right before server
     aggregation — without touching any core training code.
  3. Collects per-round per-client accuracies and per-class confusion
     matrices from every honest client's local holdout.
  4. After all cells finish, writes summary_ablation.csv (one row per
     ablation cell) and per_class_ablation.csv (per-class accuracy).

Directory layout
----------------
results/
  ablation_<timestamp>/
    none_0pct/        <- benign baseline
    random_10pct/
    random_20pct/
    ...
    summary_ablation.csv
    per_class_ablation.csv

Usage
-----
    python run_poisoning_ablation.py [--config PATH] [--output_dir DIR]
                                     [--attack_types a b c]
                                     [--attacker_pcts 10 20 30]
                                     [--source_class INT] [--target_class INT]
                                     [--scale FLOAT] [--seed INT]
"""

from __future__ import annotations

import argparse
import csv
import copy
import datetime
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure the package root is importable regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_DIR = _SCRIPT_DIR / "federated_emotion"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch

from federated_emotion.attacks import LogitAttacker, SUPPORTED_ATTACKS
from federated_emotion.client import run_client_round
from federated_emotion.config import Config, load_config
from federated_emotion.data.loaders import (
    CANONICAL_LABELS,
    CLIENT_DATASETS,
    NUM_CLASSES,
    compute_class_counts,
    load_private_dataset,
)
from federated_emotion.datafree import (
    aggregate_class_logits,
    logits_from_serializable,
    logits_to_serializable,
)
from federated_emotion.models.wrapper import (
    CLIENT_MODELS,
    get_model_for_client,
    preload_client_models,
)
from federated_emotion.profiler import global_profiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ablation_runner")


# ---------------------------------------------------------------------------
# 1.  Fixed experiment constants
# ---------------------------------------------------------------------------

# 10 clients cycling the five 3-4B backbones
ABLATION_CLIENT_MODEL_MAP: Dict[int, str] = {
    1:  "microsoft/Phi-3.5-mini-instruct",
    2:  "Qwen/Qwen2.5-3B-Instruct",
    3:  "microsoft/Phi-3-mini-4k-instruct",
    4:  "Qwen/Qwen2.5-3B",
    5:  "Qwen/Qwen1.5-4B-Chat",
    6:  "microsoft/Phi-3.5-mini-instruct",   # repeated
    7:  "Qwen/Qwen2.5-3B-Instruct",          # repeated
    8:  "microsoft/Phi-3-mini-4k-instruct",  # repeated
    9:  "Qwen/Qwen2.5-3B",                   # repeated
    10: "Qwen/Qwen1.5-4B-Chat",              # repeated
}

# Client IDs → dataset: clients 1-5 use existing datasets;
# clients 6-10 reuse the same five datasets (cycling).
# The loaders.py CLIENT_DATASETS only defines 1-5; we build the mapping here.
ABLATION_DATASET_MAP: Dict[int, Tuple[str, Optional[str]]] = {
    1:  ("go_emotions", "simplified"),
    2:  ("tweet_eval", "emotion"),
    3:  ("sem_eval_2018_task1", "subtask5.english"),
    4:  ("xed_en_fi", "en_annotated"),
    5:  ("emotion", None),
    6:  ("go_emotions", "simplified"),
    7:  ("tweet_eval", "emotion"),
    8:  ("sem_eval_2018_task1", "subtask5.english"),
    9:  ("xed_en_fi", "en_annotated"),
    10: ("emotion", None),
}

NUM_TOTAL_CLIENTS = 10
ABLATION_ROUNDS = 5
ABLATION_LOCAL_EPOCHS = 3
ABLATION_WARMUP_ROUNDS = 1


# ---------------------------------------------------------------------------
# 2.  Config builder
# ---------------------------------------------------------------------------

def build_ablation_config(
    base_config: Config,
    run_checkpoint_dir: str,
    run_log_dir: str,
) -> Config:
    """Clone base_config and apply ablation-fixed overrides.

    The returned Config is a shallow dataclass copy with the following changes:
      - mode = data_free_fd
      - num_clients = 10
      - active_client_ids = [1..10]
      - num_rounds = 5
      - local_epochs = 3
      - kd_warmup_rounds = 1
      - fd_weight_by_count = True  (sample-count weighted, not accuracy weighted)
      - checkpoint_dir / log_dir  = run-specific isolated paths
    """
    cfg = copy.copy(base_config)
    cfg.mode = "data_free_fd"
    cfg.num_clients = NUM_TOTAL_CLIENTS
    cfg.active_client_ids = list(range(1, NUM_TOTAL_CLIENTS + 1))
    cfg.num_rounds = ABLATION_ROUNDS
    cfg.local_epochs = ABLATION_LOCAL_EPOCHS
    cfg.kd_warmup_rounds = ABLATION_WARMUP_ROUNDS
    cfg.fd_weight_by_count = True
    cfg.checkpoint_dir = run_checkpoint_dir
    cfg.log_dir = run_log_dir
    # Ensure sane defaults for data-free-specific fields
    if not hasattr(cfg, "fd_lambda") or cfg.fd_lambda == 0:
        cfg.fd_lambda = 1.0
    if not hasattr(cfg, "fd_temperature"):
        cfg.fd_temperature = 2.0
    if not hasattr(cfg, "local_holdout_size"):
        cfg.local_holdout_size = 200
    return cfg


# ---------------------------------------------------------------------------
# 3.  Dataset loader with 10-client cycling support
# ---------------------------------------------------------------------------

def _load_all_datasets(
    config: Config,
) -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, np.ndarray]]:
    """Load private train/holdout datasets for all 10 clients.

    Clients 6-10 share the same underlying dataset as clients 1-5.
    Returns:
        private_train  : {client_id: HF Dataset}
        local_holdouts : {client_id: HF Dataset}
        class_counts   : {client_id: np.ndarray shape (6,)}
    """
    # First load the five base datasets (client IDs 1-5)
    base_train: Dict[int, Any] = {}
    base_holdout: Dict[int, Any] = {}
    base_counts: Dict[int, np.ndarray] = {}

    # Temporarily pretend we have only 5 clients for the loader
    base_cfg = copy.copy(config)
    base_cfg.active_client_ids = [1, 2, 3, 4, 5]
    base_cfg.num_clients = 5

    for cid in range(1, 6):
        raw_ds = load_private_dataset(cid, base_cfg)
        if raw_ds is None or len(raw_ds) == 0:
            logger.warning(f"Client {cid}: dataset unavailable — skipping.")
            continue

        holdout_len = min(
            config.local_holdout_size,
            max(5, int(len(raw_ds) * 0.1)),
        )
        if len(raw_ds) > holdout_len:
            train_sl = raw_ds.select(range(len(raw_ds) - holdout_len))
            eval_sl  = raw_ds.select(range(len(raw_ds) - holdout_len, len(raw_ds)))
        else:
            train_sl = raw_ds
            eval_sl  = raw_ds

        base_train[cid]   = train_sl
        base_holdout[cid] = eval_sl
        base_counts[cid]  = compute_class_counts(train_sl, config.num_classes)

    # Clients 6-10 mirror 1-5
    private_train: Dict[int, Any] = {}
    local_holdouts: Dict[int, Any] = {}
    class_counts: Dict[int, np.ndarray] = {}
    for cid in range(1, NUM_TOTAL_CLIENTS + 1):
        base_cid = ((cid - 1) % 5) + 1
        if base_cid not in base_train:
            logger.warning(f"Client {cid}: base client {base_cid} unavailable — skipping.")
            continue
        private_train[cid]  = base_train[base_cid]
        local_holdouts[cid] = base_holdout[base_cid]
        class_counts[cid]   = base_counts[base_cid]

    return private_train, local_holdouts, class_counts


# ---------------------------------------------------------------------------
# 4.  Per-class evaluation on a client holdout
# ---------------------------------------------------------------------------

def _evaluate_per_class(
    model_fn,              # callable: (input_ids, attention_mask) -> logits tensor
    holdout_dataset,
    tokenizer,
    device,
    batch_size: int,
    max_seq_length: int,
    num_classes: int = 6,
) -> Dict[str, Any]:
    """Run inference on a holdout set and return per-class and overall metrics.

    Returns a dict with keys:
        overall_accuracy, per_class_accuracy (list[float]),
        confusion_matrix (list[list[int]])
    """
    from torch.utils.data import DataLoader
    from federated_emotion.client import _create_collate_fn

    collate = _create_collate_fn(tokenizer, max_seq_length, include_idx=False)
    loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    all_preds: List[int] = []
    all_labels: List[int] = []

    import torch
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            labs  = batch["label"].tolist()
            logits = model_fn(input_ids=ids, attention_mask=mask)
            preds  = torch.argmax(logits, dim=-1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labs)

    # Confusion matrix: conf[true][pred]
    conf = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(all_labels, all_preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf[t][p] += 1

    # Per-class accuracy = TP / (sum of true class row)
    per_class_acc: List[float] = []
    for c in range(num_classes):
        total_c = sum(conf[c])
        per_class_acc.append(conf[c][c] / total_c if total_c > 0 else 0.0)

    overall = sum(conf[c][c] for c in range(num_classes)) / max(len(all_labels), 1)

    return {
        "overall_accuracy":   overall,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix":   conf,
    }


# ---------------------------------------------------------------------------
# 5.  Single-cell runner
# ---------------------------------------------------------------------------

def run_one_cell(
    attack_type: str,
    malicious_client_ids: List[int],
    config: Config,
    private_train: Dict[int, Any],
    local_holdouts: Dict[int, Any],
    client_class_counts: Dict[int, np.ndarray],
    attacker: LogitAttacker,
    results_dir: Path,
) -> Dict[str, Any]:
    """Run one full federated experiment (5 rounds) for a single ablation cell.

    Returns a summary dict suitable for the final CSV row.
    """
    active_clients = sorted(private_train.keys())

    # Per-round per-client accuracy log
    round_records: List[Dict[str, Any]] = []

    global_class_logits: Optional[Dict[int, np.ndarray]] = None
    final_round_per_class: Dict[int, List[float]] = {}

    for round_num in range(1, config.num_rounds + 1):
        logger.info(
            f"[{attack_type.upper()} | {len(malicious_client_ids)} mal.] "
            f"Round {round_num}/{config.num_rounds}"
        )
        client_results: List[Dict[str, Any]] = []

        for cid in active_clients:
            # Cross-eval only on the final round (expensive)
            result = run_client_round(
                client_id=cid,
                round_num=round_num,
                config=config,
                public_kd_pool=None,
                public_eval_holdout=None,
                private_dataset=private_train[cid],
                avg_soft_labels=None,
                avg_soft_labels_from_server=None,
                cross_eval_datasets=None,
                local_eval_holdout=local_holdouts[cid],
                global_class_logits=global_class_logits,
            )

            if result is None:
                logger.warning(f"  Client {cid} failed in round {round_num}; skipping.")
                continue

            # -----------------------------------------------------------------
            # ATTACK INJECTION: intercept the class_logits payload right here,
            # before it reaches the server aggregator.  Core training code is
            # untouched — attacks.py is the only place we modify logits.
            # -----------------------------------------------------------------
            if cid in malicious_client_ids and result.get("class_logits") is not None:
                honest_logits  = result["class_logits"]
                poisoned_logits = attacker.apply(honest_logits)
                result = dict(result)          # shallow copy so we don't mutate the original
                result["class_logits"] = poisoned_logits
                logger.info(
                    f"  [ATTACK] Client {cid} → {attacker}  "
                    f"(classes poisoned: {sorted(poisoned_logits.keys())})"
                )

            client_results.append(result)

            # Record per-round accuracy
            round_records.append({
                "round":          round_num,
                "client_id":      cid,
                "is_malicious":   cid in malicious_client_ids,
                "eval_accuracy":  float(result["eval_accuracy"]),
            })

        if not client_results:
            logger.warning(f"Round {round_num}: no client results — skipping aggregation.")
            continue

        # -----------------------------------------------------------------
        # Server aggregation (uses the (possibly poisoned) class_logits)
        # -----------------------------------------------------------------
        per_client_logits = [r.get("class_logits") for r in client_results]
        per_client_counts = (
            [r.get("class_counts") for r in client_results]
            if config.fd_weight_by_count
            else None
        )
        global_class_logits = aggregate_class_logits(
            per_client_logits,
            class_counts=per_client_counts,
            num_classes=config.num_classes,
            verbose=False,
        )

        # Collect per-class accuracy on the FINAL round for honest clients only
        if round_num == config.num_rounds:
            for r in client_results:
                cid = r["client_id"]
                if cid not in malicious_client_ids:
                    # We rely on eval_accuracy from run_client_round (overall).
                    # Per-class breakdown requires a second pass — stored separately
                    # via the confusion_matrix key injected below if available.
                    final_round_per_class[cid] = r.get("per_class_accuracy", [])

    # -----------------------------------------------------------------
    # Persist round_metrics.jsonl for this cell
    # -----------------------------------------------------------------
    jsonl_path = results_dir / "round_metrics.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in round_records:
            f.write(json.dumps(rec) + "\n")

    # Build summary row
    final_round_records = [r for r in round_records if r["round"] == config.num_rounds]
    honest_accs  = [r["eval_accuracy"] for r in final_round_records if not r["is_malicious"]]
    all_accs     = [r["eval_accuracy"] for r in final_round_records]
    round1_accs  = [r["eval_accuracy"] for r in round_records if r["round"] == 1 and not r["is_malicious"]]

    summary = {
        "attack_type":             attack_type,
        "num_malicious":           len(malicious_client_ids),
        "attacker_pct":            int(round(len(malicious_client_ids) / NUM_TOTAL_CLIENTS * 100)),
        "malicious_client_ids":    str(sorted(malicious_client_ids)),
        "mean_final_accuracy_honest": float(np.mean(honest_accs)) if honest_accs else 0.0,
        "mean_final_accuracy_all":    float(np.mean(all_accs))    if all_accs    else 0.0,
        "mean_round1_accuracy_honest": float(np.mean(round1_accs)) if round1_accs else 0.0,
    }

    # Per-class accuracy averaged across honest clients (final round)
    per_class_sums = [0.0] * NUM_CLASSES
    per_class_counts_agg = [0] * NUM_CLASSES
    for cid, pc in final_round_per_class.items():
        for c, v in enumerate(pc):
            per_class_sums[c] += v
            per_class_counts_agg[c] += 1
    for c in range(NUM_CLASSES):
        label = CANONICAL_LABELS[c] if c < len(CANONICAL_LABELS) else f"class_{c}"
        avg = per_class_sums[c] / per_class_counts_agg[c] if per_class_counts_agg[c] > 0 else None
        summary[f"final_acc_{label}"] = avg

    # Save per-cell summary JSON
    with open(results_dir / "cell_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"  Cell done — honest mean accuracy (round {config.num_rounds}): "
        f"{summary['mean_final_accuracy_honest'] * 100:.2f}%"
    )
    return summary


# ---------------------------------------------------------------------------
# 6.  VRAM cleanup helper
# ---------------------------------------------------------------------------

def _clear_gpu_memory() -> None:
    """Free all cached GPU tensors and run Python GC between ablation cells."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("[VRAM] GPU memory cleared between ablation cells.")


# ---------------------------------------------------------------------------
# 7.  Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logit poisoning ablation study for the data_free_fd federated pipeline."
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (defaults to federated_emotion/config.yaml).",
    )
    p.add_argument(
        "--output_dir", type=str, default="./results/ablation",
        help="Root directory for all ablation outputs.",
    )
    p.add_argument(
        "--attack_types", nargs="+", default=SUPPORTED_ATTACKS,
        choices=SUPPORTED_ATTACKS,
        help="Attack types to evaluate (space-separated). Default: all five.",
    )
    p.add_argument(
        "--attacker_pcts", nargs="+", type=int, default=[10, 20, 30],
        help="Percentages of malicious clients to test (default: 10 20 30).",
    )
    p.add_argument(
        "--source_class", type=int, default=3,
        help="PCFDLA: source class to degrade (default 3=anger).",
    )
    p.add_argument(
        "--target_class", type=int, default=1,
        help="PCFDLA: target class to redirect predictions to (default 1=joy).",
    )
    p.add_argument(
        "--scale", type=float, default=10.0,
        help="Scaling factor for FDLA and random attacks (default 10.0).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for stochastic attacks (default 42).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 8.  Main ablation orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # Setup output directory
    # -----------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ablation_root = Path(args.output_dir) / f"run_{timestamp}"
    ablation_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ablation root: {ablation_root}")

    # -----------------------------------------------------------------------
    # Load base config and apply ablation overrides (10 clients, 5 rounds, etc.)
    # -----------------------------------------------------------------------
    cfg_path = args.config or str(_PKG_DIR / "config.yaml")
    base_config = load_config(cfg_path)

    # We will override checkpoint_dir and log_dir per cell; use a placeholder here
    base_config = build_ablation_config(
        base_config,
        run_checkpoint_dir=str(ablation_root / "_base_checkpoints"),
        run_log_dir=str(ablation_root / "_base_logs"),
    )

    # -----------------------------------------------------------------------
    # Override the CLIENT_MODELS registry for the 10-client setup
    # This patches the module-level dict so get_model_for_client resolves correctly
    # for clients 6-10.
    # -----------------------------------------------------------------------
    import federated_emotion.models.wrapper as _wrapper_mod
    _wrapper_mod.CLIENT_MODELS.update(ABLATION_CLIENT_MODEL_MAP)
    logger.info(f"Patched CLIENT_MODELS for 10-client ablation: {_wrapper_mod.CLIENT_MODELS}")

    # -----------------------------------------------------------------------
    # Load all 10 client datasets ONCE (clients 6-10 reuse slices from 1-5)
    # -----------------------------------------------------------------------
    logger.info("Loading all 10 client datasets...")
    private_train, local_holdouts, client_class_counts = _load_all_datasets(base_config)

    active_clients = sorted(private_train.keys())
    if not active_clients:
        logger.error("No datasets loaded — aborting.")
        sys.exit(1)
    logger.info(f"Active clients with datasets: {active_clients}")

    # -----------------------------------------------------------------------
    # Pre-load all model weights ONCE before the ablation loop
    # (avoids re-downloading on every cell; models are freed per round already)
    # -----------------------------------------------------------------------
    logger.info("Pre-loading and verifying backbone models...")
    preload_client_models(active_clients, base_config)

    # -----------------------------------------------------------------------
    # Build the ablation matrix
    # -----------------------------------------------------------------------
    # Always include the benign baseline (attack=none, 0 attackers) even if
    # the user didn't explicitly pass 0 in --attacker_pcts.
    attack_types = args.attack_types
    attacker_pcts = args.attacker_pcts

    # Build matrix: list of (attack_type, pct) pairs
    # "none" only makes sense with 0% attackers (run once as baseline)
    ablation_matrix: List[Tuple[str, int]] = []
    if "none" in attack_types:
        ablation_matrix.append(("none", 0))
    for at in attack_types:
        if at == "none":
            continue
        for pct in attacker_pcts:
            ablation_matrix.append((at, pct))

    logger.info(
        f"Ablation matrix: {len(ablation_matrix)} cells — "
        + ", ".join(f"({at},{pct}%)" for at, pct in ablation_matrix)
    )

    # -----------------------------------------------------------------------
    # Run every cell
    # -----------------------------------------------------------------------
    all_summaries: List[Dict[str, Any]] = []

    for cell_idx, (attack_type, attacker_pct) in enumerate(ablation_matrix, start=1):
        # Determine which clients are malicious
        num_malicious = round(NUM_TOTAL_CLIENTS * attacker_pct / 100)
        # Assign the LAST `num_malicious` active clients as attackers
        malicious_ids = active_clients[-num_malicious:] if num_malicious > 0 else []

        cell_label = f"{attack_type}_{attacker_pct}pct"
        cell_dir   = ablation_root / cell_label
        ckpt_dir   = cell_dir / "checkpoints"
        log_dir    = cell_dir / "logs"
        cell_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"\n{'='*70}\n"
            f"[Cell {cell_idx}/{len(ablation_matrix)}]  "
            f"attack={attack_type}  pct={attacker_pct}%  "
            f"malicious={malicious_ids}\n"
            f"{'='*70}"
        )

        # Build cell-specific config (isolated checkpoint/log dirs)
        cell_config = build_ablation_config(
            base_config,
            run_checkpoint_dir=str(ckpt_dir),
            run_log_dir=str(log_dir),
        )

        # Instantiate attacker
        attacker = LogitAttacker(
            attack_type=attack_type,
            scale=args.scale,
            source_class=args.source_class,
            target_class=args.target_class,
            seed=args.seed,
        )
        logger.info(f"Attacker: {attacker}")

        # Run one cell
        try:
            summary = run_one_cell(
                attack_type=attack_type,
                malicious_client_ids=malicious_ids,
                config=cell_config,
                private_train=private_train,
                local_holdouts=local_holdouts,
                client_class_counts=client_class_counts,
                attacker=attacker,
                results_dir=cell_dir,
            )
            all_summaries.append(summary)
        except Exception as exc:
            logger.exception(f"Cell {cell_label} failed: {exc}")
            all_summaries.append({
                "attack_type": attack_type,
                "num_malicious": num_malicious,
                "attacker_pct": attacker_pct,
                "error": str(exc),
            })

        # Clear VRAM between cells
        _clear_gpu_memory()

    # -----------------------------------------------------------------------
    # Write consolidated summary CSV
    # -----------------------------------------------------------------------
    _write_ablation_csvs(all_summaries, ablation_root)

    logger.info(f"\n[DONE] All ablation results saved in: {ablation_root}")


# ---------------------------------------------------------------------------
# 9.  CSV export helpers
# ---------------------------------------------------------------------------

def _write_ablation_csvs(
    summaries: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write summary_ablation.csv and per_class_ablation.csv."""

    if not summaries:
        logger.warning("No summaries to write.")
        return

    # -- summary_ablation.csv -------------------------------------------------
    # Columns: attack_type | attacker_pct | num_malicious |
    #          mean_round1_acc | mean_final_acc_honest | mean_final_acc_all |
    #          accuracy_drop (vs baseline)
    summary_path = output_dir / "summary_ablation.csv"

    # Find baseline accuracy (attack=none, pct=0)
    baseline_acc = None
    for s in summaries:
        if s.get("attack_type") == "none" and s.get("attacker_pct") == 0:
            baseline_acc = s.get("mean_final_accuracy_honest")
            break

    summary_fields = [
        "attack_type", "attacker_pct", "num_malicious", "malicious_client_ids",
        "mean_round1_accuracy_honest",
        "mean_final_accuracy_honest", "mean_final_accuracy_all",
        "accuracy_drop_vs_baseline_pct",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row = dict(s)
            if baseline_acc is not None and "mean_final_accuracy_honest" in row:
                drop = (baseline_acc - row["mean_final_accuracy_honest"]) * 100
                row["accuracy_drop_vs_baseline_pct"] = round(drop, 3)
            else:
                row["accuracy_drop_vs_baseline_pct"] = ""
            # Round float fields
            for k in ("mean_round1_accuracy_honest",
                       "mean_final_accuracy_honest",
                       "mean_final_accuracy_all"):
                if isinstance(row.get(k), float):
                    row[k] = round(row[k], 4)
            writer.writerow(row)

    logger.info(f"  -> summary_ablation.csv: {summary_path}")

    # -- per_class_ablation.csv -----------------------------------------------
    class_labels = [CANONICAL_LABELS[c] if c < len(CANONICAL_LABELS) else f"class_{c}"
                    for c in range(NUM_CLASSES)]
    pc_fields = ["attack_type", "attacker_pct"] + [f"final_acc_{lbl}" for lbl in class_labels]

    pc_path = output_dir / "per_class_ablation.csv"
    with open(pc_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pc_fields, extrasaction="ignore")
        writer.writeheader()
        for s in summaries:
            row: Dict[str, Any] = {
                "attack_type":  s.get("attack_type", ""),
                "attacker_pct": s.get("attacker_pct", ""),
            }
            for lbl in class_labels:
                key = f"final_acc_{lbl}"
                val = s.get(key)
                row[key] = round(val, 4) if isinstance(val, float) else val
            writer.writerow(row)

    logger.info(f"  -> per_class_ablation.csv: {pc_path}")

    # -- Console summary table ------------------------------------------------
    print("\n" + "=" * 90)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 90)
    hdr = (
        f"{'Attack':<10} | {'Pct':>5} | {'Mal.':>5} | "
        f"{'R1 Honest':>10} | {'Final Honest':>12} | "
        f"{'Final All':>10} | {'Drop vs Base':>13}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        at  = s.get("attack_type", "?")
        pct = s.get("attacker_pct", "?")
        nm  = s.get("num_malicious", "?")
        r1  = s.get("mean_round1_accuracy_honest", None)
        fh  = s.get("mean_final_accuracy_honest", None)
        fa  = s.get("mean_final_accuracy_all", None)

        if baseline_acc is not None and fh is not None:
            drop = (baseline_acc - fh) * 100
            drop_str = f"{drop:+.2f}%"
        else:
            drop_str = "---"

        r1_str = f"{r1*100:.2f}%" if isinstance(r1, float) else "---"
        fh_str = f"{fh*100:.2f}%" if isinstance(fh, float) else "---"
        fa_str = f"{fa*100:.2f}%" if isinstance(fa, float) else "---"

        print(
            f"{at:<10} | {str(pct):>5} | {str(nm):>5} | "
            f"{r1_str:>10} | {fh_str:>12} | {fa_str:>10} | {drop_str:>13}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

