"""Main orchestration loop for the Federated Emotion Distillation pipeline.

This entrypoint executes:
1. Loading and validating global configuration (config.yaml).
2. Loading public KD transfer pool and public evaluation holdout datasets.
3. Pre-loading local private datasets across all clients (skipping unavailable ones).
4. Sequential federated communication rounds:
   - Client local fine-tuning (Supervised CE + Knowledge Distillation).
   - Client inference over public KD pool and evaluation on holdout set.
   - Server logit aggregation with dynamic temperature-scaled weighting.
   - Consensus soft label distribution persistence.
5. Resumption support (--resume) to bypass re-training when round checkpoints exist.
6. End-of-run performance summarization and round-over-round gain analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure root package directory is discoverable on sys.path
_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from typing import Any, Dict, List, Optional

import numpy as np

from federated_emotion.client import run_client_round
from federated_emotion.config import Config, load_config
from federated_emotion.data.loaders import (
    CLIENT_DATASETS,
    load_private_dataset,
    load_public_dataset,
)
from federated_emotion.eval_utils import MetricTracker, summarize_run
from federated_emotion.server import aggregate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("federated_emotion.main")


# ---------------------------------------------------------------------------
# 1. Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the orchestration pipeline."""
    parser = argparse.ArgumentParser(
        description="Heterogeneous Federated Emotion Classification with Knowledge Distillation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (defaults to config.yaml in package).",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Override total number of communication rounds.",
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=None,
        help="Override number of local training epochs per client.",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Override total number of federated clients.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume pipeline from existing client round checkpoints if present.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Checkpoint Cache Helpers for Resumption
# ---------------------------------------------------------------------------
def _load_cached_client_result(
    checkpoint_dir: Path,
    client_id: int,
    round_num: int,
) -> Optional[Dict[str, Any]]:
    """Attempt to reload saved logits and eval accuracy for a completed client round."""
    round_dir = checkpoint_dir / f"client_{client_id}" / f"round_{round_num}"
    logits_path = round_dir / "logits_kd.npy"
    meta_path = round_dir / "meta.json"

    if logits_path.exists() and meta_path.exists():
        try:
            logits = np.load(logits_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return {
                "client_id": client_id,
                "logits_on_kd_pool": logits,
                "eval_accuracy": float(meta.get("eval_accuracy", 0.0)),
            }
        except Exception as e:
            logger.warning(
                f"Failed to read cache for Client {client_id} Round {round_num}: {e}"
            )
            return None
    return None


def _cache_client_result(
    checkpoint_dir: Path,
    client_id: int,
    round_num: int,
    result: Dict[str, Any],
) -> None:
    """Save client logits and evaluation metadata alongside adapter weights."""
    round_dir = checkpoint_dir / f"client_{client_id}" / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    try:
        logits_path = round_dir / "logits_kd.npy"
        np.save(logits_path, result["logits_on_kd_pool"])

        meta_path = round_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "client_id": client_id,
                    "round_num": round_num,
                    "eval_accuracy": float(result["eval_accuracy"]),
                },
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning(
            f"Could not write cache metadata for Client {client_id} Round {round_num}: {e}"
        )


# ---------------------------------------------------------------------------
# 3. Main Pipeline Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(config: Config, resume: bool = False) -> None:
    """Execute the end-to-end federated distillation training and aggregation loop."""
    print("=" * 80)
    print("       HETEROGENEOUS FEDERATED DISTILLATION PIPELINE")
    print("=" * 80)
    print(f"Total Clients Configured : {config.num_clients}")
    print(f"Communication Rounds     : {config.num_rounds}")
    print(f"Local Epochs / Client    : {config.local_epochs}")
    print(f"Aggregation Mode         : {config.aggregation_mode} (T={config.aggregation_temperature})")
    print(f"Knowledge Distillation   : Lambda={config.kd_lambda}, T={config.kd_temperature}, Warmup={config.kd_warmup_rounds}")
    print(f"Checkpoints Path         : {config.checkpoint_dir}")
    print(f"Logs Path                : {config.log_dir}")
    print(f"Resume Mode              : {resume}")
    print("=" * 80 + "\n")

    checkpoint_path = Path(config.checkpoint_dir)
    log_path = Path(config.log_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    # Initialize Metric Tracker
    metric_tracker = MetricTracker(log_dir=str(log_path))

    # 1. Load Public Datasets
    print("[1/3] Loading Public Knowledge Distillation and Evaluation Datasets...")
    public_kd_pool, public_eval_holdout = load_public_dataset(config)
    print(
        f"  -> Public KD Pool Loaded        : {len(public_kd_pool)} instances"
    )
    print(
        f"  -> Public Eval Holdout Loaded  : {len(public_eval_holdout)} instances\n"
    )

    # 2. Pre-load Client Private Datasets
    print("[2/3] Pre-loading Client Private Datasets...")
    private_datasets: Dict[int, Any] = {}
    active_clients: List[int] = []

    # Map across targeted client IDs (active_client_ids if present, else 1..num_clients)
    target_client_ids = (
        config.active_client_ids
        if config.active_client_ids is not None
        else list(range(1, config.num_clients + 1))
    )

    for client_id in target_client_ids:
        ds = load_private_dataset(client_id, config)
        if ds is not None and len(ds) > 0:
            private_datasets[client_id] = ds
            active_clients.append(client_id)
        else:
            print(f"  [SKIPPED] Client {client_id:02d}: Dataset unavailable or empty.")

    if not active_clients:
        print("\n[CRITICAL ERROR] No active clients available with valid datasets. Exiting.")
        sys.exit(1)

    print(
        f"\nActive Clients for this Run ({len(active_clients)}/{len(target_client_ids)}): "
        f"{active_clients}\n"
    )

    # 3. Communication Rounds Loop
    print("[3/3] Commencing Federated Communication Rounds...\n")
    avg_soft_labels: Optional[np.ndarray] = None

    for round_num in range(1, config.num_rounds + 1):
        print(f"\n>>>>>>>> STARTING COMMUNICATION ROUND {round_num}/{config.num_rounds} <<<<<<<<")

        # If resuming and soft labels already saved from previous round, load them if needed
        if resume and avg_soft_labels is None and round_num > 1:
            prev_labels_file = log_path / f"round_{round_num - 1}_avg_soft_labels.npy"
            if prev_labels_file.exists():
                try:
                    avg_soft_labels = np.load(prev_labels_file)
                    print(f"  -> Resumed previous consensus soft labels from {prev_labels_file}")
                except Exception as e:
                    logger.warning(f"Could not load previous soft labels: {e}")

        client_results: List[Dict[str, Any]] = []

        # Execute training sequentially per client to manage GPU VRAM
        for client_id in active_clients:
            result = None

            # Check if resumption is possible for this client and round
            if resume:
                cached_res = _load_cached_client_result(checkpoint_path, client_id, round_num)
                if cached_res is not None:
                    print(
                        f"[Client {client_id:02d}] Checkpoint for Round {round_num} found. "
                        f"Resuming with cached accuracy: {cached_res['eval_accuracy'] * 100:.2f}%."
                    )
                    result = cached_res

            if result is None:
                result = run_client_round(
                    client_id=client_id,
                    round_num=round_num,
                    config=config,
                    avg_soft_labels_from_server=avg_soft_labels,
                    public_kd_pool=public_kd_pool,
                    public_eval_holdout=public_eval_holdout,
                    private_dataset=private_datasets[client_id],
                )

                if result is not None:
                    _cache_client_result(checkpoint_path, client_id, round_num, result)

            if result is not None:
                client_results.append(result)
            else:
                print(f"[WARNING] Skipping Client {client_id:02d} for Round {round_num} aggregation.")

        # Server Aggregation of Logits
        if client_results:
            avg_soft_labels = aggregate(client_results, config)

            # Persist consensus teacher soft labels
            if avg_soft_labels is not None:
                soft_labels_path = log_path / f"round_{round_num}_avg_soft_labels.npy"
                np.save(soft_labels_path, avg_soft_labels)
                print(f"  -> Persisted consensus teacher soft labels to: {soft_labels_path}")

            # Record round metrics
            round_accuracies = {
                int(r["client_id"]): float(r["eval_accuracy"]) for r in client_results
            }

            # Append to round_metrics.jsonl (one JSON line per client per round)
            jsonl_path = log_path / "round_metrics.jsonl"
            with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                for r in client_results:
                    record = {
                        "round": round_num,
                        "client_id": int(r["client_id"]),
                        "eval_accuracy": float(r["eval_accuracy"]),
                    }
                    f_jsonl.write(json.dumps(record) + "\n")

            metric_tracker.log_round(
                round_num=round_num,
                metrics={
                    "active_clients_count": len(client_results),
                    "client_accuracies": round_accuracies,
                    "mean_accuracy": float(np.mean(list(round_accuracies.values()))),
                },
            )
        else:
            print(f"[CRITICAL] Round {round_num}: No client succeeded. Moving to next round.")

    # 4. Final Run Summarization
    print("\n" + "=" * 80)
    print("       FEDERATED TRAINING COMPLETE - GENERATING SUMMARY")
    print("=" * 80)
    metric_tracker.save_summary()
    summarize_run(config)


# ---------------------------------------------------------------------------
# 4. Main Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)

    # Apply command-line overrides
    if args.num_rounds is not None:
        config.num_rounds = args.num_rounds
    if args.local_epochs is not None:
        config.local_epochs = args.local_epochs
    if args.num_clients is not None:
        config.num_clients = args.num_clients

    run_pipeline(config=config, resume=args.resume)


if __name__ == "__main__":
    main()
