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

import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from federated_emotion.client import run_client_round
from federated_emotion.config import Config, load_config
from federated_emotion.profiler import global_profiler
from federated_emotion.data.loaders import (
    CANONICAL_LABELS,
    CLIENT_DATASETS,
    compute_class_counts,
    load_private_dataset,
    load_public_dataset,
)
from federated_emotion.datafree import (
    aggregate_class_logits,
    logits_from_serializable,
    logits_to_serializable,
)
from federated_emotion.eval_utils import (
    MetricTracker,
    print_and_export_cross_dataset_matrix,
    summarize_run,
)
from federated_emotion.models.wrapper import (
    CLIENT_MODELS,
    get_model_for_client,
    preload_client_models,
)
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
    """Attempt to reload a completed client round's upload payload and eval accuracy.

    Restores whichever payload the mode uses: ``logits_on_kd_pool`` for public_set, or
    ``class_logits`` + ``class_counts`` for data_free_fd.
    """
    round_dir = checkpoint_dir / f"client_{client_id}" / f"round_{round_num}"
    logits_path = round_dir / "logits_kd.npy"
    class_logits_path = round_dir / "class_logits.json"
    meta_path = round_dir / "meta.json"

    if not meta_path.exists():
        return None
    if not (logits_path.exists() or class_logits_path.exists()):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        out: Dict[str, Any] = {
            "client_id": client_id,
            "mode": meta.get("mode"),
            "eval_accuracy": float(meta.get("eval_accuracy", 0.0)),
            "logits_on_kd_pool": None,
            "class_logits": None,
            "class_counts": None,
            "cross_eval_accuracies": {},
        }

        if logits_path.exists():
            out["logits_on_kd_pool"] = np.load(logits_path)

        if class_logits_path.exists():
            with open(class_logits_path, "r", encoding="utf-8") as f:
                out["class_logits"] = logits_from_serializable(json.load(f))
            counts = meta.get("class_counts")
            if counts is not None:
                out["class_counts"] = np.asarray(counts, dtype=np.int64)

        return out
    except Exception as e:
        logger.warning(
            f"Failed to read cache for Client {client_id} Round {round_num}: {e}"
        )
        return None


def _cache_client_result(
    checkpoint_dir: Path,
    client_id: int,
    round_num: int,
    result: Dict[str, Any],
) -> None:
    """Save the client's upload payload and evaluation metadata alongside adapter weights."""
    round_dir = checkpoint_dir / f"client_{client_id}" / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)
    try:
        if result.get("logits_on_kd_pool") is not None:
            np.save(round_dir / "logits_kd.npy", result["logits_on_kd_pool"])

        if result.get("class_logits"):
            with open(round_dir / "class_logits.json", "w", encoding="utf-8") as f:
                json.dump(logits_to_serializable(result["class_logits"]), f, indent=2)

        counts = result.get("class_counts")
        meta_path = round_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "client_id": client_id,
                    "round_num": round_num,
                    "mode": result.get("mode"),
                    "eval_accuracy": float(result["eval_accuracy"]),
                    "eval_source": result.get("eval_source"),
                    "class_counts": (
                        np.asarray(counts, dtype=np.int64).tolist() if counts is not None else None
                    ),
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
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("./results") / f"run_{run_timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(config.checkpoint_dir)
    log_path = Path(config.log_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    # Clean up stale metrics in log_path if starting fresh run (not resuming)
    if not resume:
        for stale_file in ["round_metrics.jsonl", "metrics_history.json"]:
            p = log_path / stale_file
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    print("=" * 80)
    print("       HETEROGENEOUS FEDERATED DISTILLATION PIPELINE")
    print("=" * 80)
    print(f"Run Output Directory     : {results_dir}")
    print(f"Total Clients Configured : {config.num_clients}")
    print(f"Communication Rounds     : {config.num_rounds}")
    print(f"Local Epochs / Client    : {config.local_epochs}")
    print(f"Aggregation Mode         : {config.aggregation_mode} (T={config.aggregation_temperature})")
    print(f"Knowledge Distillation   : Lambda={config.kd_lambda}, T={config.kd_temperature}, Warmup={config.kd_warmup_rounds}")
    print(f"Checkpoints Path         : {config.checkpoint_dir}")
    print(f"Logs Path                : {config.log_dir}")
    print(f"Resume Mode              : {resume}")
    print("=" * 80 + "\n")

    # Initialize Metric Tracker
    metric_tracker = MetricTracker(log_dir=str(log_path))

    # 1. Load Public Datasets (public_set mode only)
    public_kd_pool = None
    public_eval_holdout = None
    if config.is_data_free:
        print("[1/4] Data-free mode: no public transfer pool or public holdout is loaded.")
        print("      Clients exchange per-class averaged logits and are scored on their own")
        print("      held-out slices, so no shared corpus touches the algorithm.\n")
    else:
        print("[1/4] Loading Public Knowledge Distillation and Evaluation Datasets...")
        global_profiler.start("Server_Load_Public_Data")
        public_kd_pool, public_eval_holdout = load_public_dataset(config)
        global_profiler.stop("Server_Load_Public_Data")
        print(
            f"  -> Public KD Pool Loaded        : {len(public_kd_pool)} instances"
        )
        print(
            f"  -> Public Eval Holdout Loaded  : {len(public_eval_holdout)} instances\n"
        )

    # 2. Pre-load Client Private Datasets & Build Cross-Dataset Evaluation Suite
    print("[2/4] Pre-loading Client Private Datasets & Slicing Holdouts...")
    private_train_datasets: Dict[int, Any] = {}
    local_eval_holdouts: Dict[int, Any] = {}
    client_class_counts: Dict[int, Any] = {}
    cross_eval_datasets: Dict[str, Any] = {}
    active_clients: List[int] = []

    # Map across targeted client IDs (active_client_ids if present, else 1..num_clients)
    target_client_ids = (
        config.active_client_ids
        if config.active_client_ids is not None
        else list(range(1, config.num_clients + 1))
    )

    client_manifest_entries: List[str] = []

    for client_id in target_client_ids:
        global_profiler.start("Server_Load_Private_Data")
        raw_ds = load_private_dataset(client_id, config)
        global_profiler.stop("Server_Load_Private_Data")
        if raw_ds is not None and len(raw_ds) > 0:
            m_id = get_model_for_client(client_id)
            m_short = m_id.split("/")[-1]
            ds_info = CLIENT_DATASETS.get(client_id, ("Custom", None))
            ds_name = ds_info[0] + (f" ({ds_info[1]})" if ds_info[1] else "")

            # Carve this client's own held-out slice. In data-free mode this is the *only*
            # evaluation set the client has, so it needs enough examples to be readable --
            # config.local_holdout_size, not the old hard-coded 50 (~8 per class).
            holdout_len = min(
                config.local_holdout_size,
                max(5, int(len(raw_ds) * 0.1)),
            )
            if len(raw_ds) > holdout_len:
                train_slice = raw_ds.select(range(len(raw_ds) - holdout_len))
                eval_slice = raw_ds.select(range(len(raw_ds) - holdout_len, len(raw_ds)))
            else:
                train_slice = raw_ds
                eval_slice = raw_ds

            private_train_datasets[client_id] = train_slice
            local_eval_holdouts[client_id] = eval_slice
            client_class_counts[client_id] = compute_class_counts(train_slice, config.num_classes)
            active_clients.append(client_id)

            col_header = f"Dataset: {ds_name} [Client {client_id:02d}: {m_short}]"
            cross_eval_datasets[col_header] = eval_slice
            client_manifest_entries.append(
                f"Client {client_id:02d} | Model: {m_id:<42} | Dataset: {ds_name:<30} | Train: {len(train_slice):<5} | Eval Holdout: {len(eval_slice)}"
            )
        else:
            print(f"  [SKIPPED] Client {client_id:02d}: Dataset unavailable or empty.")

    if not active_clients:
        print("\n[CRITICAL ERROR] No active clients available with valid datasets. Exiting.")
        sys.exit(1)

    # Also include Public Holdout in cross-dataset evaluation (public_set mode only)
    if not config.is_data_free:
        cross_eval_datasets["Public Holdout [dair-ai/emotion]"] = public_eval_holdout

    # Report class coverage: both modes aggregate per class, so absent or thin classes matter.
    print("\n  Class coverage of the active roster (train slices):")
    header = "    " + f"{'client':<8}" + "".join(f"{c:>10}" for c in CANONICAL_LABELS[:config.num_classes])
    print(header)
    for cid in active_clients:
        counts = client_class_counts[cid]
        print("    " + f"{cid:<8}" + "".join(f"{int(n):>10}" for n in counts))
    totals = np.sum([client_class_counts[c] for c in active_clients], axis=0)
    print("    " + f"{'TOTAL':<8}" + "".join(f"{int(n):>10}" for n in totals))
    holders = np.sum([client_class_counts[c] > 0 for c in active_clients], axis=0)
    print("    " + f"{'holders':<8}" + "".join(f"{int(n):>10}" for n in holders))
    for j, name in enumerate(CANONICAL_LABELS[:config.num_classes]):
        if holders[j] == 0:
            print(f"    [FATAL] class '{name}' is absent from every client; it cannot be learned.")
        elif holders[j] == 1:
            print(f"    [WARN ] class '{name}' is held by only 1 client; no consensus is possible.")
    print()

    # Write run_info.txt
    # The rank actually handed to LoraConfig is sparse_training.reduced_rank whenever sparse
    # training is on -- report that, not config.lora_rank, which the manifest used to print
    # regardless and which silently misreported every sparse run.
    effective_lora_rank = (
        config.sparse_training.reduced_rank
        if config.sparse_training.enabled
        else config.lora_rank
    )

    run_info_file = results_dir / "run_info.txt"
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_vram = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB" if torch.cuda.is_available() else "N/A"
        info_lines = [
            "=" * 85,
            "       HETEROGENEOUS FEDERATED DISTILLATION - EXPERIMENT RUN MANIFEST",
            "=" * 85,
            f"Run Timestamp          : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Run Output Directory   : {results_dir.resolve()}",
            f"Host Platform          : {sys.platform} | Python {sys.version.split()[0]} | PyTorch {torch.__version__}",
            f"Hardware Accelerator   : {gpu_name} (Total VRAM: {gpu_vram})",
            "",
            "-" * 85,
            "GLOBAL CONFIGURATION & HYPERPARAMETERS",
            "-" * 85,
            f"Federation Mode        : {config.mode}",
            f"Active Clients         : {len(active_clients)} of {config.num_clients} configured",
            f"Communication Rounds   : {config.num_rounds}",
            f"Local Epochs / Client  : {config.local_epochs}",
            f"Batch Size (Train/Inf) : {config.batch_size_train} / {config.batch_size_infer}",
            f"Max Sequence Length    : {config.max_seq_length}",
            (
                f"Quantization           : {config.quant_bits}-bit NF4"
                if config.quant_bits in (4, 8)
                else f"Quantization           : DISABLED (quant_bits={config.quant_bits}; "
                     f"NF4 requires 4 or 8)"
            ),
            f"LoRA Hyperparameters   : Rank={effective_lora_rank}, Alpha={config.lora_alpha}, Dropout={config.lora_dropout}",
            f"Sparse Training        : {config.sparse_training.enabled}"
            + (f" (reduced_rank={config.sparse_training.reduced_rank})" if config.sparse_training.enabled else ""),
            f"Learning Rate          : {config.learning_rate}",
        ] + (
            [
                f"Distillation (FD)      : Lambda={config.fd_lambda}, Temperature={config.fd_temperature}, "
                f"Warmup={config.kd_warmup_rounds} rounds",
                f"Aggregation            : per-class logits, "
                f"{'count-weighted' if config.fd_weight_by_count else 'unweighted (reference FD)'}",
                f"Public Transfer Set    : NONE (data-free)",
                f"Client Eval Source     : own local holdout ({config.local_holdout_size} max per client)",
            ]
            if config.is_data_free
            else [
                f"Knowledge Distillation : Lambda={config.kd_lambda}, Temperature={config.kd_temperature}, Warmup={config.kd_warmup_rounds} rounds",
                f"Consensus Aggregation  : Mode={config.aggregation_mode}, Temperature={config.aggregation_temperature}",
                f"Public KD Pool Size    : {len(public_kd_pool)}",
                f"Public Holdout Size    : {len(public_eval_holdout)}",
            ]
        ) + [
            "",
            "-" * 85,
            "PARTICIPATING CLIENTS & BACKBONE ARCHITECTURES",
            "-" * 85,
        ] + client_manifest_entries + [
            "=" * 85,
            "",
        ]
        with open(run_info_file, "w", encoding="utf-8") as f_info:
            f_info.write("\n".join(info_lines))
        print(f"  -> Created experiment run manifest at: {run_info_file}")
    except Exception as e:
        logger.warning(f"Could not write run_info.txt: {e}")

    print(
        f"\nActive Clients for this Run ({len(active_clients)}/{len(target_client_ids)}): "
        f"{active_clients}\n"
    )

    # 3. Pre-load & Verify all Client Models Upfront
    print("[3/4] Pre-loading & Verifying Backbone Models for All Active Clients...")
    if len(active_clients) > 0:
        global_profiler.start("Server_Model_Preloading")
        preload_client_models(active_clients, config)
        global_profiler.stop("Server_Model_Preloading")

    # 4. Communication Rounds Loop
    print("[4/4] Commencing Federated Communication Rounds...\n")
    avg_soft_labels: Optional[np.ndarray] = None          # public_set consensus
    global_class_logits: Optional[Dict[int, np.ndarray]] = None   # data_free_fd consensus

    for round_num in range(1, config.num_rounds + 1):
        print(f"\n>>>>>>>> STARTING COMMUNICATION ROUND {round_num}/{config.num_rounds} <<<<<<<<")

        # If resuming, reload the previous round's consensus artifact
        if resume and round_num > 1:
            if config.is_data_free and global_class_logits is None:
                prev_fd_file = log_path / f"round_{round_num - 1}_global_class_logits.json"
                if prev_fd_file.exists():
                    try:
                        with open(prev_fd_file, "r", encoding="utf-8") as f_fd:
                            global_class_logits = logits_from_serializable(json.load(f_fd))
                        print(f"  -> Resumed previous global class logits from {prev_fd_file}")
                    except Exception as e:
                        logger.warning(f"Could not load previous global class logits: {e}")
            elif not config.is_data_free and avg_soft_labels is None:
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
                # Cross-dataset testing: Only evaluate across all datasets on the final round
                current_cross_eval = cross_eval_datasets if round_num == config.num_rounds else None

                result = run_client_round(
                    client_id=client_id,
                    round_num=round_num,
                    config=config,
                    avg_soft_labels_from_server=avg_soft_labels,
                    public_kd_pool=public_kd_pool,
                    public_eval_holdout=public_eval_holdout,
                    private_dataset=private_train_datasets[client_id],
                    cross_eval_datasets=current_cross_eval,
                    local_eval_holdout=local_eval_holdouts[client_id],
                    global_class_logits=global_class_logits,
                )

                if result is not None:
                    _cache_client_result(checkpoint_path, client_id, round_num, result)

            if result is not None:
                client_results.append(result)
            else:
                print(f"[WARNING] Skipping Client {client_id:02d} for Round {round_num} aggregation.")

        # Server Aggregation of Logits & Cross-Dataset Matrix Computation
        if client_results:
            global_profiler.start("Server_Aggregation")
            # Print & Export Model x Dataset Cross-Evaluation Performance Matrix (final round only)
            if round_num == config.num_rounds:
                print_and_export_cross_dataset_matrix(round_num, client_results, results_dir, config)
                print_and_export_cross_dataset_matrix(round_num, client_results, log_path, config)

            if config.is_data_free:
                # Average each class's logit vector across the clients that hold that class.
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
                )

                if global_class_logits is not None:
                    payload = logits_to_serializable(global_class_logits)
                    for target_dir in [log_path, results_dir]:
                        fd_path = target_dir / f"round_{round_num}_global_class_logits.json"
                        with open(fd_path, "w", encoding="utf-8") as f_fd:
                            json.dump(payload, f_fd, indent=2)
                    n_bytes = len(payload) * config.num_classes * 4
                    print(
                        f"  -> Persisted global per-class logits "
                        f"({len(payload)} classes, {n_bytes} bytes) to: "
                        f"{log_path / f'round_{round_num}_global_class_logits.json'}"
                    )
            else:
                avg_soft_labels = aggregate(client_results, config)

                # Persist consensus teacher soft labels
                if avg_soft_labels is not None:
                    soft_labels_path = log_path / f"round_{round_num}_avg_soft_labels.npy"
                    np.save(soft_labels_path, avg_soft_labels)
                    np.save(results_dir / f"round_{round_num}_avg_soft_labels.npy", avg_soft_labels)
                    print(f"  -> Persisted consensus teacher soft labels to: {soft_labels_path}")

            global_profiler.stop("Server_Aggregation")

            # Record round metrics
            round_accuracies = {
                int(r["client_id"]): float(r["eval_accuracy"]) for r in client_results
            }

            # Append to round_metrics.jsonl
            for target_dir in [log_path, results_dir]:
                jsonl_path = target_dir / "round_metrics.jsonl"
                with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                    for r in client_results:
                        record = {
                            "round": round_num,
                            "client_id": int(r["client_id"]),
                            "model_id": r.get("model_id", ""),
                            "dataset_name": r.get("dataset_name", ""),
                            "eval_accuracy": float(r["eval_accuracy"]),
                            "eval_correct": int(r.get("eval_correct", 0)),
                            "eval_total": int(r.get("eval_total", 0)),
                            "avg_ce_loss": float(r.get("avg_ce_loss", 0.0)),
                            "avg_kd_loss": float(r.get("avg_kd_loss", 0.0)),
                            "avg_total_loss": float(r.get("avg_total_loss", 0.0)),
                            "kd_active": bool(r.get("kd_active", False)),
                            "num_private_examples": int(r.get("num_private_examples", 0)),
                            "num_train_steps": int(r.get("num_train_steps", 0)),
                            "local_epochs": int(r.get("local_epochs", 0)),
                            "num_kd_pool": int(r.get("num_kd_pool", 0)),
                            "num_eval_holdout": int(r.get("num_eval_holdout", 0)),
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
    summarize_run(config, results_dir=results_dir)

    # Save execution time profiler summary
    global_profiler.save_summary(results_dir / "timing_summary.json")
    print(f"\n[COMPLETE] All run artifacts and manifests saved in: {results_dir.resolve()}\n")


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
