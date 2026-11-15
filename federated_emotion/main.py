from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from federated_emotion.client import client_label, run_client_round
from federated_emotion.config import Config, load_config
from federated_emotion.data.loaders import AG_NEWS_LABELS, CANONICAL_LABELS, compute_class_counts, load_emotion_partitions
from federated_emotion.datafree import (
    aggregate_class_payloads,
    payload_from_serializable,
    payload_to_serializable,
)
from federated_emotion.eval_utils import MetricTracker, print_and_export_cross_dataset_matrix, summarize_run
from federated_emotion.models.bert import build_client_models
from federated_emotion.profiler import global_profiler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("federated_emotion.main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pruned-BERT data-free federated distillation (FCFD)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_rounds", type=int, default=None)
    parser.add_argument("--local_epochs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _round_dir(checkpoint_dir: Path, client_id: int, round_num: int) -> Path:
    return checkpoint_dir / f"client_{client_id}" / f"round_{round_num}"


def _cache_client_result(checkpoint_dir: Path, client_id: int, round_num: int, result: Dict[str, Any]) -> None:
    d = _round_dir(checkpoint_dir, client_id, round_num)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "class_payload.json", "w", encoding="utf-8") as f:
        json.dump(payload_to_serializable(result["class_payload"]), f)
    meta = {k: result[k] for k in ("client_id", "mode", "eval_accuracy", "test_accuracy", "model_id", "dataset_name")}
    meta["round_num"] = round_num
    meta["class_counts"] = np.asarray(result["class_counts"]).tolist()
    meta["reliability"] = np.asarray(result["reliability"]).tolist()
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _load_cached_client_result(checkpoint_dir: Path, client_id: int, round_num: int) -> Optional[Dict[str, Any]]:
    d = _round_dir(checkpoint_dir, client_id, round_num)
    if not ((d / "meta.json").exists() and (d / "class_payload.json").exists() and (d / "model.pt").exists()):
        return None
    try:
        with open(d / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(d / "class_payload.json", "r", encoding="utf-8") as f:
            payload = payload_from_serializable(json.load(f))
        return {
            **meta,
            "class_payload": payload,
            "class_logits": payload["logits"] if payload else None,
            "class_counts": np.asarray(meta["class_counts"], dtype=np.int64),
            "reliability": np.asarray(meta["reliability"], dtype=np.float64),
            "class_losses": None,
            "cross_eval_accuracies": {},
        }
    except Exception as e:
        logger.warning(f"Failed to read cache for client {client_id} round {round_num}: {e}")
        return None


def _write_manifest(path: Path, config: Config, client_lines: List[str]) -> None:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    lines = [
        "=" * 85,
        "       PRUNED-BERT DATA-FREE FEDERATED DISTILLATION (FCFD) - RUN MANIFEST",
        "=" * 85,
        f"Run Timestamp        : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Platform             : {sys.platform} | Python {sys.version.split()[0]} | PyTorch {torch.__version__} | {gpu}",
        f"Mode                 : {config.mode}",
        f"Global model         : {config.bert_model_name}",
        f"Client capacities    : {config.client_capacities}",
        f"Dataset              : dair-ai/emotion, Dirichlet alpha={config.dirichlet_alpha}",
        f"Rounds / epochs      : {config.num_rounds} / {config.local_epochs}",
        f"Batch / seq / lr     : {config.batch_size_train} / {config.max_seq_length} / {config.learning_rate}",
        f"FCFD                 : positions={config.fcfd_positions} paths={config.fcfd_num_paths} "
        f"kd={config.fcfd_kd_weight} app={config.fcfd_app_weight} func={config.fcfd_func_weight} T={config.fd_temperature}",
        f"Warmup rounds        : {config.kd_warmup_rounds}",
        f"Aggregation          : count^{config.fd_count_exponent if config.fd_weight_by_count else 0}"
        + (f" x reliability^{config.fd_reliability_exponent}" if config.fd_use_reliability else ""),
        "-" * 85,
    ] + client_lines + ["=" * 85, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(config: Config, resume: bool = False) -> None:
    if not config.is_data_free:
        raise ValueError("The pruned-BERT pipeline only supports mode='data_free_fd'.")

    results_dir = Path("./results") / f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config.checkpoint_dir)
    log_path = Path(config.log_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    if not resume:
        for stale in ["round_metrics.jsonl", "metrics_history.json"]:
            (log_path / stale).unlink(missing_ok=True)

    client_ids = config.active_client_ids or list(range(1, config.num_clients + 1))
    if len(config.client_capacities) != len(client_ids):
        raise ValueError(
            f"client_capacities has {len(config.client_capacities)} entries but there are {len(client_ids)} clients."
        )

    print("=" * 80)
    print("       PRUNED-BERT DATA-FREE FEDERATED DISTILLATION (FCFD)")
    print("=" * 80)
    print(f"Output: {results_dir} | clients: {client_ids} | capacities: {config.client_capacities}\n")

    metric_tracker = MetricTracker(log_dir=str(log_path))

    d_name = getattr(config, "dataset_name", "dair-ai/emotion")
    print(f"[1/3] Partitioning {d_name} across clients (Dirichlet non-IID)...")
    global_profiler.start("Server_Load_Data")
    partitions, global_test = load_emotion_partitions(config, client_ids)
    global_profiler.stop("Server_Load_Data")
    dataset_name = f"{d_name} (dir a={config.dirichlet_alpha})"

    train_sets: Dict[int, Any] = {}
    holdouts: Dict[int, Any] = {}
    counts: Dict[int, np.ndarray] = {}
    client_lines: List[str] = []
    cross_eval: Dict[str, Any] = {}
    for cid in client_ids:
        ds = partitions[cid]
        n_hold = min(config.local_holdout_size, max(5, int(len(ds) * 0.1)))
        train_sets[cid] = ds.select(range(len(ds) - n_hold))
        holdouts[cid] = ds.select(range(len(ds) - n_hold, len(ds)))
        counts[cid] = compute_class_counts(train_sets[cid], config.num_classes)
        label = client_label(cid, config)
        cross_eval[f"Holdout [Client {cid:02d}: {label}]"] = holdouts[cid]
        client_lines.append(f"Client {cid:02d} | {label:<26} | train {len(train_sets[cid]):<5} | holdout {n_hold}")
    cross_eval[f"Global test [{d_name}]"] = global_test

    class_names = AG_NEWS_LABELS if "ag_news" in d_name.lower() or "agnews" in d_name.lower() else CANONICAL_LABELS
    print("\n  Class coverage (train slices):")
    print("    " + f"{'client':<8}" + "".join(f"{c:>10}" for c in class_names[: config.num_classes]))
    for cid in client_ids:
        print("    " + f"{cid:<8}" + "".join(f"{int(n):>10}" for n in counts[cid]))
    print(f"  Global test set: {len(global_test)} examples\n")
    _write_manifest(results_dir / "run_info.txt", config, client_lines)

    print("[2/3] Server: pruning the global BERT into client models...")
    global_profiler.start("Server_Pruning")
    if resume and all((_round_dir(checkpoint_path, c, 0) / "model.pt").exists() for c in client_ids):
        print("  -> Reusing existing round_0 pruned models.")
    else:
        build_client_models(
            config.bert_model_name,
            config.num_classes,
            config.client_capacities,
            client_ids,
            checkpoint_path,
        )
    global_profiler.stop("Server_Pruning")

    print("\n[3/3] Federated rounds...")
    global_payload: Optional[Dict[str, object]] = None
    for round_num in range(1, config.num_rounds + 1):
        print(f"\n>>>>>>>> ROUND {round_num}/{config.num_rounds} <<<<<<<<")
        if resume and round_num > 1 and global_payload is None:
            prev = log_path / f"round_{round_num - 1}_global_payload.json"
            if prev.exists():
                with open(prev, "r", encoding="utf-8") as f:
                    global_payload = payload_from_serializable(json.load(f))
                print(f"  -> Resumed global payload from {prev}")

        results: List[Dict[str, Any]] = []
        for cid in client_ids:
            result = _load_cached_client_result(checkpoint_path, cid, round_num) if resume else None
            if result is not None:
                print(f"[Client {cid:02d}] cached round {round_num} result (acc {result['eval_accuracy'] * 100:.2f}%)")
            else:
                result = run_client_round(
                    client_id=cid,
                    round_num=round_num,
                    config=config,
                    private_dataset=train_sets[cid],
                    local_eval_holdout=holdouts[cid],
                    global_test=global_test,
                    global_payload=global_payload,
                    cross_eval_datasets=cross_eval if round_num == config.num_rounds else None,
                    dataset_name=dataset_name,
                )
                if result is not None:
                    _cache_client_result(checkpoint_path, cid, round_num, result)
            if result is not None:
                results.append(result)
            else:
                print(f"[WARNING] Client {cid:02d} skipped for round {round_num}.")

        if not results:
            print(f"[CRITICAL] Round {round_num}: no client succeeded.")
            continue

        global_profiler.start("Server_Aggregation")
        if round_num == config.num_rounds:
            print_and_export_cross_dataset_matrix(round_num, results, results_dir, config)

        global_payload = aggregate_class_payloads(
            [r.get("class_payload") for r in results],
            config.fcfd_positions,
            class_counts=[r["class_counts"] for r in results] if config.fd_weight_by_count else None,
            reliability=[r.get("reliability") for r in results] if config.fd_use_reliability else None,
            class_losses=None,
            count_exponent=config.fd_count_exponent,
            reliability_exponent=config.fd_reliability_exponent,
            num_classes=config.num_classes,
            client_ids=[int(r["client_id"]) for r in results],
        )
        if global_payload is not None:
            serial = payload_to_serializable(global_payload)
            for d in (log_path, results_dir):
                with open(d / f"round_{round_num}_global_payload.json", "w", encoding="utf-8") as f:
                    json.dump(serial, f)
            n_feat = sum(len(v) for v in global_payload["features"].values())
            print(
                f"  -> Global payload: {len(global_payload['logits'])} class logit vectors, "
                f"{n_feat} class feature prototypes over positions {sorted(global_payload['features'])}"
            )
        global_profiler.stop("Server_Aggregation")

        accs = {int(r["client_id"]): float(r["eval_accuracy"]) for r in results}
        tests = {int(r["client_id"]): r.get("test_accuracy") for r in results}
        for d in (log_path, results_dir):
            with open(d / "round_metrics.jsonl", "a", encoding="utf-8") as f:
                for r in results:
                    record = {
                        "round": round_num,
                        "client_id": int(r["client_id"]),
                        "model_id": r.get("model_id", ""),
                        "dataset_name": r.get("dataset_name", ""),
                        "capacity": r.get("capacity"),
                        "eval_accuracy": float(r["eval_accuracy"]),
                        "test_accuracy": r.get("test_accuracy"),
                        "test_macro_f1": r.get("test_macro_f1"),
                        "eval_correct": int(r.get("eval_correct", 0)),
                        "eval_total": int(r.get("eval_total", 0)),
                        "avg_ce_loss": float(r.get("avg_ce_loss", 0.0)),
                        "avg_kd_loss": float(r.get("avg_kd_loss", 0.0)),
                        "avg_total_loss": float(r.get("avg_total_loss", 0.0)),
                        "kd_active": bool(r.get("kd_active", False)),
                        "num_private_examples": int(r.get("num_private_examples", 0)),
                        "num_train_steps": int(r.get("num_train_steps", 0)),
                        "local_epochs": int(r.get("local_epochs", 0)),
                        "num_kd_pool": 0,
                        "num_eval_holdout": int(r.get("num_eval_holdout", 0)),
                    }
                    f.write(json.dumps(record) + "\n")
        valid_tests = [v for v in tests.values() if v is not None]
        metric_tracker.log_round(
            round_num=round_num,
            metrics={
                "active_clients_count": len(results),
                "client_accuracies": accs,
                "client_test_accuracies": tests,
                "mean_accuracy": float(np.mean(list(accs.values()))),
                "mean_test_accuracy": float(np.mean(valid_tests)) if valid_tests else None,
            },
        )
        print(
            "  -> Round summary (global test acc): "
            + ", ".join(f"c{c}={v * 100:.1f}%" for c, v in tests.items() if v is not None)
        )

    print("\n" + "=" * 80)
    print("       FEDERATED TRAINING COMPLETE")
    print("=" * 80)
    metric_tracker.save_summary()
    summarize_run(config, results_dir=results_dir)
    global_profiler.save_summary(results_dir / "timing_summary.json")
    print(f"\n[COMPLETE] Artifacts saved in: {results_dir.resolve()}\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.num_rounds is not None:
        config.num_rounds = args.num_rounds
    if args.local_epochs is not None:
        config.local_epochs = args.local_epochs
    run_pipeline(config=config, resume=args.resume)


if __name__ == "__main__":
    main()
