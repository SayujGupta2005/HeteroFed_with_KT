"""Evaluation, metrics computation, and performance logging utilities for federated distillation.

This module provides:
1. compute_classification_metrics: Multi-class accuracy, Macro-F1, Weighted-F1, Precision, and Recall.
2. compute_distillation_divergence: KL divergence between student logits and teacher soft distribution.
3. MetricTracker: Logs per-round metrics and outputs metrics_history.json.
4. summarize_run: Reads log_dir/round_metrics.jsonl, prints a tabular performance matrix of
   client accuracy across communication rounds with group mean trends, and exports log_dir/summary.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from federated_emotion.config import Config
from federated_emotion.data.loaders import CLIENT_DATASETS
from federated_emotion.models.wrapper import CLIENT_MODELS, get_model_for_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Classification Metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute standard multi-class emotion classification metrics.

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels or probability distributions (takes argmax if 2D).
        labels: Optional canonical class label names.

    Returns:
        Dictionary of computed metric scores.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_pred_arr.ndim > 1:
        y_pred_arr = np.argmax(y_pred_arr, axis=-1)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
    }


# ---------------------------------------------------------------------------
# 2. Knowledge Distillation Divergence
# ---------------------------------------------------------------------------
def compute_distillation_divergence(
    student_logits: Union[np.ndarray, Any],
    teacher_probs: Union[np.ndarray, Any],
    temperature: float = 2.0,
) -> float:
    """Compute mean Kullback-Leibler divergence between student logits and teacher probabilities.

    Args:
        student_logits: Logits from student client, shape [N, num_classes].
        teacher_probs: Soft probability distribution from teacher/consensus, shape [N, num_classes].
        temperature: Softmax scaling temperature.

    Returns:
        Mean KL divergence scalar value.
    """
    logits_arr = np.asarray(student_logits, dtype=np.float64)
    teacher_arr = np.asarray(teacher_probs, dtype=np.float64)

    # Student scaled log probabilities
    scaled_student = logits_arr / max(temperature, 1e-8)
    max_s = np.max(scaled_student, axis=-1, keepdims=True)
    exp_s = np.exp(scaled_student - max_s)
    student_probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    log_student_probs = np.log(np.clip(student_probs, a_min=1e-12, a_max=1.0))

    # Target teacher distribution
    target_probs = np.clip(teacher_arr, a_min=1e-12, a_max=1.0)
    target_probs = target_probs / np.sum(target_probs, axis=-1, keepdims=True)

    # KL(target || student) = sum(target * (log(target) - log(student)))
    kl_per_sample = np.sum(target_probs * (np.log(target_probs) - log_student_probs), axis=-1)
    return float(np.mean(kl_per_sample) * (temperature ** 2))


# ---------------------------------------------------------------------------
# 3. Metric Tracker
# ---------------------------------------------------------------------------
class MetricTracker:
    """Utility class to track, summarize, and persist round-by-round federated metrics."""

    def __init__(self, log_dir: str, experiment_name: Optional[str] = None) -> None:
        """Initialize metric tracker with output log directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.history: List[Dict[str, Any]] = []

    def log_round(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """Record metrics for a completed communication round."""
        entry = {
            "round": round_num,
            **metrics,
        }
        self.history.append(entry)

        # Write progressive JSON file
        json_path = self.log_dir / "metrics_history.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def save_summary(self) -> str:
        """Export accumulated history to JSON file."""
        json_path = self.log_dir / "metrics_history.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return str(json_path)


# ---------------------------------------------------------------------------
# 4. Final Run Summarizer & Progress Matrix
# ---------------------------------------------------------------------------
def summarize_run(
    config: Config,
    results_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Read logged per-round per-client metrics, print tabular matrices, and export CSVs.

    Outputs:
    1. **summary.csv**: Accuracy comparison matrix (rows = clients, columns = rounds)
       with net gain and group mean row.
    2. **detailed_metrics.csv**: Comprehensive per-client per-round CSV with all metrics:
       model backbone, dataset, CE/KD/total loss, KD active flag, training stats,
       eval accuracy, eval correct/total, private dataset size.
    3. Console: Formatted tables for both outputs.

    Args:
        config: Global Config dataclass instance.
        results_dir: Optional run results subfolder (e.g. results/run_YYYYMMDD_HHMMSS).
    """
    log_dir = Path(config.log_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    target_out_dirs = [log_dir]
    if results_dir is not None:
        target_out_dirs.append(Path(results_dir))

    # -----------------------------------------------------------------------
    # A. Read ALL records from round_metrics.jsonl
    # -----------------------------------------------------------------------
    all_records: List[Dict[str, Any]] = []
    
    # Prefer reading from the run-specific results_dir if available
    jsonl_path = None
    if results_dir is not None and (Path(results_dir) / "round_metrics.jsonl").exists():
        jsonl_path = Path(results_dir) / "round_metrics.jsonl"
    elif (log_dir / "round_metrics.jsonl").exists():
        jsonl_path = log_dir / "round_metrics.jsonl"

    if jsonl_path is not None and jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line.strip())
                        r_val = int(rec.get("round", 0))
                        if 1 <= r_val <= config.num_rounds:
                            all_records.append(rec)
        except Exception as e:
            logger.warning(f"Error reading {jsonl_path}: {e}")

    # Build quick lookup: client_id -> {round_num: record}
    client_round_data: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for rec in all_records:
        c_id = int(rec.get("client_id", 0))
        r_num = int(rec.get("round", 0))
        if c_id not in client_round_data:
            client_round_data[c_id] = {}
        client_round_data[c_id][r_num] = rec

    # Fallback to metrics_history.json if JSONL was empty
    if not client_round_data:
        history_path = None
        if results_dir is not None and (Path(results_dir) / "metrics_history.json").exists():
            history_path = Path(results_dir) / "metrics_history.json"
        elif (log_dir / "metrics_history.json").exists():
            history_path = log_dir / "metrics_history.json"

        if history_path is not None and history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history_data = json.load(f)
                for h in history_data:
                    r_num = int(h["round"])
                    if not (1 <= r_num <= config.num_rounds):
                        continue
                    c_accs = h.get("client_accuracies", {})
                    for c_id_str, acc_val in c_accs.items():
                        c_id = int(c_id_str)
                        if c_id not in client_round_data:
                            client_round_data[c_id] = {}
                        client_round_data[c_id][r_num] = {
                            "client_id": c_id,
                            "round": r_num,
                            "eval_accuracy": float(acc_val),
                        }
            except Exception as e:
                logger.warning(f"Error reading {history_path}: {e}")

    # Fallback to scanning checkpoint directories
    if not client_round_data and checkpoint_dir.exists():
        for c_id in range(1, config.num_clients + 1):
            for r_num in range(1, config.num_rounds + 1):
                meta_path = checkpoint_dir / f"client_{c_id}" / f"round_{r_num}" / "meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if c_id not in client_round_data:
                            client_round_data[c_id] = {}
                        client_round_data[c_id][r_num] = {
                            "client_id": c_id,
                            "round": r_num,
                            "eval_accuracy": float(meta["eval_accuracy"]),
                        }
                    except Exception:
                        pass

    sorted_clients = sorted(list(client_round_data.keys()))
    if not sorted_clients:
        print("[SUMMARY] No client evaluation accuracy metrics found to summarize.")
        return

    all_rounds = sorted(list(set(
        r for c_dict in client_round_data.values() for r in c_dict.keys()
        if 1 <= r <= config.num_rounds
    )))
    if not all_rounds:
        all_rounds = list(range(1, config.num_rounds + 1))

    # -----------------------------------------------------------------------
    # B. OUTPUT 1: Accuracy Comparison Matrix (summary.csv)
    # -----------------------------------------------------------------------
    header_cols = ["Client", "Model Backbone", "Dataset"] + [f"Round {r}" for r in all_rounds] + ["Net Gain"]
    col_widths = [10, 28, 24] + [10 for _ in all_rounds] + [10]

    def format_row(cols: List[str]) -> str:
        parts = []
        for text, width in zip(cols, col_widths):
            parts.append(f"{text:<{width}}" if parts else f"{text:<{width}}")
        return " | ".join(parts)

    table_header = format_row(header_cols)
    divider = "-" * len(table_header)

    lines: List[str] = []
    lines.append("\n" + "=" * len(table_header))
    lines.append("        FEDERATED EMOTION CLASSIFICATION - ROUND PERFORMANCE SUMMARY")
    lines.append("=" * len(table_header))
    lines.append(table_header)
    lines.append(divider)

    round_client_accs: Dict[int, List[float]] = {r: [] for r in all_rounds}
    summary_csv_rows: List[Dict[str, Any]] = []

    for cid in sorted_clients:
        model_id = get_model_for_client(cid)
        model_short = model_id.split("/")[-1]
        ds_info = CLIENT_DATASETS.get(cid, ("Unknown", None))
        ds_name = f"{ds_info[0]}" + (f" ({ds_info[1]})" if ds_info[1] else "")
        ds_name_trunc = (ds_name[:21] + "..") if len(ds_name) > 24 else ds_name

        acc_strs = []
        acc_floats: List[Optional[float]] = []
        csv_row_entry: Dict[str, Any] = {
            "client_id": cid,
            "model_id": model_id,
            "dataset": ds_name,
        }

        for r in all_rounds:
            rec = client_round_data.get(cid, {}).get(r)
            if rec is not None:
                acc = float(rec.get("eval_accuracy", 0.0))
                acc_strs.append(f"{acc * 100:.2f}%")
                acc_floats.append(acc)
                round_client_accs[r].append(acc)
                csv_row_entry[f"round_{r}_accuracy"] = round(acc, 4)
            else:
                acc_strs.append("N/A")
                acc_floats.append(None)
                csv_row_entry[f"round_{r}_accuracy"] = ""

        valid_floats = [a for a in acc_floats if a is not None]
        if len(valid_floats) >= 2:
            net_gain = (valid_floats[-1] - valid_floats[0]) * 100
            gain_str = f"{net_gain:>+6.2f}%"
            csv_row_entry["net_gain_pct"] = round(net_gain, 4)
        else:
            gain_str = "---"
            csv_row_entry["net_gain_pct"] = ""

        row_cells = [f"Client {cid:02d}", model_short, ds_name_trunc] + acc_strs + [gain_str]
        lines.append(format_row(row_cells))
        summary_csv_rows.append(csv_row_entry)

    lines.append(divider)

    # Mean row
    mean_strs = []
    mean_floats = []
    csv_mean_entry: Dict[str, Any] = {
        "client_id": "MEAN",
        "model_id": "ALL_MODELS",
        "dataset": "ALL_DATASETS",
    }

    for r in all_rounds:
        vals = round_client_accs[r]
        if vals:
            mean_acc = float(np.mean(vals))
            mean_strs.append(f"{mean_acc * 100:.2f}%")
            mean_floats.append(mean_acc)
            csv_mean_entry[f"round_{r}_accuracy"] = round(mean_acc, 4)
        else:
            mean_strs.append("N/A")
            csv_mean_entry[f"round_{r}_accuracy"] = ""

    if len(mean_floats) >= 2:
        group_gain = (mean_floats[-1] - mean_floats[0]) * 100
        group_gain_str = f"{group_gain:>+6.2f}%"
        csv_mean_entry["net_gain_pct"] = round(group_gain, 4)
    else:
        group_gain_str = "---"
        csv_mean_entry["net_gain_pct"] = ""

    mean_cells = ["MEAN", "GROUP AVERAGE", "--"] + mean_strs + [group_gain_str]
    lines.append(format_row(mean_cells))
    lines.append("=" * len(table_header) + "\n")
    summary_csv_rows.append(csv_mean_entry)

    # Print summary table
    table_text = "\n".join(lines)
    print(table_text)

    # Write summary.csv to all output destinations
    summary_fieldnames = (
        ["client_id", "model_id", "dataset"]
        + [f"round_{r}_accuracy" for r in all_rounds]
        + ["net_gain_pct"]
    )
    for out_dir in target_out_dirs:
        summary_csv_path = out_dir / "summary.csv"
        try:
            with open(summary_csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=summary_fieldnames)
                writer.writeheader()
                writer.writerows(summary_csv_rows)
            print(f"  -> Exported accuracy summary to: {summary_csv_path}")
        except Exception as e:
            logger.warning(f"Could not export {summary_csv_path}: {e}")

    # -----------------------------------------------------------------------
    # C. OUTPUT 2: Comprehensive Detailed Metrics CSV (detailed_metrics.csv)
    # -----------------------------------------------------------------------
    detailed_fieldnames = [
        "round",
        "client_id",
        "model_id",
        "dataset_name",
        "num_private_examples",
        "local_epochs",
        "num_train_steps",
        "avg_ce_loss",
        "avg_kd_loss",
        "avg_total_loss",
        "kd_active",
        "eval_accuracy",
        "eval_accuracy_pct",
        "eval_correct",
        "eval_total",
        "num_kd_pool",
        "num_eval_holdout",
    ]

    detailed_rows: List[Dict[str, Any]] = []
    for r in all_rounds:
        round_accs = []
        for cid in sorted_clients:
            rec = client_round_data.get(cid, {}).get(r)
            if rec is None:
                continue
            acc = float(rec.get("eval_accuracy", 0.0))
            round_accs.append(acc)
            detailed_rows.append({
                "round": r,
                "client_id": cid,
                "model_id": rec.get("model_id", get_model_for_client(cid)),
                "dataset_name": rec.get("dataset_name", ""),
                "num_private_examples": rec.get("num_private_examples", ""),
                "local_epochs": rec.get("local_epochs", ""),
                "num_train_steps": rec.get("num_train_steps", ""),
                "avg_ce_loss": round(float(rec.get("avg_ce_loss", 0)), 6) if rec.get("avg_ce_loss") else "",
                "avg_kd_loss": round(float(rec.get("avg_kd_loss", 0)), 6) if rec.get("avg_kd_loss") else "",
                "avg_total_loss": round(float(rec.get("avg_total_loss", 0)), 6) if rec.get("avg_total_loss") else "",
                "kd_active": rec.get("kd_active", ""),
                "eval_accuracy": round(acc, 6),
                "eval_accuracy_pct": f"{acc * 100:.2f}%",
                "eval_correct": rec.get("eval_correct", ""),
                "eval_total": rec.get("eval_total", ""),
                "num_kd_pool": rec.get("num_kd_pool", ""),
                "num_eval_holdout": rec.get("num_eval_holdout", ""),
            })

        # Add a MEAN row per round
        if round_accs:
            mean_acc = float(np.mean(round_accs))
            detailed_rows.append({
                "round": r,
                "client_id": "MEAN",
                "model_id": "ALL",
                "dataset_name": "ALL",
                "num_private_examples": "",
                "local_epochs": "",
                "num_train_steps": "",
                "avg_ce_loss": "",
                "avg_kd_loss": "",
                "avg_total_loss": "",
                "kd_active": "",
                "eval_accuracy": round(mean_acc, 6),
                "eval_accuracy_pct": f"{mean_acc * 100:.2f}%",
                "eval_correct": "",
                "eval_total": "",
                "num_kd_pool": "",
                "num_eval_holdout": "",
            })

    for out_dir in target_out_dirs:
        detailed_csv_path = out_dir / "detailed_metrics.csv"
        try:
            with open(detailed_csv_path, "w", newline="", encoding="utf-8") as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=detailed_fieldnames)
                writer.writeheader()
                writer.writerows(detailed_rows)
            print(f"  -> Exported detailed metrics to: {detailed_csv_path}")
        except Exception as e:
            logger.warning(f"Could not export {detailed_csv_path}: {e}")

    # -----------------------------------------------------------------------
    # D. Print Detailed Metrics Table to Console
    # -----------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("        DETAILED PER-ROUND PER-CLIENT METRICS")
    print("=" * 120)

    detail_header = (
        f"{'Round':<6} | {'Client':<10} | {'Model':<28} | {'Dataset':<22} | "
        f"{'CE Loss':<10} | {'KD Loss':<10} | {'Total Loss':<11} | "
        f"{'KD Active':<10} | {'Accuracy':<10} | {'Correct':<10}"
    )
    print(detail_header)
    print("-" * len(detail_header))

    for row in detailed_rows:
        cid_str = str(row["client_id"])
        model_short = str(row.get("model_id", "")).split("/")[-1][:26]
        ds_short = str(row.get("dataset_name", ""))[:20]
        ce = f"{row['avg_ce_loss']:.4f}" if isinstance(row.get("avg_ce_loss"), (int, float)) and row["avg_ce_loss"] != "" else "---"
        kd = f"{row['avg_kd_loss']:.4f}" if isinstance(row.get("avg_kd_loss"), (int, float)) and row["avg_kd_loss"] != "" else "---"
        tot = f"{row['avg_total_loss']:.4f}" if isinstance(row.get("avg_total_loss"), (int, float)) and row["avg_total_loss"] != "" else "---"
        kd_act = str(row.get("kd_active", "---"))
        acc_str = row.get("eval_accuracy_pct", "---")
        correct_str = f"{row.get('eval_correct', '---')}/{row.get('eval_total', '---')}" if row.get("eval_correct") != "" else "---"

        print(
            f"{row['round']:<6} | {cid_str:<10} | {model_short:<28} | {ds_short:<22} | "
            f"{ce:<10} | {kd:<10} | {tot:<11} | "
            f"{kd_act:<10} | {acc_str:<10} | {correct_str:<10}"
        )

    print("=" * 120 + "\n")


# ---------------------------------------------------------------------------
# 5. Cross-Dataset Performance Matrix (Model × Dataset)
# ---------------------------------------------------------------------------
def print_and_export_cross_dataset_matrix(
    round_num: int,
    client_results: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    config: Optional[Config] = None,
) -> Optional[Path]:
    """Format and print the Model x Dataset cross-evaluation performance matrix and export to CSV.

    Rows (Left Column) : Participating Client Models (e.g. 'Client 01 (openchat-3.5-0106)')
    Columns (Top Header): Datasets with the model they privately belong to in brackets
                          (e.g. 'Dataset: go_emotions [Client 01: openchat-3.5-0106]')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all dataset column names across client results
    dataset_cols: List[str] = []
    for res in client_results:
        cross_dict = res.get("cross_eval_accuracies", {})
        for col_name in cross_dict.keys():
            if col_name not in dataset_cols:
                dataset_cols.append(col_name)

    if not dataset_cols:
        return None

    # Build row dictionaries
    matrix_rows: List[Dict[str, Any]] = []
    for res in client_results:
        cid = res["client_id"]
        model_id = res.get("model_id", get_model_for_client(cid))
        model_short = model_id.split("/")[-1]
        row_label = f"Client {cid:02d} ({model_short})"
        cross_dict = res.get("cross_eval_accuracies", {})

        row_dict: Dict[str, Any] = {
            "client_id": cid,
            "model_backbone": row_label,
        }
        for d_col in dataset_cols:
            acc = cross_dict.get(d_col)
            row_dict[d_col] = round(acc * 100, 2) if acc is not None else ""
        matrix_rows.append(row_dict)

    # Print formatted matrix table
    model_col_w = max(len("Model Backbone (Left)"), max(len(r["model_backbone"]) for r in matrix_rows)) + 2
    col_ws = [max(len(d), 10) + 2 for d in dataset_cols]

    lines = []
    total_w = model_col_w + sum(col_ws) + len(col_ws) * 3
    lines.append("\n" + "=" * total_w)
    lines.append(f"        ROUND {round_num} CROSS-DATASET EVALUATION PERFORMANCE MATRIX (Model x Dataset)")
    lines.append("=" * total_w)

    hdr = f"{'Model Backbone (Left)':<{model_col_w}} | " + " | ".join(f"{d:<{w}}" for d, w in zip(dataset_cols, col_ws))
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for r in matrix_rows:
        vals = []
        for d, w in zip(dataset_cols, col_ws):
            v = r.get(d, "")
            vals.append(f"{v:.2f}%" if isinstance(v, (int, float)) else "---")
        row_str = f"{r['model_backbone']:<{model_col_w}} | " + " | ".join(f"{v:<{w}}" for v, w in zip(vals, col_ws))
        lines.append(row_str)

    lines.append("=" * total_w + "\n")
    print("\n".join(lines))

    # Export to CSV
    csv_file = output_path / f"cross_eval_round_{round_num}.csv"
    try:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["client_id", "model_backbone"] + dataset_cols)
            writer.writeheader()
            writer.writerows(matrix_rows)
        print(f"  -> Exported cross-dataset matrix to: {csv_file}")
    except Exception as e:
        logger.warning(f"Could not export cross-dataset matrix {csv_file}: {e}")

    return csv_file


# ---------------------------------------------------------------------------
# 6. Package Exports
# ---------------------------------------------------------------------------
__all__ = [
    "compute_classification_metrics",
    "compute_distillation_divergence",
    "MetricTracker",
    "print_and_export_cross_dataset_matrix",
    "summarize_run",
]

