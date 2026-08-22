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
from federated_emotion.models.wrapper import CLIENT_MODELS

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
def summarize_run(config: Config) -> None:
    """Read logged per-round per-client accuracy values, print tabular matrix, and export summary.csv.

    Requirements:
    1. Reads all logged per-round, per-client accuracy values from log_dir/round_metrics.jsonl
       (with fallback to metrics_history.json or checkpoint metadata).
    2. Prints a formatted table: rows = clients, columns = rounds, values = eval accuracy on public_eval_holdout.
    3. Prints mean accuracy across active clients per round to illustrate KD group progression.
    4. Saves this summary table as a CSV file in log_dir/summary.csv.
    5. Uses only standard libraries (no external plotting dependencies).

    Args:
        config: Global Config dataclass instance.
    """
    log_dir = Path(config.log_dir)
    checkpoint_dir = Path(config.checkpoint_dir)

    # Dictionary structure: client_id -> {round_num: accuracy}
    client_round_acc: Dict[int, Dict[int, float]] = {}
    total_rounds = config.num_rounds

    # 1. Read from round_metrics.jsonl
    jsonl_path = log_dir / "round_metrics.jsonl"
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line.strip())
                        r_num = int(record["round"])
                        c_id = int(record["client_id"])
                        acc = float(record["eval_accuracy"])
                        if c_id not in client_round_acc:
                            client_round_acc[c_id] = {}
                        client_round_acc[c_id][r_num] = acc
        except Exception as e:
            logger.warning(f"Error reading {jsonl_path}: {e}")

    # Fallback to metrics_history.json if jsonl was not populated
    history_path = log_dir / "metrics_history.json"
    if not client_round_acc and history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            for h in history_data:
                r_num = int(h["round"])
                c_accs = h.get("client_accuracies", {})
                for c_id_str, acc_val in c_accs.items():
                    c_id = int(c_id_str)
                    if c_id not in client_round_acc:
                        client_round_acc[c_id] = {}
                    client_round_acc[c_id][r_num] = float(acc_val)
        except Exception as e:
            logger.warning(f"Error reading {history_path}: {e}")

    # Fallback to scanning checkpoint directories
    if not client_round_acc and checkpoint_dir.exists():
        for c_id in range(1, config.num_clients + 1):
            for r_num in range(1, total_rounds + 1):
                meta_path = checkpoint_dir / f"client_{c_id}" / f"round_{r_num}" / "meta.json"
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if c_id not in client_round_acc:
                            client_round_acc[c_id] = {}
                        client_round_acc[c_id][r_num] = float(meta["eval_accuracy"])
                    except Exception:
                        pass

    sorted_clients = sorted(list(client_round_acc.keys()))
    if not sorted_clients:
        print("[SUMMARY] No client evaluation accuracy metrics found to summarize.")
        return

    # Determine unique communication rounds
    all_rounds = sorted(list(set(
        r for c_dict in client_round_acc.values() for r in c_dict.keys()
    )))
    if not all_rounds:
        all_rounds = list(range(1, total_rounds + 1))

    # 2. Build and Print Formatted Table
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
    csv_rows: List[Dict[str, Any]] = []

    for cid in sorted_clients:
        model_id = CLIENT_MODELS.get(cid, "Unknown")
        model_short = model_id.split("/")[-1]
        ds_info = CLIENT_DATASETS.get(cid, ("Unknown", None))
        ds_name = f"{ds_info[0]}" + (f" ({ds_info[1]})" if ds_info[1] else "")
        ds_name = (ds_name[:21] + "..") if len(ds_name) > 24 else ds_name

        acc_strs = []
        acc_floats: List[Optional[float]] = []
        csv_row_entry: Dict[str, Any] = {
            "client_id": cid,
            "model_id": model_id,
            "dataset": ds_name,
        }

        for r in all_rounds:
            if r in client_round_acc[cid]:
                acc = client_round_acc[cid][r]
                acc_strs.append(f"{acc * 100:.2f}%")
                acc_floats.append(acc)
                round_client_accs[r].append(acc)
                csv_row_entry[f"round_{r}"] = round(acc, 4)
            else:
                acc_strs.append("N/A")
                acc_floats.append(None)
                csv_row_entry[f"round_{r}"] = ""

        # Compute net accuracy gain (last recorded round - first recorded round)
        valid_floats = [a for a in acc_floats if a is not None]
        if len(valid_floats) >= 2:
            net_gain = (valid_floats[-1] - valid_floats[0]) * 100
            gain_str = f"{net_gain:>+6.2f}%"
            csv_row_entry["net_gain"] = round(net_gain, 4)
        else:
            gain_str = "---"
            csv_row_entry["net_gain"] = ""

        row_cells = [f"Client {cid:02d}", model_short, ds_name] + acc_strs + [gain_str]
        lines.append(format_row(row_cells))
        csv_rows.append(csv_row_entry)

    lines.append(divider)

    # 3. Compute and Print Mean Accuracy per Round
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
            csv_mean_entry[f"round_{r}"] = round(mean_acc, 4)
        else:
            mean_strs.append("N/A")
            csv_mean_entry[f"round_{r}"] = ""

    if len(mean_floats) >= 2:
        group_gain = (mean_floats[-1] - mean_floats[0]) * 100
        group_gain_str = f"{group_gain:>+6.2f}%"
        csv_mean_entry["net_gain"] = round(group_gain, 4)
    else:
        group_gain_str = "---"
        csv_mean_entry["net_gain"] = ""

    mean_cells = ["MEAN", "GROUP AVERAGE", "--"] + mean_strs + [group_gain_str]
    lines.append(format_row(mean_cells))
    lines.append("=" * len(table_header) + "\n")
    csv_rows.append(csv_mean_entry)

    # Print summary table to console
    table_text = "\n".join(lines)
    print(table_text)

    # 4. Save Summary Table as CSV (log_dir/summary.csv)
    summary_csv_path = log_dir / "summary.csv"
    csv_fieldnames = ["client_id", "model_id", "dataset"] + [f"round_{r}" for r in all_rounds] + ["net_gain"]

    try:
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  -> Exported CSV summary table to: {summary_csv_path}")
    except Exception as e:
        logger.warning(f"Could not export {summary_csv_path}: {e}")


# ---------------------------------------------------------------------------
# 5. Package Exports
# ---------------------------------------------------------------------------
__all__ = [
    "compute_classification_metrics",
    "compute_distillation_divergence",
    "MetricTracker",
    "summarize_run",
]
