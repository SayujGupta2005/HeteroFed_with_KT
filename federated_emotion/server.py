"""Central Federated Server module for logit consensus aggregation and client weighting.

This module provides:
1. aggregate: Aggregates client-generated logits over the public KD transfer pool into
   consensus teacher soft label distributions using uniform or accuracy-weighted softmax aggregation.
2. FederatedServer: High-level server orchestrator managing communication rounds and state.
"""

from __future__ import annotations

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

from federated_emotion.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Numerical Helper Functions
# ---------------------------------------------------------------------------
def _stable_softmax(
    x: np.ndarray,
    temperature: float = 1.0,
    axis: int = -1,
) -> np.ndarray:
    """Compute numerically stable softmax probabilities over numpy arrays."""
    temp = max(float(temperature), 1e-8)
    scaled = x / temp
    max_val = np.max(scaled, axis=axis, keepdims=True)
    exp_x = np.exp(scaled - max_val)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / np.clip(sum_exp, a_min=1e-12, a_max=None)


# ---------------------------------------------------------------------------
# 2. Main Aggregation Function
# ---------------------------------------------------------------------------
def aggregate(
    client_results: List[Optional[Dict[str, Any]]],
    config: Config,
) -> Optional[np.ndarray]:
    """Aggregate per-client public dataset logits into a consensus soft label distribution.

    Algorithm:
    1. Filters out failed clients (None or malformed result dicts).
    2. Converts each client's raw logits on public_kd_pool to temperature-scaled
       softmax probability distribution P_i using config.kd_temperature.
    3. Calculates client aggregation weights w_i:
       - Single client: w_0 = 1.0
       - "uniform": w_i = 1 / K
       - "accuracy_weighted": w_i = softmax(accuracy_vector / config.aggregation_temperature)
    4. Computes weighted sum of soft distributions:
       P_consensus = sum_i(w_i * P_i) of shape (public_kd_pool_size, 6).

    Args:
        client_results: List of result dictionaries from run_client_round, where each dict has
                        {"client_id": int, "logits_on_kd_pool": np.ndarray, "eval_accuracy": float}.
        config: Global Config dataclass instance.

    Returns:
        Consensus teacher soft probabilities array of shape (public_kd_pool_size, 6),
        or None if no valid client results are present.
    """
    # Filter out None entries from failed clients
    valid_results: List[Dict[str, Any]] = [
        r for r in client_results
        if r is not None and "logits_on_kd_pool" in r and "eval_accuracy" in r
    ]

    num_valid = len(valid_results)
    if num_valid == 0:
        print("\n[WARNING] Server Aggregation: No valid client results to aggregate!")
        logger.warning("No valid client results received by server.")
        return None

    client_ids = [int(r["client_id"]) for r in valid_results]
    accuracies = np.array([float(r["eval_accuracy"]) for r in valid_results], dtype=np.float64)

    # 1. Convert each client's logits into temperature-scaled softmax probabilities
    probs_list: List[np.ndarray] = []
    for r in valid_results:
        raw_logits = np.asarray(r["logits_on_kd_pool"], dtype=np.float32)
        # Apply temperature-scaled softmax across class dimension (axis=-1)
        soft_probs = _stable_softmax(raw_logits, temperature=config.kd_temperature, axis=-1)
        probs_list.append(soft_probs)

    # 2. Compute Aggregation Weights
    if num_valid == 1:
        # Edge case: exactly 1 client succeeded
        weights = np.array([1.0], dtype=np.float64)
    elif config.aggregation_mode == "uniform":
        weights = np.ones(num_valid, dtype=np.float64) / float(num_valid)
    elif config.aggregation_mode == "accuracy_weighted":
        # Softmax over accuracy vector scaled by aggregation_temperature
        weights = _stable_softmax(
            accuracies,
            temperature=config.aggregation_temperature,
            axis=0,
        )
    else:
        logger.warning(
            f"Unknown aggregation_mode '{config.aggregation_mode}'; falling back to uniform."
        )
        weights = np.ones(num_valid, dtype=np.float64) / float(num_valid)

    # 3. Print computed weights for transparency and debugging
    print("\n" + "-" * 70)
    print(f"[Server] Consensus Logit Aggregation ({num_valid} Active Clients)")
    print(f"  Aggregation Mode        : {config.aggregation_mode}")
    print(f"  Aggregation Temperature : {config.aggregation_temperature}")
    print(f"  KD Temperature          : {config.kd_temperature}")
    print("  Client Weight Allocation:")
    for cid, acc, w in zip(client_ids, accuracies, weights):
        print(f"    - Client {cid:02d}: weight = {w:.4f} ({w * 100:.2f}%) | Holdout Accuracy = {acc * 100:.2f}%")
    print("-" * 70)

    # 4. Compute Weighted Sum of Probabilities
    # Shape: (public_kd_pool_size, num_classes)
    avg_soft_labels = np.zeros_like(probs_list[0], dtype=np.float32)
    for w, p in zip(weights, probs_list):
        avg_soft_labels += (float(w) * p).astype(np.float32)

    # Ensure row-level probability normalization
    row_sums = np.sum(avg_soft_labels, axis=-1, keepdims=True)
    avg_soft_labels = avg_soft_labels / np.clip(row_sums, a_min=1e-12, a_max=None)

    return avg_soft_labels


# ---------------------------------------------------------------------------
# 3. Server Class Encapsulation
# ---------------------------------------------------------------------------
class FederatedServer:
    """Stateful server managing consensus Knowledge Distillation across communication rounds."""

    def __init__(self, config: Config) -> None:
        """Initialize the federated server with pipeline config."""
        self.config = config
        self.current_round: int = 0
        self.history: List[Dict[str, Any]] = []

    def aggregate(
        self,
        client_results: List[Optional[Dict[str, Any]]],
    ) -> Optional[np.ndarray]:
        """Aggregate client outputs using module-level aggregate function."""
        return aggregate(client_results, self.config)


# ---------------------------------------------------------------------------
# 4. Package Exports
# ---------------------------------------------------------------------------
__all__ = [
    "aggregate",
    "FederatedServer",
]
