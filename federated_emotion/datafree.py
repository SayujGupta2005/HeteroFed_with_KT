"""Data-free federated distillation (FD / FedDistill) primitives.

Implements the ``data_free_fd`` federation mode: clients never share a corpus. Instead each
client uploads one averaged logit vector per class it holds, the server averages those across
clients, and clients distil against the resulting per-class targets during local training.

Reference
---------
Jeong et al., "Communication-Efficient On-Device Machine Learning: Federated Distillation and
Augmentation under Non-IID Private Data" (2018), arXiv:1811.11479. Reference implementation
consulted: HtFLlib ``flcore/clients/clientfd.py`` (Zhang et al., KDD 2025).

Deviations from the reference, both switchable for ablation
----------------------------------------------------------
1. ``aggregate_class_logits(..., class_counts=...)`` weights each client's class-c vector by how
   many class-c examples it holds. The reference uses an unweighted mean, which gives a client
   holding 13 examples of a class the same vote as one holding 475. Pass ``class_counts=None``
   to reproduce the reference.
2. ``distillation_loss`` uses a temperature-scaled KL divergence. The reference applies
   ``nn.CrossEntropyLoss(output, logit_target)`` where ``logit_target`` holds raw, unnormalised
   logits -- PyTorch then treats them as class probabilities even though they may be negative
   and do not sum to one. That is not a valid divergence, though it does train.

Communication cost is ``num_classes * num_classes`` floats per client per round: 144 bytes for
6 classes in float32.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Client side: collect and average per-class logits
# ---------------------------------------------------------------------------
def collect_class_logits(
    output: torch.Tensor,
    labels: torch.Tensor,
    store: Dict[int, List[torch.Tensor]],
) -> None:
    """Bucket a batch's logits by true label, in place.

    Called on the same forward pass that produces the supervised loss, so collecting the
    upload payload costs no additional compute.

    Args:
        output: Logit tensor of shape (batch, num_classes).
        labels: Integer label tensor of shape (batch,).
        store: Accumulator mapping class id -> list of per-example logit vectors.
    """
    detached = output.detach().float().cpu()
    for i, yy in enumerate(labels):
        store[int(yy.item())].append(detached[i])


def average_class_logits(
    store: Dict[int, List[torch.Tensor]],
) -> Dict[int, np.ndarray]:
    """Reduce collected logits to one mean vector per class.

    Args:
        store: Accumulator produced by collect_class_logits.

    Returns:
        Mapping class id -> mean logit vector of shape (num_classes,). Classes with no
        observations are absent from the result, which is what lets the server skip them.
    """
    out: Dict[int, np.ndarray] = {}
    for cls, vectors in store.items():
        if not vectors:
            continue
        out[int(cls)] = torch.stack(vectors).mean(dim=0).numpy().astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 2. Server side: aggregate across clients
# ---------------------------------------------------------------------------
def aggregate_class_logits(
    client_logits: Sequence[Optional[Dict[int, np.ndarray]]],
    class_counts: Optional[Sequence[np.ndarray]] = None,
    reliability: Optional[Sequence[Optional[np.ndarray]]] = None,
    class_losses: Optional[Sequence[Optional[np.ndarray]]] = None,
    count_exponent: float = 1.0,
    reliability_exponent: float = 1.0,
    num_classes: int = 6,
    client_ids: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> Optional[Dict[int, np.ndarray]]:
    """Aggregate per-class logit vectors across clients with optional reliability correction.

    # Weight w_k^c proportional to (n_k^c ** count_exponent) * (r_k^c ** reliability_exponent) * L_k^c
    # where n_k^c is example count, r_k^c is shrunk accuracy, and L_k^c is optional 1/(1+FD loss).
    # Reliability separates clients with equal counts but different competence.

    # Ablation: count_exp=1, rel_exp=0 is baseline; count_exp=0, rel_exp=0 is unweighted;
    # count_exp=0.5 is sqrt damping.

    # Caution: loss-based weighting is self-reinforcing. Reliability is ground-truth based
    # and avoids this. Use class_losses for ablation only.

    Args:
        client_logits: Per-client mappings from average_class_logits. None skipped.
        class_counts: Per-client example counts. None gives unweighted mean.
        reliability: Per-client reliability arrays. None disables reliability factor.
        class_losses: Per-client mean FD loss. None disables loss factor.
        count_exponent: Exponent on the count factor.
        reliability_exponent: Exponent on the reliability factor.
        num_classes: Size of the canonical label space.
        client_ids: Optional real client ids for logging; defaults to positional indices.
        verbose: Print per-class weight allocation and decomposition.

    Returns:
        Mapping class id -> aggregated logit vector, or None if no client contributed anything.
    """
    buckets: Dict[int, List[np.ndarray]] = defaultdict(list)
    weights: Dict[int, List[float]] = defaultdict(list)
    owners: Dict[int, List[int]] = defaultdict(list)
    # Tracks weight factors for logging.
    factors: Dict[int, List[Dict[str, float]]] = defaultdict(list)

    use_reliability = reliability is not None and reliability_exponent != 0.0
    use_losses = class_losses is not None

    for k, per_client in enumerate(client_logits):
        if not per_client:
            continue
        counts = class_counts[k] if class_counts is not None else None
        rel = reliability[k] if use_reliability else None
        loss = class_losses[k] if use_losses else None

        for cls, vec in per_client.items():
            c = int(cls)
            if not (0 <= c < num_classes):
                continue

            n = float(counts[c]) if counts is not None else 1.0
            if counts is not None and n <= 0.0:
                # Client reported a vector for a class its count says it does not hold.
                # Trust the count and drop the contribution.
                continue

            w_count = (n ** float(count_exponent)) if counts is not None else 1.0

            w_rel = 1.0
            if rel is not None:
                # Floor at 1e-3: prevents a single bad round from permanently silencing
                # the only holder of a rare class.
                r = float(np.clip(rel[c], 1e-3, 1.0))
                w_rel = r ** float(reliability_exponent)

            w_loss = 1.0
            if loss is not None and np.isfinite(loss[c]):
                w_loss = 1.0 / (1.0 + max(float(loss[c]), 0.0))

            w = w_count * w_rel * w_loss
            if w <= 0.0:
                continue

            buckets[c].append(np.asarray(vec, dtype=np.float32))
            weights[c].append(w)
            owners[c].append(k)
            factors[c].append({"n": n, "count": w_count, "rel": w_rel, "loss": w_loss})

    if not buckets:
        print("\n[WARNING] Data-free aggregation: no client contributed any class logits.")
        return None

    global_logits: Dict[int, np.ndarray] = {}
    normalised: Dict[int, np.ndarray] = {}
    for cls in sorted(buckets):
        w = np.asarray(weights[cls], dtype=np.float64)
        w = w / max(w.sum(), 1e-12)
        normalised[cls] = w
        stacked = np.stack(buckets[cls], axis=0)
        global_logits[cls] = (stacked * w[:, None]).sum(axis=0).astype(np.float32)

    if verbose:
        if class_counts is None:
            mode = "unweighted (reference FD)"
        else:
            # Resolve aggregation mode label for logging.
            bits = [f"count^{count_exponent:g}"]
            if use_reliability:
                bits.append(f"reliability^{reliability_exponent:g}")
            if use_losses:
                bits.append("1/(1+loss)")
            mode = " x ".join(bits)

        def label(idx: int) -> str:
            return f"c{client_ids[idx]}" if client_ids is not None else f"c{idx}"

        print("\n" + "-" * 78)
        print(f"[Server] Data-free per-class logit aggregation ({mode})")
        for cls in sorted(buckets):
            w = normalised[cls]
            print(f"  class {cls}: {len(w)} contributor(s)")
            for i in range(len(w)):
                f = factors[cls][i]
                extra = ""
                if use_reliability:
                    extra += f" rel={f['rel'] ** (1.0 / max(reliability_exponent, 1e-9)):.3f}"
                if use_losses:
                    extra += f" loss_factor={f['loss']:.3f}"
                print(
                    f"      {label(owners[cls][i]):<6} n={int(f['n']):>4}"
                    f"{extra}  ->  weight {w[i] * 100:5.1f}%"
                )
        missing = [c for c in range(num_classes) if c not in global_logits]
        if missing:
            print(f"  [WARNING] no client holds class(es) {missing}; they cannot be learned.")
        print("-" * 78)

    return global_logits


# ---------------------------------------------------------------------------
# 3. Client side: the distillation term
# ---------------------------------------------------------------------------
def build_logit_targets(
    output: torch.Tensor,
    labels: torch.Tensor,
    global_logits: Dict[int, np.ndarray],
) -> Optional[torch.Tensor]:
    """Assemble the per-example distillation target for a batch.

    Row i of the target is the federation's averaged logit vector for example i's true class.
    Rows whose class is absent from global_logits keep the model's own output, so they
    contribute exactly zero to the divergence.

    Args:
        output: Logit tensor of shape (batch, num_classes).
        labels: Integer label tensor of shape (batch,).
        global_logits: Server-aggregated mapping class id -> logit vector.

    Returns:
        Detached target tensor of shape (batch, num_classes), or None if no row in this batch
        has a corresponding global vector.
    """
    if not global_logits:
        return None

    target = output.detach().clone()
    matched = 0
    for i, yy in enumerate(labels):
        cls = int(yy.item())
        vec = global_logits.get(cls)
        if vec is not None:
            target[i, :] = torch.as_tensor(vec, dtype=target.dtype, device=target.device)
            matched += 1

    return target if matched > 0 else None


def distillation_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 2.0,
    return_per_example: bool = False,
) -> torch.Tensor:
    """Temperature-scaled KL divergence from per-class targets.

    # T**2 restores gradient magnitude after dividing logits by T (Hinton et al., 2015).

    Args:
        output: Student logits of shape (batch, num_classes).
        target: Detached teacher logits of the same shape, from build_logit_targets.
        temperature: Softmax temperature. Must be positive.
        return_per_example: If True, return un-reduced divergence to attribute
            difficulty per class. No extra compute cost.

    Returns:
        Scalar loss, or (scalar loss, per-example loss of shape (batch,)).
    """
    t = max(float(temperature), 1e-8)
    student_log_probs = F.log_softmax(output / t, dim=-1)
    teacher_probs = F.softmax(target / t, dim=-1)

    per_example = F.kl_div(
        student_log_probs, teacher_probs, reduction="none"
    ).sum(dim=-1) * (t ** 2)
    scalar = per_example.mean()

    if return_per_example:
        return scalar, per_example.detach()
    return scalar


def accumulate_class_losses(
    per_example_loss: torch.Tensor,
    labels: torch.Tensor,
    store: Dict[int, List[float]],
) -> None:
    """Bucket per-example distillation losses by label, in place.

    Args:
        per_example_loss: Detached per-example divergence of shape (batch,).
        labels: Integer label tensor of shape (batch,).
        store: Accumulator mapping class id -> list of per-example losses.
    """
    losses = per_example_loss.detach().float().cpu()
    for i, yy in enumerate(labels):
        store[int(yy.item())].append(float(losses[i]))


# ---------------------------------------------------------------------------
# 3b. Reliability: how much does a client actually know about each class?
# ---------------------------------------------------------------------------
def compute_reliability(
    per_class_correct: np.ndarray,
    per_class_total: np.ndarray,
    overall_accuracy: float,
    prior_strength: float = 5.0,
    num_classes: int = 6,
) -> np.ndarray:
    """Shrunk per-class accuracy for aggregation reliability weighting.

    # Raw accuracy on small holdouts is noisy. Shrink each estimate toward overall
    # accuracy using an empirical-Bayes pseudo-count.

    # m=0: raw accuracy; large m: collapses to overall accuracy.

    Args:
        per_class_correct: Correct predictions per class on the local holdout.
        per_class_total: Held-out examples per class.
        overall_accuracy: The client's accuracy across all held-out examples.
        prior_strength: Pseudo-count m. Larger = more shrinkage toward the overall rate.
        num_classes: Size of the canonical label space.

    Returns array in [0, 1]. Missing classes fall back to overall accuracy.
    """
    correct = np.asarray(per_class_correct, dtype=np.float64)[:num_classes]
    total = np.asarray(per_class_total, dtype=np.float64)[:num_classes]
    m = max(float(prior_strength), 0.0)
    prior = float(np.clip(overall_accuracy, 0.0, 1.0))

    numer = correct + m * prior
    denom = total + m
    # Classes with no held-out examples and m == 0 would divide by zero; fall back to prior.
    out = np.where(denom > 0, numer / np.clip(denom, 1e-12, None), prior)
    return np.clip(out, 0.0, 1.0).astype(np.float64)


# ---------------------------------------------------------------------------
# 4. Persistence helpers
# ---------------------------------------------------------------------------
def logits_to_serializable(global_logits: Optional[Dict[int, np.ndarray]]) -> Dict[str, List[float]]:
    """Convert an aggregated logit mapping to a JSON-friendly dict."""
    if not global_logits:
        return {}
    return {str(int(k)): np.asarray(v, dtype=np.float32).tolist() for k, v in global_logits.items()}


def logits_from_serializable(data: Optional[Dict[str, List[float]]]) -> Optional[Dict[int, np.ndarray]]:
    """Inverse of logits_to_serializable."""
    if not data:
        return None
    return {int(k): np.asarray(v, dtype=np.float32) for k, v in data.items()}


__all__ = [
    "collect_class_logits",
    "average_class_logits",
    "aggregate_class_logits",
    "accumulate_class_losses",
    "compute_reliability",
    "build_logit_targets",
    "distillation_loss",
    "logits_to_serializable",
    "logits_from_serializable",
]
