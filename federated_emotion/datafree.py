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
    num_classes: int = 6,
    verbose: bool = True,
) -> Optional[Dict[int, np.ndarray]]:
    """Average per-class logit vectors across clients.

    A client contributes to class c only if it actually holds examples of class c. When
    class_counts is supplied, contributions are weighted by those counts so that a client with
    a handful of examples cannot outvote one with hundreds.

    Args:
        client_logits: Per-client mappings from average_class_logits. None entries (failed
            clients) are skipped.
        class_counts: Optional per-client arrays of shape (num_classes,) giving each client's
            example count per class. Must align positionally with client_logits. Pass None for
            the reference unweighted mean.
        num_classes: Size of the canonical label space.
        verbose: Print the per-class weight allocation.

    Returns:
        Mapping class id -> aggregated logit vector, or None if no client contributed anything.
    """
    buckets: Dict[int, List[np.ndarray]] = defaultdict(list)
    weights: Dict[int, List[float]] = defaultdict(list)
    owners: Dict[int, List[int]] = defaultdict(list)

    for k, per_client in enumerate(client_logits):
        if not per_client:
            continue
        counts = class_counts[k] if class_counts is not None else None
        for cls, vec in per_client.items():
            if not (0 <= int(cls) < num_classes):
                continue
            w = float(counts[int(cls)]) if counts is not None else 1.0
            if w <= 0.0:
                # Client reported a vector for a class its count says it does not hold.
                # Trust the count and drop the contribution.
                continue
            buckets[int(cls)].append(np.asarray(vec, dtype=np.float32))
            weights[int(cls)].append(w)
            owners[int(cls)].append(k)

    if not buckets:
        print("\n[WARNING] Data-free aggregation: no client contributed any class logits.")
        return None

    global_logits: Dict[int, np.ndarray] = {}
    for cls in sorted(buckets):
        w = np.asarray(weights[cls], dtype=np.float64)
        w = w / max(w.sum(), 1e-12)
        stacked = np.stack(buckets[cls], axis=0)
        global_logits[cls] = (stacked * w[:, None]).sum(axis=0).astype(np.float32)

    if verbose:
        mode = "count-weighted" if class_counts is not None else "unweighted (reference FD)"
        print("\n" + "-" * 70)
        print(f"[Server] Data-free per-class logit aggregation ({mode})")
        for cls in sorted(buckets):
            w = np.asarray(weights[cls], dtype=np.float64)
            w = w / max(w.sum(), 1e-12)
            parts = ", ".join(
                f"c{owners[cls][i]}={w[i] * 100:.1f}%" for i in range(len(w))
            )
            print(f"  class {cls}: {len(w)} contributor(s) | {parts}")
        missing = [c for c in range(num_classes) if c not in global_logits]
        if missing:
            print(f"  [WARNING] no client holds class(es) {missing}; they cannot be learned.")
        print("-" * 70)

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
) -> torch.Tensor:
    """Temperature-scaled KL divergence from the federation's per-class targets.

    The T**2 factor restores the gradient magnitude that dividing the logits by T removes, so
    the loss weight means the same thing at any temperature (Hinton et al., 2015).

    Args:
        output: Student logits of shape (batch, num_classes).
        target: Detached teacher logits of the same shape, from build_logit_targets.
        temperature: Softmax temperature. Must be positive.

    Returns:
        Scalar loss.
    """
    t = max(float(temperature), 1e-8)
    student_log_probs = F.log_softmax(output / t, dim=-1)
    teacher_probs = F.softmax(target / t, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t ** 2)


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
    "build_logit_targets",
    "distillation_loss",
    "logits_to_serializable",
    "logits_from_serializable",
]
