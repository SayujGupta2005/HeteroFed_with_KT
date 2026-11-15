"""Logit poisoning attack module for data_free_fd federated distillation.

This module implements five attack strategies that intercept the per-class averaged
logit payload ({class_id: np.ndarray[6]}) that a malicious client would upload to
the server each round.  All strategies are stateless callables; the runner script
decides which clients are malicious and calls ``apply_attack`` before aggregation.

Attack taxonomy
---------------
none      Benign baseline — payload is returned unchanged.
random    High-magnitude random noise with sign inversion; disrupts all classes.
zero      All-zero logit vectors — maximum-entropy teacher distribution after
          softmax, suppressing all class confidence.
fdla      Federated Distillation Logits Attack (Jeong et al. 2024, arXiv:2401.03685).
          Shuffles logit elements to maximise per-class entropy, then scales the
          magnitude by a factor γ so the poisoned vectors dominate the weighted
          average on the server.
pcfdla    Peak-Controlled FDLA (2024).  A targeted variant of FDLA: for a chosen
          source class the logit peak is relocated to a chosen target class while
          the vector L∞-norm is capped at the estimated benign peak so the attack
          evades simple norm-based detectors.  Other classes are left untouched.

Usage
-----
    from federated_emotion.attacks import LogitAttacker

    attacker = LogitAttacker(
        attack_type="pcfdla",
        source_class=3,   # anger
        target_class=1,   # -> joy
    )
    poisoned_logits = attacker.apply(honest_class_logits)

The returned dict has the same key set as the input.  Absent classes (the client
holds no examples for that class) remain absent.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Canonical emotion classes used throughout the pipeline
CANONICAL_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# ---------------------------------------------------------------------------
# Attack helper functions
# ---------------------------------------------------------------------------

def _attack_none(
    class_logits: Dict[int, np.ndarray],
    **_kwargs,
) -> Dict[int, np.ndarray]:
    """Benign baseline: return the payload unchanged."""
    return class_logits


def _attack_random(
    class_logits: Dict[int, np.ndarray],
    scale: float = 10.0,
    seed: Optional[int] = None,
    **_kwargs,
) -> Dict[int, np.ndarray]:
    """Sign-inverted, high-magnitude random noise for all classes.

    Flips the sign of honest logits then adds Gaussian noise scaled by `scale`.
    This replicates and formalises the exploratory attack already tested on this
    pipeline (sign-flip + scaling).
    """
    rng = np.random.default_rng(seed)
    poisoned: Dict[int, np.ndarray] = {}
    for cls, vec in class_logits.items():
        noise = rng.standard_normal(vec.shape).astype(np.float32)
        poisoned[cls] = (-vec + scale * noise).astype(np.float32)
    return poisoned


def _attack_zero(
    class_logits: Dict[int, np.ndarray],
    **_kwargs,
) -> Dict[int, np.ndarray]:
    """All-zero logit vectors.

    After softmax on the server, a zero vector produces a uniform distribution
    over all classes (maximum entropy), so the per-class distillation target
    degrades to random noise.
    """
    return {cls: np.zeros_like(vec, dtype=np.float32) for cls, vec in class_logits.items()}


def _attack_fdla(
    class_logits: Dict[int, np.ndarray],
    scale: float = 10.0,
    seed: Optional[int] = None,
    **_kwargs,
) -> Dict[int, np.ndarray]:
    """Federated Distillation Logits Attack (FDLA, arXiv:2401.03685).

    Two-stage attack:
      Stage 1 — Shuffle logit elements within each class vector to maximise the
                 entropy of the resulting softmax distribution.  The shuffle is
                 done by sorting in descending order then reversing rank positions
                 so the originally-confident class gets the smallest value (entropy
                 maximising permutation).
      Stage 2 — Scale the shuffled vector by `scale` so the poisoned entry
                 dominates the count-weighted average computed by
                 `aggregate_class_logits` on the server.

    Reference
    ---------
    "Logit Poisoning Attack in Federated Distillation and its Countermeasures",
    Yu et al., arXiv:2401.03685, January 2024.
    """
    rng = np.random.default_rng(seed)
    poisoned: Dict[int, np.ndarray] = {}
    for cls, vec in class_logits.items():
        v = vec.astype(np.float64)
        # Stage 1: entropy-maximising permutation
        # Sort descending, then assign the positions in ascending order so the
        # element that was most confident (largest) is now least confident.
        sort_idx = np.argsort(v)[::-1]          # indices of vals from large -> small
        reverse_rank = np.empty_like(sort_idx)
        reverse_rank[sort_idx] = np.arange(len(v))[::-1]  # invert ranks
        shuffled = v[reverse_rank]
        # Optional: add tiny jitter so identical values still break ties randomly
        shuffled += rng.uniform(-0.01, 0.01, shuffled.shape)
        # Stage 2: scale to amplify poisoned vector
        poisoned[cls] = (scale * shuffled).astype(np.float32)
    return poisoned


def _attack_pcfdla(
    class_logits: Dict[int, np.ndarray],
    source_class: int = 3,   # anger (default)
    target_class: int = 1,   # joy   (default)
    benign_peak_estimate: Optional[float] = None,
    **_kwargs,
) -> Dict[int, np.ndarray]:
    """Peak-Controlled FDLA (PCFDLA, 2024).

    Targeted attack on `source_class` only.  All other classes are left untouched
    so the payload appears mostly honest.

    For the source class vector:
      1. Estimate the benign peak value (L∞-norm of the honest vectors or use the
         caller-supplied `benign_peak_estimate`).
      2. Build a crafted vector where:
           - `target_class` logit = +benign_peak
           - `source_class` logit = -benign_peak  (suppressed)
           - all other class logits are set to small uniform negative values
         This redirects the softmax mass from source_class to target_class while
         keeping the vector's peak within the natural range of honest clients,
         thereby evading L∞-norm filters.

    Parameters
    ----------
    source_class : int
        The emotion class to degrade (default 3 = anger).
    target_class : int
        The class to which predictions are redirected (default 1 = joy).
    benign_peak_estimate : float, optional
        Override for the estimated L∞ of benign logit vectors.  When None,
        inferred as the max absolute value across all uploaded honest vectors.
    """
    poisoned: Dict[int, np.ndarray] = {}
    num_classes = 6

    # Estimate the benign peak across all classes this client uploaded
    if benign_peak_estimate is None:
        all_vals = np.concatenate(
            [np.abs(vec) for vec in class_logits.values()], axis=0
        )
        benign_peak_estimate = float(np.max(all_vals)) if len(all_vals) > 0 else 4.0

    peak = float(benign_peak_estimate)
    # Guard against degenerate near-zero peaks (e.g. very early in training)
    peak = max(peak, 1.0)

    for cls, vec in class_logits.items():
        if cls != source_class:
            # Leave non-target classes untouched
            poisoned[cls] = vec.copy()
            continue

        # Craft the poisoned vector for the source class
        crafted = np.full(num_classes, -peak / (num_classes - 1), dtype=np.float32)
        crafted[target_class] = peak        # redirect mass here
        crafted[source_class] = -peak       # suppress source class
        poisoned[cls] = crafted

    return poisoned


# ---------------------------------------------------------------------------
# Public attacker class
# ---------------------------------------------------------------------------

_ATTACK_FNS = {
    "none":    _attack_none,
    "random":  _attack_random,
    "zero":    _attack_zero,
    "fdla":    _attack_fdla,
    "pcfdla":  _attack_pcfdla,
}

SUPPORTED_ATTACKS = list(_ATTACK_FNS.keys())


class LogitAttacker:
    """Stateless callable that wraps one of the five logit poisoning strategies.

    Instantiate once per (attack_type, configuration) and call ``apply`` on each
    malicious client's honest payload before passing it to the server aggregator.

    Parameters
    ----------
    attack_type : str
        One of ``"none" | "random" | "zero" | "fdla" | "pcfdla"``.
    scale : float
        Scaling multiplier used by ``random`` and ``fdla``.  Ignored by others.
    source_class : int
        Source class index for ``pcfdla`` (the class whose predictions are
        redirected).  Default 3 = anger.
    target_class : int
        Target class index for ``pcfdla`` (where predictions are redirected).
        Default 1 = joy.
    seed : int, optional
        RNG seed for ``random`` and ``fdla``.  None = non-deterministic.
    """

    def __init__(
        self,
        attack_type: str = "none",
        scale: float = 10.0,
        source_class: int = 3,
        target_class: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        if attack_type not in _ATTACK_FNS:
            raise ValueError(
                f"Unknown attack_type '{attack_type}'. "
                f"Choose from: {SUPPORTED_ATTACKS}"
            )
        self.attack_type = attack_type
        self.scale = scale
        self.source_class = source_class
        self.target_class = target_class
        self.seed = seed
        self._fn = _ATTACK_FNS[attack_type]

    # ------------------------------------------------------------------
    def apply(
        self,
        class_logits: Optional[Dict[int, np.ndarray]],
    ) -> Optional[Dict[int, np.ndarray]]:
        """Return a (possibly poisoned) copy of ``class_logits``.

        Args:
            class_logits: Honest per-class logit mapping from
                ``average_class_logits``.  None (failed client) is passed
                through unchanged.

        Returns:
            Poisoned logit dict, or None if input was None.
        """
        if class_logits is None:
            return None
        if self.attack_type == "none":
            return class_logits  # no copy needed — attacker is benign

        try:
            return self._fn(
                class_logits,
                scale=self.scale,
                source_class=self.source_class,
                target_class=self.target_class,
                seed=self.seed,
            )
        except Exception as exc:
            logger.warning(
                f"[LogitAttacker/{self.attack_type}] Attack failed; "
                f"returning honest payload. Error: {exc}"
            )
            return class_logits

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        parts = [f"attack_type={self.attack_type!r}"]
        if self.attack_type in ("random", "fdla"):
            parts.append(f"scale={self.scale}")
        if self.attack_type == "pcfdla":
            src = CANONICAL_LABELS[self.source_class] if self.source_class < 6 else self.source_class
            tgt = CANONICAL_LABELS[self.target_class] if self.target_class < 6 else self.target_class
            parts.append(f"source={self.source_class}({src})")
            parts.append(f"target={self.target_class}({tgt})")
        return f"LogitAttacker({', '.join(parts)})"


__all__ = [
    "SUPPORTED_ATTACKS",
    "LogitAttacker",
]

