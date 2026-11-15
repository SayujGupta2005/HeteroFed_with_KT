from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from federated_emotion.models.bert import masked_mean


def identity_bridge(hidden_size: int) -> nn.Linear:
    bridge = nn.Linear(hidden_size, hidden_size)
    with torch.no_grad():
        bridge.weight.copy_(torch.eye(hidden_size))
        bridge.bias.zero_()
    return bridge


def logit_kd(s_logits: torch.Tensor, t_logits: torch.Tensor, T: float) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(s_logits / T, dim=-1),
        F.softmax(t_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)


def inject_prototype(hidden_seq: torch.Tensor, mask: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden_seq.dtype)
    centered = hidden_seq - masked_mean(hidden_seq, mask).unsqueeze(1)
    return (centered + proto.unsqueeze(1)) * m


class ClassTable:
    def __init__(self, table: torch.Tensor, present: torch.Tensor):
        self.table = table
        self.present = present

    @classmethod
    def from_dict(cls, vectors: Dict[int, np.ndarray], num_classes: int, device: torch.device) -> "ClassTable":
        dim = len(next(iter(vectors.values())))
        table = torch.zeros(num_classes, dim, device=device)
        present = torch.zeros(num_classes, dtype=torch.bool, device=device)
        for c, v in vectors.items():
            if 0 <= int(c) < num_classes:
                table[int(c)] = torch.as_tensor(v, dtype=torch.float32, device=device)
                present[int(c)] = True
        return cls(table, present)

    def lookup(self, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.table[labels], self.present[labels]


class GlobalTargets:
    def __init__(self, payload: Dict[str, object], num_classes: int, device: torch.device):
        self.logits = ClassTable.from_dict(payload["logits"], num_classes, device)
        self.features = {
            int(k): ClassTable.from_dict(v, num_classes, device)
            for k, v in payload.get("features", {}).items()
            if v
        }


class FCFDLoss(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        positions: Sequence[int],
        num_paths: int = 2,
        kd_weight: float = 1.0,
        app_weight: float = 1.0,
        func_weight: float = 1.0,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.positions = [int(k) for k in positions]
        self.num_paths = num_paths
        self.kd_weight = kd_weight
        self.app_weight = app_weight
        self.func_weight = func_weight
        self.T = temperature
        self.bridge_st = nn.ModuleDict({str(k): identity_bridge(hidden_size) for k in self.positions})
        self.bridge_ts = nn.ModuleDict({str(k): identity_bridge(hidden_size) for k in self.positions})

    def sample_positions(self, available: List[int]) -> List[int]:
        return random.sample(available, min(self.num_paths, len(available)))

    def forward(
        self,
        student: nn.Module,
        out: Dict[str, object],
        labels: torch.Tensor,
        mask: torch.Tensor,
        targets: GlobalTargets,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        zero = out["logits"].new_zeros(())
        t_logits, has_logit = targets.logits.lookup(labels)
        if not has_logit.any():
            return zero, {"kd": 0.0, "app": 0.0, "func": 0.0}

        s_logits = out["logits"][has_logit]
        t_sel = t_logits[has_logit]
        l_kd = logit_kd(s_logits, t_sel, self.T)

        l_app = zero
        available = [k for k in self.positions if k in targets.features]
        for k in available:
            proto, has = targets.features[k].lookup(labels)
            if has.any():
                bridged = self.bridge_st[str(k)](out["pooled"][k][has])
                l_app = l_app + F.mse_loss(bridged, proto[has])

        l_func = zero
        for k in self.sample_positions(available):
            proto, has = targets.features[k].lookup(labels)
            keep = has & has_logit
            if not keep.any():
                continue
            injected = inject_prototype(
                out["hidden_states"][k][keep],
                mask[keep],
                self.bridge_ts[str(k)](proto[keep]),
            )
            ts_logits = student.forward_from_layer(injected, k, mask[keep])
            l_func = l_func + logit_kd(ts_logits, t_logits[keep], self.T)

        total = self.kd_weight * l_kd + self.app_weight * l_app + self.func_weight * l_func
        return total, {"kd": l_kd.item(), "app": float(l_app.detach()), "func": float(l_func.detach())}


__all__ = ["identity_bridge", "logit_kd", "inject_prototype", "GlobalTargets", "FCFDLoss"]
