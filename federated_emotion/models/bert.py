from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertModel

DEFAULT_BERT = "bert-base-uncased"


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


def _slice_linear(layer: nn.Linear, index: torch.Tensor, dim: int) -> nn.Linear:
    index = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, index).clone().detach()
    bias = layer.bias.clone().detach() if dim == 1 else layer.bias.index_select(0, index).clone().detach()
    out_f, in_f = weight.shape
    new = nn.Linear(in_f, out_f, device=weight.device, dtype=weight.dtype)
    new.weight.data.copy_(weight)
    new.bias.data.copy_(bias)
    return new


class PrunableBert(nn.Module):
    def __init__(self, model_name: str = DEFAULT_BERT, num_labels: int = 6, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        if pretrained:
            self.bert = BertModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            self.bert = BertModel(BertConfig.from_pretrained(model_name), add_pooling_layer=False)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden, num_labels)
        self.capacity = 1.0
        self.pruned_heads: Dict[int, List[int]] = {}

    @property
    def hidden_size(self) -> int:
        return self.bert.config.hidden_size

    @property
    def num_layers(self) -> int:
        return self.bert.config.num_hidden_layers

    def head(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(pooled))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, object]:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states
        pooled = [masked_mean(h, attention_mask) for h in hidden_states]
        return {
            "logits": self.head(pooled[-1]),
            "hidden_states": hidden_states,
            "pooled": pooled,
        }

    def extended_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        ext = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - ext) * torch.finfo(dtype).min

    def forward_from_layer(self, hidden: torch.Tensor, start: int, attention_mask: torch.Tensor) -> torch.Tensor:
        ext = self.extended_mask(attention_mask, hidden.dtype)
        h = hidden
        for layer in self.bert.encoder.layer[start:]:
            out = layer(h, attention_mask=ext)
            h = out[0] if isinstance(out, tuple) else out
        return self.head(masked_mean(h, attention_mask))

    def head_scores(self, layer_idx: int) -> torch.Tensor:
        att = self.bert.encoder.layer[layer_idx].attention
        n_heads = att.self.num_attention_heads
        d = att.self.attention_head_size
        q, k, v = att.self.query.weight, att.self.key.weight, att.self.value.weight
        o = att.output.dense.weight
        rows = (q.abs() + k.abs() + v.abs()).sum(dim=1).view(n_heads, d).sum(dim=1)
        cols = o.abs().sum(dim=0).view(n_heads, d).sum(dim=1)
        return rows + cols

    def ffn_scores(self, layer_idx: int) -> torch.Tensor:
        layer = self.bert.encoder.layer[layer_idx]
        return layer.intermediate.dense.weight.abs().sum(dim=1) + layer.output.dense.weight.abs().sum(dim=0)

    def prune_ffn(self, layer_idx: int, keep: torch.Tensor) -> None:
        layer = self.bert.encoder.layer[layer_idx]
        keep = keep.sort().values
        layer.intermediate.dense = _slice_linear(layer.intermediate.dense, keep, dim=0)
        layer.output.dense = _slice_linear(layer.output.dense, keep, dim=1)

    @torch.no_grad()
    def prune(self, capacity: float) -> None:
        if capacity >= 1.0:
            return
        heads_to_prune: Dict[int, List[int]] = {}
        for l in range(self.num_layers):
            scores = self.head_scores(l)
            n_keep = max(1, int(round(scores.numel() * capacity)))
            keep = set(torch.topk(scores, n_keep).indices.tolist())
            heads_to_prune[l] = [h for h in range(scores.numel()) if h not in keep]

            f_scores = self.ffn_scores(l)
            f_keep = max(1, int(round(f_scores.numel() * capacity)))
            self.prune_ffn(l, torch.topk(f_scores, f_keep).indices)
        self.bert.prune_heads(heads_to_prune)
        self.pruned_heads = heads_to_prune
        self.capacity = float(capacity)

    def structure(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "capacity": self.capacity,
            "pruned_heads": {str(k): v for k, v in self.pruned_heads.items()},
            "ffn_sizes": [l.intermediate.dense.out_features for l in self.bert.encoder.layer],
        }

    def describe(self) -> str:
        heads = [l.attention.self.num_attention_heads for l in self.bert.encoder.layer]
        ffn = [l.intermediate.dense.out_features for l in self.bert.encoder.layer]
        return (
            f"capacity={self.capacity:.2f} heads/layer={heads[0]} ffn/layer={ffn[0]} "
            f"encoder_params={encoder_param_count(self):,} total_params={param_count(self):,}"
        )


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def encoder_param_count(model: PrunableBert) -> int:
    return param_count(model.bert.encoder)


def save_bert(model: PrunableBert, path: Union[str, Path], extra: Optional[Dict[str, torch.Tensor]] = None) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "structure.json", "w", encoding="utf-8") as f:
        json.dump(model.structure(), f, indent=2)
    torch.save(model.state_dict(), path / "model.pt")
    if extra:
        torch.save(extra, path / "extra.pt")


def load_bert(path: Union[str, Path], device: Optional[torch.device] = None) -> PrunableBert:
    path = Path(path)
    with open(path / "structure.json", "r", encoding="utf-8") as f:
        s = json.load(f)
    model = PrunableBert(s["model_name"], s["num_labels"], pretrained=False)
    heads = {int(k): v for k, v in s["pruned_heads"].items()}
    for l, size in enumerate(s["ffn_sizes"]):
        if size != model.bert.encoder.layer[l].intermediate.dense.out_features:
            model.prune_ffn(l, torch.arange(size))
    if heads:
        model.bert.prune_heads(heads)
    model.pruned_heads = heads
    model.capacity = float(s["capacity"])
    state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device) if device is not None else model


def load_extra(path: Union[str, Path]) -> Optional[Dict[str, torch.Tensor]]:
    p = Path(path) / "extra.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=True)


def build_client_models(
    model_name: str,
    num_labels: int,
    capacities: Sequence[float],
    client_ids: Sequence[int],
    out_dir: Union[str, Path],
) -> Dict[int, str]:
    global_model = PrunableBert(model_name, num_labels, pretrained=True)
    full_encoder = encoder_param_count(global_model)
    full_total = param_count(global_model)
    print(f"  Global BERT '{model_name}': {global_model.describe()}")
    paths: Dict[int, str] = {}
    for cid, cap in zip(client_ids, capacities):
        client_model = PrunableBert(model_name, num_labels, pretrained=False)
        client_model.load_state_dict(global_model.state_dict())
        client_model.prune(cap)
        target = Path(out_dir) / f"client_{cid}" / "round_0"
        save_bert(client_model, target)
        paths[cid] = str(target)
        enc_ratio = encoder_param_count(client_model) / full_encoder
        tot_ratio = param_count(client_model) / full_total
        print(
            f"  -> Client {cid:02d}: {client_model.describe()} "
            f"| encoder {enc_ratio * 100:.1f}% | total {tot_ratio * 100:.1f}% of global"
        )
    return paths


def get_bert_tokenizer(model_name: str = DEFAULT_BERT):
    return AutoTokenizer.from_pretrained(model_name)


__all__ = [
    "PrunableBert",
    "masked_mean",
    "param_count",
    "encoder_param_count",
    "save_bert",
    "load_bert",
    "load_extra",
    "build_client_models",
    "get_bert_tokenizer",
]
