from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from federated_emotion.config import Config
from federated_emotion.datafree import (
    average_class_features,
    average_class_logits,
    collect_class_features,
    collect_class_logits,
    compute_reliability,
)
from federated_emotion.data.loaders import compute_class_counts
from federated_emotion.fcfd import FCFDLoss, GlobalTargets
from federated_emotion.models.bert import get_bert_tokenizer, load_bert, load_extra, save_bert
from federated_emotion.profiler import global_profiler

logger = logging.getLogger(__name__)


def make_collate(tokenizer: Any, max_length: int):
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        enc = tokenizer(
            [b["text"] for b in batch],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": torch.tensor([int(b["label"]) for b in batch], dtype=torch.long),
        }

    return collate


def client_label(client_id: int, config: Config) -> str:
    ids = config.active_client_ids or list(range(1, config.num_clients + 1))
    cap = config.client_capacities[ids.index(client_id)] if client_id in ids else 1.0
    return f"{config.bert_model_name}@{int(round(cap * 100))}%"


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> Dict[str, Any]:
    model.eval()
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    predicted = np.zeros(num_classes, dtype=np.int64)
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))["logits"]
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels = batch["label"].numpy()
        for c in range(num_classes):
            total[c] += int((labels == c).sum())
            predicted[c] += int((preds == c).sum())
            correct[c] += int(((labels == c) & (preds == c)).sum())
    acc = float(correct.sum() / max(total.sum(), 1))
    denom = predicted + total
    f1 = np.where(denom > 0, 2.0 * correct / np.clip(denom, 1, None), 0.0)
    macro_f1 = float(f1[total > 0].mean()) if (total > 0).any() else 0.0
    return {"accuracy": acc, "correct": correct, "total": total, "f1": f1, "macro_f1": macro_f1}


def build_fcfd(model: nn.Module, config: Config) -> FCFDLoss:
    return FCFDLoss(
        hidden_size=model.hidden_size,
        positions=config.fcfd_positions,
        num_paths=config.fcfd_num_paths,
        kd_weight=config.fcfd_kd_weight,
        app_weight=config.fcfd_app_weight,
        func_weight=config.fcfd_func_weight,
        temperature=config.fd_temperature,
    )


def run_client_round(
    client_id: int,
    round_num: int,
    config: Config,
    private_dataset: Dataset,
    local_eval_holdout: Dataset,
    global_test: Optional[Dataset] = None,
    global_payload: Optional[Dict[str, object]] = None,
    cross_eval_datasets: Optional[Dict[str, Dataset]] = None,
    dataset_name: str = "",
) -> Optional[Dict[str, Any]]:
    model = None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = client_label(client_id, config)
        ckpt_root = Path(config.checkpoint_dir) / f"client_{client_id}"
        prev_dir = ckpt_root / f"round_{round_num - 1}"

        print("\n" + "=" * 80)
        print(f"[Client {client_id:02d}] Round {round_num}/{config.num_rounds} | {model_id}")
        print(f"  Train: {len(private_dataset)} | Local holdout: {len(local_eval_holdout)}")
        print("=" * 80)

        model = load_bert(prev_dir, device)
        fcfd = build_fcfd(model, config).to(device)
        extra = load_extra(prev_dir)
        if extra is not None:
            fcfd.load_state_dict(extra)
        print(f"  -> Loaded {prev_dir} ({model.describe()})")

        tokenizer = get_bert_tokenizer(config.bert_model_name)
        collate = make_collate(tokenizer, config.max_seq_length)
        train_loader = DataLoader(private_dataset, batch_size=config.batch_size_train, shuffle=True, collate_fn=collate)

        is_kd_active = round_num > config.kd_warmup_rounds and bool(global_payload)
        targets = GlobalTargets(global_payload, config.num_classes, device) if is_kd_active else None
        print(
            "  -> FCFD distillation "
            + (f"ACTIVE (positions={config.fcfd_positions}, paths={config.fcfd_num_paths})" if is_kd_active else "INACTIVE (warmup)")
        )

        params = list(model.parameters()) + (list(fcfd.parameters()) if is_kd_active else [])
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        ce_loss_fn = nn.CrossEntropyLoss()

        logit_store: Dict[int, List[torch.Tensor]] = defaultdict(list)
        feature_store: Dict[int, Dict[int, List[torch.Tensor]]] = {}
        last_epoch = config.local_epochs

        global_profiler.start("Client_Local_Training")
        for epoch in range(1, config.local_epochs + 1):
            model.train()
            sums = defaultdict(float)
            steps = 0
            for batch in tqdm(train_loader, desc=f"  Epoch [{epoch:02d}/{config.local_epochs:02d}]", leave=False):
                input_ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                out = model(input_ids, mask)
                loss_ce = ce_loss_fn(out["logits"], labels)
                loss = loss_ce
                stats = {"kd": 0.0, "app": 0.0, "func": 0.0}
                if is_kd_active:
                    loss_fcfd, stats = fcfd(model, out, labels, mask, targets)
                    loss = loss + loss_fcfd
                loss.backward()
                optimizer.step()

                if epoch == last_epoch:
                    collect_class_logits(out["logits"], labels, logit_store)
                    collect_class_features(out["pooled"], labels, config.fcfd_positions, feature_store)

                sums["ce"] += loss_ce.item()
                sums["total"] += loss.item()
                for k, v in stats.items():
                    sums[k] += v
                steps += 1

            avg = {k: v / max(steps, 1) for k, v in sums.items()}
            print(
                f"  Epoch [{epoch:02d}/{config.local_epochs:02d}] total {avg['total']:.4f} | ce {avg['ce']:.4f}"
                + (f" | kd {avg['kd']:.4f} app {avg['app']:.4f} func {avg['func']:.4f}" if is_kd_active else "")
            )
        global_profiler.stop("Client_Local_Training")

        payload = {
            "logits": average_class_logits(logit_store),
            "features": average_class_features(feature_store),
        }
        class_counts = compute_class_counts(private_dataset, config.num_classes)
        held = sorted(payload["logits"].keys())
        n_floats = len(held) * (config.num_classes + len(config.fcfd_positions) * model.hidden_size)
        print(f"  -> Per-class payload for classes {held}: {n_floats} floats ({n_floats * 4 / 1024:.1f} KiB)")

        global_profiler.start("Client_Local_Eval")
        eval_loader = DataLoader(local_eval_holdout, batch_size=config.batch_size_infer, shuffle=False, collate_fn=collate)
        local = evaluate(model, eval_loader, device, config.num_classes)
        test = None
        if global_test is not None and len(global_test) > 0:
            test_loader = DataLoader(global_test, batch_size=config.batch_size_infer, shuffle=False, collate_fn=collate)
            test = evaluate(model, test_loader, device, config.num_classes)
        global_profiler.stop("Client_Local_Eval")

        reliability = compute_reliability(
            local["correct"],
            local["total"],
            local["accuracy"],
            prior_strength=config.fd_reliability_prior,
            num_classes=config.num_classes,
        )
        print(
            f"  [Client {client_id:02d}] local acc {local['accuracy'] * 100:.2f}% (macro-F1 {local['macro_f1']:.3f})"
            + (f" | global test acc {test['accuracy'] * 100:.2f}% (macro-F1 {test['macro_f1']:.3f})" if test else "")
        )

        cross_eval_accuracies: Dict[str, float] = {}
        if cross_eval_datasets:
            global_profiler.start("Client_Cross_Eval")
            for name, ds in cross_eval_datasets.items():
                if len(ds) == 0:
                    continue
                loader = DataLoader(ds, batch_size=config.batch_size_infer, shuffle=False, collate_fn=collate)
                cross_eval_accuracies[name] = evaluate(model, loader, device, config.num_classes)["accuracy"]
            global_profiler.stop("Client_Cross_Eval")

        save_dir = ckpt_root / f"round_{round_num}"
        save_bert(model, save_dir, extra=fcfd.state_dict())
        print(f"  -> Checkpoint saved to: {save_dir}")

        result = {
            "client_id": client_id,
            "mode": config.mode,
            "eval_accuracy": local["accuracy"],
            "eval_correct": int(local["correct"].sum()),
            "eval_total": int(local["total"].sum()),
            "eval_source": "own local holdout",
            "test_accuracy": test["accuracy"] if test else None,
            "test_macro_f1": test["macro_f1"] if test else None,
            "macro_f1": local["macro_f1"],
            "per_class_f1": local["f1"],
            "cross_eval_accuracies": cross_eval_accuracies,
            "class_payload": payload,
            "class_logits": payload["logits"],
            "class_counts": class_counts,
            "reliability": reliability,
            "class_losses": None,
            "model_id": model_id,
            "capacity": model.capacity,
            "dataset_name": dataset_name,
            "num_private_examples": len(private_dataset),
            "num_eval_holdout": len(local_eval_holdout),
            "num_kd_pool": 0,
            "num_train_steps": steps,
            "local_epochs": config.local_epochs,
            "avg_ce_loss": avg["ce"],
            "avg_kd_loss": avg["kd"] + avg["app"] + avg["func"] if is_kd_active else 0.0,
            "avg_total_loss": avg["total"],
            "kd_active": is_kd_active,
        }
        del model, fcfd, optimizer
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    except Exception as e:
        print(f"\n[WARNING] Client {client_id} failed in round {round_num}: {e}")
        logger.exception(f"Client {client_id} round failure:")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


__all__ = ["run_client_round", "client_label"]
