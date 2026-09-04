"""Federated Client module for local fine-tuning and Knowledge Distillation inference.

This module provides:
1. run_client_round: Executes a single communication round for a federated client:
   - Initializes / restores quantized LLM backbone with PEFT LoRA adapter.
   - Performs local supervised fine-tuning with cross-entropy loss.
   - Computes Knowledge Distillation (KD) loss against server consensus soft labels.
   - Evaluates client performance on public evaluation holdout.
   - Generates and returns student logits over public KD pool.
   - Persists client checkpoints and reclaims GPU VRAM.
2. Error resilience: Catches and logs runtime errors, freeing VRAM and returning None
   to keep the central orchestration pipeline running smoothly.
"""

from __future__ import annotations

import itertools
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from federated_emotion.profiler import global_profiler
from federated_emotion.config import Config
from federated_emotion.data.loaders import CLIENT_DATASETS, NUM_CLASSES
from federated_emotion.models.wrapper import (
    CLIENT_MODELS,
    FederatedClassifier,
    free_model,
    get_model_for_client,
    get_tokenizer,
    load_adapter,
    save_adapter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Custom Collation Helpers
# ---------------------------------------------------------------------------
def _create_collate_fn(tokenizer: Any, max_length: int, include_idx: bool = False):
    """Factory creating dynamic tokenization and batch collation functions."""
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [item["text"] for item in batch]
        encoded = tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        result: Dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "label" in batch[0]:
            labels = [int(item["label"]) for item in batch]
            result["label"] = torch.tensor(labels, dtype=torch.long)

        if include_idx and "idx" in batch[0]:
            indices = [int(item["idx"]) for item in batch]
            result["idx"] = torch.tensor(indices, dtype=torch.long)

        return result

    return collate_fn


# ---------------------------------------------------------------------------
# 2. Main Client Execution Function
# ---------------------------------------------------------------------------
def run_client_round(
    client_id: int,
    round_num: int,
    config: Config,
    public_kd_pool: Dataset,
    public_eval_holdout: Dataset,
    private_dataset: Dataset,
    avg_soft_labels: Optional[np.ndarray] = None,
    avg_soft_labels_from_server: Optional[np.ndarray] = None,
    cross_eval_datasets: Optional[Dict[str, Dataset]] = None,
) -> Optional[Dict[str, Any]]:
    """Execute a single local training and distillation round for an active client.

    Args:
        client_id: Client identifier (1-10 or 0-9).
        round_num: Current communication round index (1..num_rounds).
        config: Global pipeline Config dataclass instance.
        public_kd_pool: Public transfer dataset for logit distillation averaging.
        public_eval_holdout: Separate held-out Hugging Face Dataset for client evaluation.
        private_dataset: Local private Hugging Face Dataset with ["text", "label"].
        avg_soft_labels: Consensus soft labels from server if KD active.
        avg_soft_labels_from_server: Alias for avg_soft_labels.
        cross_eval_datasets: Optional dictionary mapping dataset label names to holdout datasets.

    Returns:
        Dictionary containing client metrics, logits, eval accuracy, and cross-dataset matrix entries.
    """
    if avg_soft_labels is None and avg_soft_labels_from_server is not None:
        avg_soft_labels = avg_soft_labels_from_server

    model: Optional[FederatedClassifier] = None

    try:
        # 1. Resolve model ID and dataset metadata
        model_id = get_model_for_client(client_id)
        reg_id = client_id if client_id in CLIENT_DATASETS else client_id + 1
        dataset_info = CLIENT_DATASETS.get(reg_id, ("Custom / Private", None))
        dataset_name = f"{dataset_info[0]}" + (f" ({dataset_info[1]})" if dataset_info[1] else "")

        print("\n" + "=" * 80)
        print(
            f"[Client {client_id:02d}] Communication Round {round_num}/{config.num_rounds}"
        )
        print(f"  Model Architecture : {model_id}")
        print(f"  Private Dataset    : {dataset_name} ({len(private_dataset)} examples)")
        print(f"  KD Pool Size       : {len(public_kd_pool)} | Eval Holdout: {len(public_eval_holdout)}")
        print("=" * 80)

        # 2. Tokenizer initialization
        tokenizer = get_tokenizer(model_id, config=config)

        # 3. Model initialization & checkpoint restoration
        model = FederatedClassifier(model_id=model_id, num_labels=NUM_CLASSES, config=config)

        if round_num > 1:
            # Check for prior round checkpoint
            prev_round = round_num - 1
            prev_checkpoint_dir = Path(config.checkpoint_dir) / f"client_{client_id}" / f"round_{prev_round}"
            if prev_checkpoint_dir.exists():
                print(f"  -> Restoring client checkpoint from: {prev_checkpoint_dir}")
                load_adapter(model, prev_checkpoint_dir)
            else:
                logger.warning(
                    f"Prior round checkpoint not found at {prev_checkpoint_dir}; starting from base model."
                )

        # Device determination
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.head.to(device)

        # 4. Prepare DataLoaders
        collate_private = _create_collate_fn(tokenizer, config.max_seq_length, include_idx=False)
        train_loader = DataLoader(
            private_dataset,
            batch_size=config.batch_size_train,
            shuffle=True,
            collate_fn=collate_private,
        )

        # Add row index to public_kd_pool for aligned KD label lookup
        indexed_kd_pool = public_kd_pool.map(
            lambda ex, i: {"idx": i},
            with_indices=True,
            desc="Indexing KD pool",
        )
        collate_kd = _create_collate_fn(tokenizer, config.max_seq_length, include_idx=True)
        kd_loader = DataLoader(
            indexed_kd_pool,
            batch_size=config.batch_size_train,
            shuffle=True,
            collate_fn=collate_kd,
        )

        # 5. Prepare Optimizer over trainable parameters (LoRA + Head)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError(f"No trainable parameters found in model for client {client_id}.")

        use_8bit = getattr(config, "optimizer_8bit", True) and HAS_BNB
        if use_8bit:
            try:
                optimizer = bnb.optim.AdamW8bit(trainable_params, lr=config.learning_rate)
                print(f"  -> Optimizer: bitsandbytes 8-bit AdamW")
            except Exception as e_bnb:
                logger.warning(f"Could not initialize 8-bit AdamW ({e_bnb}); falling back to torch.optim.AdamW")
                optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
                print(f"  -> Optimizer: Standard AdamW (fallback due to {e_bnb})")
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
            print("  -> Optimizer: Standard AdamW" + (" (bitsandbytes not found)" if not HAS_BNB else ""))

        ce_loss_fn = nn.CrossEntropyLoss()
        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

        # Knowledge Distillation state
        is_kd_active = (
            round_num > config.kd_warmup_rounds
            and avg_soft_labels_from_server is not None
        )

        teacher_soft_tensor: Optional[torch.Tensor] = None
        if is_kd_active:
            if isinstance(avg_soft_labels_from_server, np.ndarray):
                teacher_soft_tensor = torch.tensor(
                    avg_soft_labels_from_server, dtype=torch.float32, device=device
                )
            else:
                teacher_soft_tensor = avg_soft_labels_from_server.to(device=device, dtype=torch.float32)

            print(
                f"  -> Knowledge Distillation ACTIVE (lambda={config.kd_lambda}, "
                f"temperature={config.kd_temperature})"
            )
        else:
            print("  -> Knowledge Distillation INACTIVE (Warmup phase or round 1)")

        # 6. Local Training Epochs
        global_profiler.start("Client_Local_Training")
        model.train()
        for epoch in range(1, config.local_epochs + 1):
            total_ce_loss = 0.0
            total_kd_loss = 0.0
            total_combined_loss = 0.0
            num_steps = 0

            # Interleave private batches with KD batches
            kd_iter = itertools.cycle(kd_loader) if is_kd_active else None

            pbar = tqdm(train_loader, desc=f"  Epoch [{epoch:02d}/{config.local_epochs:02d}]", leave=False)
            for batch_priv in pbar:
                optimizer.zero_grad()

                # A. Supervised Task Loss
                input_ids = batch_priv["input_ids"].to(device)
                attention_mask = batch_priv["attention_mask"].to(device)
                labels = batch_priv["label"].to(device)

                logits_priv = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_ce = ce_loss_fn(logits_priv, labels)
                loss = loss_ce

                # B. Knowledge Distillation Loss
                loss_kd = torch.tensor(0.0, device=device)
                if is_kd_active and kd_iter is not None and teacher_soft_tensor is not None:
                    batch_kd = next(kd_iter)
                    kd_input_ids = batch_kd["input_ids"].to(device)
                    kd_attention_mask = batch_kd["attention_mask"].to(device)
                    kd_indices = batch_kd["idx"].to(device)

                    logits_kd = model(input_ids=kd_input_ids, attention_mask=kd_attention_mask)
                    # Student scaled log probabilities
                    student_log_probs = F.log_softmax(logits_kd / config.kd_temperature, dim=-1)
                    # Target teacher soft distribution
                    target_teacher = teacher_soft_tensor[kd_indices]

                    loss_kd = kl_loss_fn(student_log_probs, target_teacher) * (config.kd_temperature ** 2)
                    loss = loss + (config.kd_lambda * loss_kd)

                loss.backward()
                optimizer.step()

                total_ce_loss += loss_ce.item()
                total_kd_loss += loss_kd.item() if is_kd_active else 0.0
                total_combined_loss += loss.item()
                num_steps += 1

            avg_ce = total_ce_loss / max(num_steps, 1)
            avg_kd = total_kd_loss / max(num_steps, 1)
            avg_total = total_combined_loss / max(num_steps, 1)

            print(
                f"  Epoch [{epoch:02d}/{config.local_epochs:02d}] "
                f"Loss Total: {avg_total:.4f} | CE: {avg_ce:.4f}"
                + (f" | KD: {avg_kd:.4f}" if is_kd_active else "")
            )
        global_profiler.stop("Client_Local_Training")

        # 7. Post-Training Inference on Public KD Pool (no_grad)
        global_profiler.start("Client_KD_Inference")
        model.eval()
        print("  -> Generating client logits over public KD pool...")
        infer_collate = _create_collate_fn(tokenizer, config.max_seq_length, include_idx=False)
        kd_infer_loader = DataLoader(
            public_kd_pool,
            batch_size=config.batch_size_infer,
            shuffle=False,
            collate_fn=infer_collate,
        )

        kd_logits_list: List[np.ndarray] = []
        with torch.no_grad():
            for batch_kd_infer in kd_infer_loader:
                input_ids = batch_kd_infer["input_ids"].to(device)
                attention_mask = batch_kd_infer["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                kd_logits_list.append(logits.detach().cpu().to(torch.float32).numpy())

        logits_on_kd_pool: np.ndarray = np.concatenate(kd_logits_list, axis=0)
        global_profiler.stop("Client_KD_Inference")

        # 8. Client Evaluation on Public Eval Holdout
        global_profiler.start("Client_Local_Eval")
        print("  -> Evaluating on public eval holdout...")
        eval_collate = _create_collate_fn(tokenizer, config.max_seq_length, include_idx=False)
        eval_loader = DataLoader(
            public_eval_holdout,
            batch_size=config.batch_size_infer,
            shuffle=False,
            collate_fn=eval_collate,
        )

        total_correct = 0
        total_eval_samples = 0
        with torch.no_grad():
            for batch_eval in eval_loader:
                input_ids = batch_eval["input_ids"].to(device)
                attention_mask = batch_eval["attention_mask"].to(device)
                labels = batch_eval["label"].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_eval_samples += labels.size(0)

        eval_accuracy = float(total_correct / max(total_eval_samples, 1))
        print(
            f"  [Client {client_id:02d}] Final Eval Accuracy: {eval_accuracy * 100:.2f}% "
            f"({total_correct}/{total_eval_samples})"
        )
        global_profiler.stop("Client_Local_Eval")

        # 9. Cross-Dataset Evaluation across all client dataset holdouts
        cross_eval_accuracies: Dict[str, float] = {}
        if cross_eval_datasets:
            global_profiler.start("Client_Cross_Eval")
            with torch.no_grad():
                for ds_key, ds_slice in cross_eval_datasets.items():
                    if len(ds_slice) == 0:
                        continue
                    cross_loader = DataLoader(
                        ds_slice,
                        batch_size=config.batch_size_infer,
                        shuffle=False,
                        collate_fn=eval_collate,
                    )
                    c_corr = 0
                    c_tot = 0
                    for c_batch in cross_loader:
                        c_in = c_batch["input_ids"].to(device)
                        c_mask = c_batch["attention_mask"].to(device)
                        c_lab = c_batch["label"].to(device)
                        c_out = model(input_ids=c_in, attention_mask=c_mask)
                        c_preds = torch.argmax(c_out, dim=-1)
                        c_corr += (c_preds == c_lab).sum().item()
                        c_tot += c_lab.size(0)
                    cross_eval_accuracies[ds_key] = float(c_corr / max(c_tot, 1))
            global_profiler.stop("Client_Cross_Eval")

        # 10. Save Checkpoint (Adapter + Head)
        checkpoint_path = (
            Path(config.checkpoint_dir) / f"client_{client_id}" / f"round_{round_num}"
        )
        save_adapter(model, checkpoint_path)
        print(f"  -> Checkpoint saved to: {checkpoint_path}")

        # 11. Free Model VRAM
        free_model(model)
        model = None

        return {
            "client_id": client_id,
            "logits_on_kd_pool": logits_on_kd_pool,
            "eval_accuracy": eval_accuracy,
            "cross_eval_accuracies": cross_eval_accuracies,
            # --- Comprehensive metrics for detailed CSV output ---
            "model_id": model_id,
            "dataset_name": dataset_name,
            "num_private_examples": len(private_dataset),
            "num_kd_pool": len(public_kd_pool),
            "num_eval_holdout": len(public_eval_holdout),
            "num_train_steps": num_steps,
            "local_epochs": config.local_epochs,
            "avg_ce_loss": avg_ce,
            "avg_kd_loss": avg_kd if is_kd_active else 0.0,
            "avg_total_loss": avg_total,
            "kd_active": is_kd_active,
            "eval_correct": total_correct,
            "eval_total": total_eval_samples,
        }

    except Exception as e:
        print(
            f"\n[WARNING] Client {client_id} encountered an error during Round {round_num}: {e}"
        )
        logger.exception(f"Client {client_id} round failure:")
        if model is not None:
            free_model(model)
        return None


# ---------------------------------------------------------------------------
# 3. Package Exports
# ---------------------------------------------------------------------------
__all__ = [
    "run_client_round",
]
