"""Model wrapper module combining quantized transformer backbones, PEFT LoRA adapters,
mean pooling, and trainable classification heads for federated learning.

This module provides:
1. CLIENT_MODELS registry mapping client IDs 1-10 to heterogeneous 3B LLM backbones.
2. FederatedClassifier nn.Module combining 4-bit NF4 quantized backbone, LoRA adapters,
   mean pooling, and FP32 classification head.
3. get_tokenizer helper handling architecture-specific tokenizers (including OpenELM).
4. save_adapter / load_adapter saving/loading LoRA weights + classification head state.
5. free_model helper for purging model tensors and clearing GPU VRAM between client runs.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_pkg_dir = _current_dir.parent
_parent_dir = _pkg_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)

from federated_emotion.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Heterogeneous Client Model Registry (Client IDs 1-10)
# ---------------------------------------------------------------------------
CLIENT_MODELS: Dict[int, str] = {
    1: "openchat/openchat-3.5-0106",                   # Mistral 7B (Ungated)
    2: "HuggingFaceH4/zephyr-7b-beta",                  # Mistral 7B (Ungated)
    3: "Qwen/Qwen2.5-7B",                              # Qwen 7B
    4: "Qwen/Qwen2.5-7B-Instruct",                     # Qwen 7B
    5: "microsoft/Phi-3.5-mini-instruct",              # Phi 3.8B
    6: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",      # DeepSeek 7B
    7: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",     # DeepSeek 8B
    8: "mistralai/Mistral-Nemo-Base-2407",             # Mistral Nemo 12B (Ungated)
    9: "Qwen/Qwen2.5-14B",                             # Qwen 14B
    10: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",    # DeepSeek 14B
}

DEFAULT_FALLBACK_MODEL: str = "Qwen/Qwen2.5-7B"       # Default for client IDs beyond registry (5-7B tier)


def get_model_for_client(client_id: int) -> str:
    """Return model identifier for client_id (1-indexed or 0-indexed), with default fallback."""
    if client_id in CLIENT_MODELS:
        return CLIENT_MODELS[client_id]
    if (client_id + 1) in CLIENT_MODELS:
        return CLIENT_MODELS[client_id + 1]
    return DEFAULT_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# 2. Tokenizer Resolver
# ---------------------------------------------------------------------------
def get_tokenizer(
    model_id: str,
    config: Optional[Config] = None,
) -> PreTrainedTokenizerBase:
    """Retrieve and configure the appropriate tokenizer for a given model architecture.

    Special case for apple/OpenELM:
    OpenELM models do not ship with a tokenizer repo on Hugging Face; they were designed
    to use the LLaMA tokenizer ("meta-llama/Llama-2-7b-hf" or "huggyllama/llama-7b").

    Args:
        model_id: Hugging Face model repository ID.
        config: Optional Config dataclass instance for HF auth token.

    Returns:
        Configured Hugging Face PreTrainedTokenizerBase instance.
    """
    token = config.hf_token if config else None

    # Handle OpenELM tokenizer special case
    if "openelm" in model_id.lower():
        tokenizer_candidates = [
            "huggyllama/llama-7b",
            "openlm-research/open_llama_3b_v2",
            "meta-llama/Llama-2-7b-hf",
        ]
        tokenizer = None
        for candidate in tokenizer_candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    candidate,
                    token=token,
                    trust_remote_code=True,
                )
                logger.info(
                    f"Loaded fallback LLaMA tokenizer '{candidate}' for OpenELM model '{model_id}'."
                )
                break
            except Exception as e:
                logger.warning(
                    f"Could not load tokenizer '{candidate}' for OpenELM: {e}. Trying next candidate..."
                )
        if tokenizer is None:
            # Fallback to general base tokenizer if LLaMA gated access is not present
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", token=token)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=token,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer for '{model_id}' with token ({e}); retrying without token..."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

    # Ensure padding token is set for sequence batching
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


# ---------------------------------------------------------------------------
# 3. Dynamic Hidden Size Extractor
# ---------------------------------------------------------------------------
def _extract_hidden_size(model_config: Any) -> int:
    """Dynamically determine backbone hidden dimension across diverse architectures."""
    for attr in ["hidden_size", "d_model", "model_dim", "dim", "n_embd"]:
        val = getattr(model_config, attr, None)
        if val is not None and isinstance(val, int) and val > 0:
            return val
    # Fallback default if hidden_size is nested or differently named
    if hasattr(model_config, "text_config"):
        return _extract_hidden_size(model_config.text_config)
    raise ValueError(f"Could not determine hidden_size from model config: {model_config}")


# ---------------------------------------------------------------------------
# 4. FederatedClassifier Model Wrapper
# ---------------------------------------------------------------------------
class FederatedClassifier(nn.Module):
    """Federated sequence classification model combining quantized LLM backbone,
    LoRA adapters, sequence mean-pooling, and a dedicated FP32 classification head.
    """

    def __init__(
        self,
        model_id: str,
        num_labels: int = 6,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize backbone with 4-bit quantization, LoRA adapters, and classification head.

        Args:
            model_id: Hugging Face model repository identifier.
            num_labels: Number of target classification classes (default 6 for canonical emotions).
            config: Global Config dataclass instance containing LoRA and quantization parameters.
        """
        super().__init__()
        self.model_id = model_id
        self.num_labels = num_labels
        self.config = config

        token = config.hf_token if config else None
        lora_r = config.lora_rank if config else 8
        lora_alpha = config.lora_alpha if config else 16
        lora_dropout = config.lora_dropout if config else 0.05
        quant_bits = config.quant_bits if config else 4

        # Check if bitsandbytes is actually available
        has_bnb = False
        try:
            import bitsandbytes  # noqa: F401
            has_bnb = True
        except (ImportError, Exception):
            has_bnb = False

        bnb_config = None
        compute_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )

        if torch.cuda.is_available() and quant_bits == 4 and has_bnb:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as e_bnb:
                logger.warning(f"BitsAndBytesConfig failed ({e_bnb}); falling back without 4-bit quant.")
                bnb_config = None
        elif torch.cuda.is_available() and quant_bits == 8 and has_bnb:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            except Exception:
                bnb_config = None

        logger.info(
            f"Loading backbone '{model_id}' (quantization: {quant_bits if has_bnb else 'None'}-bit, device_map: auto)..."
        )

        # Load transformer backbone
        try:
            raw_backbone = AutoModel.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto" if torch.cuda.is_available() else None,
                dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                token=token,
            )
        except Exception as e:
            logger.warning(
                f"AutoModel.from_pretrained failed for '{model_id}' ({e}); attempting fallback without quantization..."
            )
            raw_backbone = AutoModel.from_pretrained(
                model_id,
                device_map="auto" if torch.cuda.is_available() else None,
                dtype=compute_dtype if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                token=token,
            )

        # Dynamic hidden size resolution
        self.hidden_size = _extract_hidden_size(raw_backbone.config)

        # Wrap with PEFT LoRA adapter
        lora_config_all = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules="all-linear",
        )

        try:
            self.backbone = get_peft_model(raw_backbone, lora_config_all)
            logger.info(f"Successfully applied LoRA to all linear layers of '{model_id}'.")
        except Exception as e_lora:
            logger.warning(
                f"Target modules 'all-linear' failed for '{model_id}' ({e_lora}). Falling back to explicit projection targets..."
            )
            fallback_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "dense",
                "fc1",
                "fc2",
                "W_pack",
                "out_proj",
            ]
            lora_config_fallback = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=fallback_target_modules,
            )
            self.backbone = get_peft_model(raw_backbone, lora_config_fallback)

        # Determine backbone device
        try:
            device = next(raw_backbone.parameters()).device
        except StopIteration:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Classification head: kept in FP32 and fully trainable on target device
        self.head = nn.Linear(self.hidden_size, self.num_labels, dtype=torch.float32, device=device)

        # Ensure head parameters are explicitly marked trainable
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through backbone, LoRA adapters, attention mean-pooling, and head.

        Args:
            input_ids: Input token ID tensor of shape [batch_size, seq_len].
            attention_mask: Attention mask tensor of shape [batch_size, seq_len].

        Returns:
            Logits tensor of shape [batch_size, num_labels] with dtype float32.
        """
        # Forward through PEFT LoRA backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        # Extract last hidden state: [batch_size, seq_len, hidden_size]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            last_hidden_state = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            last_hidden_state = outputs[0]

        # Mean-pool across sequence tokens using attention mask
        if attention_mask is not None:
            # Mask shape: [batch_size, seq_len, 1]
            mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand_as(last_hidden_state)
                .to(last_hidden_state.dtype)
            )
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = last_hidden_state.mean(dim=1)

        # Move pooled representation to classification head device and cast to float32
        head_device = self.head.weight.device
        pooled_fp32 = pooled.to(device=head_device, dtype=torch.float32)

        # Pass through classification head
        logits = self.head(pooled_fp32)
        return logits


# ---------------------------------------------------------------------------
# 5. Checkpointing & Persistence (Adapter + Head)
# ---------------------------------------------------------------------------
def save_adapter(
    model: FederatedClassifier,
    path: Union[str, Path],
) -> None:
    """Save both the PEFT LoRA adapter weights and the classification head.

    Args:
        model: FederatedClassifier instance.
        path: Target directory path to store adapter and head checkpoints.
    """
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save PEFT LoRA adapter weights and config
    if hasattr(model, "backbone") and isinstance(model.backbone, PeftModel):
        model.backbone.save_pretrained(str(save_dir))
    elif hasattr(model, "backbone"):
        model.backbone.save_pretrained(str(save_dir))

    # 2. Save classification head state dict separately
    head_path = save_dir / "head.pt"
    torch.save(model.head.state_dict(), head_path)
    logger.info(f"Saved LoRA adapter and head checkpoint to {save_dir}.")


def load_adapter(
    model: FederatedClassifier,
    path: Union[str, Path],
) -> None:
    """Load previously saved PEFT LoRA adapter weights and classification head.

    Args:
        model: FederatedClassifier instance to receive loaded weights.
        path: Source directory containing adapter files and head.pt.
    """
    load_dir = Path(path)
    if not load_dir.exists():
        raise FileNotFoundError(f"Adapter checkpoint directory does not exist: {load_dir}")

    # 1. Load LoRA adapter
    if hasattr(model, "backbone") and isinstance(model.backbone, PeftModel):
        model.backbone.load_adapter(str(load_dir), adapter_name="default")
    else:
        logger.warning(f"Backbone of model is not a PeftModel; skipping load_adapter for LoRA.")

    # 2. Load classification head
    head_path = load_dir / "head.pt"
    if head_path.exists():
        head_state = torch.load(head_path, map_location=self_or_cpu_device(model), weights_only=True)
        model.head.load_state_dict(head_state)
        logger.info(f"Loaded classification head weights from {head_path}.")
    else:
        logger.warning(f"Classification head checkpoint not found at {head_path}.")


def self_or_cpu_device(model: nn.Module) -> torch.device:
    """Helper to get current device of a module or fallback to CPU."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# 6. Memory Management Helper
# ---------------------------------------------------------------------------
def free_model(model: Optional[Any]) -> None:
    """Delete model reference, execute garbage collection, and clear CUDA VRAM caches.

    This should be invoked at the end of each federated client step to ensure
    memory isolation and prevent OOM errors during sequential multi-client rounds.

    Args:
        model: Model or object reference to free.
    """
    if model is not None:
        # Skip .cpu() for quantized models — it triggers dequantization into RAM
        # which can OOM on memory-constrained machines. Just del + GC is sufficient.
        del model

    # Trigger garbage collection
    gc.collect()

    # Clear CUDA memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# 7. Package Exports
# ---------------------------------------------------------------------------
__all__ = [
    "CLIENT_MODELS",
    "FederatedClassifier",
    "get_tokenizer",
    "save_adapter",
    "load_adapter",
    "free_model",
]
