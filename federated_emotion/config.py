"""Configuration module for the Federated Emotion Distillation pipeline.

Loads configuration parameters from config.yaml into a structured dataclass
with validation for required fields, type casting, dot-notation access,
and resolution of Hugging Face authentication tokens from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml


REQUIRED_CONFIG_FIELDS: Set[str] = {
    "num_clients",
    "num_rounds",
    "local_epochs",
    "batch_size_train",
    "batch_size_infer",
    "max_seq_length",
    "learning_rate",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "quant_bits",
    "kd_lambda",
    "kd_warmup_rounds",
    "kd_temperature",
    "aggregation_mode",
    "aggregation_temperature",
    "public_kd_pool_size",
    "public_eval_holdout_size",
    "private_dataset_max_size",
    "seed_base",
    "checkpoint_dir",
    "log_dir",
    "hf_token_env_var",
}


@dataclass
class SparseTrainingConfig:
    """Configuration for sparse/reduced-rank training."""

    enabled: bool = False
    reduced_rank: int = 2

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> SparseTrainingConfig:
        """Instantiate SparseTrainingConfig from dictionary."""
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            reduced_rank=int(data.get("reduced_rank", 2)),
        )


@dataclass
class Config:
    """Dataclass holding all hyperparameters and configurations for the pipeline."""

    num_clients: int
    num_rounds: int
    local_epochs: int
    batch_size_train: int
    batch_size_infer: int
    max_seq_length: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    quant_bits: int
    kd_lambda: float
    kd_warmup_rounds: int
    kd_temperature: float
    aggregation_mode: str
    aggregation_temperature: float
    public_kd_pool_size: int
    public_eval_holdout_size: int
    seed_base: int
    checkpoint_dir: str
    log_dir: str
    hf_token_env_var: str
    private_dataset_max_size: int = 2000
    active_client_ids: Optional[List[int]] = None
    num_classes: int = 6
    sparse_training: SparseTrainingConfig = field(default_factory=SparseTrainingConfig)

    @property
    def hf_token(self) -> Optional[str]:
        """Retrieve Hugging Face API token from env var or direct token string."""
        if not self.hf_token_env_var:
            return os.environ.get("HF_TOKEN")
        # If user directly provided a raw token string (e.g. hf_...)
        if self.hf_token_env_var.startswith("hf_"):
            return self.hf_token_env_var
        # Otherwise look up from environment
        return os.environ.get(self.hf_token_env_var) or os.environ.get("HF_TOKEN")

    def get_seed_for_client(self, client_id: int) -> int:
        """Helper to derive reproducible client-specific random seed."""
        return self.seed_base + client_id

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        """Instantiate Config from dictionary with missing-field verification and type casting.

        Args:
            data: Dictionary containing configuration keys and values.

        Returns:
            Config: Populated dataclass instance.

        Raises:
            ValueError: If any required configuration field is missing.
        """
        missing_fields = REQUIRED_CONFIG_FIELDS - set(data.keys())
        if missing_fields:
            raise ValueError(
                f"Missing required configuration field(s) in config.yaml: {sorted(list(missing_fields))}"
            )

        active_clients = None
        if "active_client_ids" in data and data["active_client_ids"] is not None:
            active_clients = [int(x) for x in data["active_client_ids"]]

        sparse_training_raw = data.get("sparse_training")
        if isinstance(sparse_training_raw, dict):
            sparse_training_cfg = SparseTrainingConfig.from_dict(sparse_training_raw)
        elif isinstance(sparse_training_raw, SparseTrainingConfig):
            sparse_training_cfg = sparse_training_raw
        else:
            sparse_training_cfg = SparseTrainingConfig()

        # Field-type casting for safety (e.g. exponential notation strings or int-float mismatches)
        return cls(
            num_clients=int(data["num_clients"]),
            num_rounds=int(data["num_rounds"]),
            local_epochs=int(data["local_epochs"]),
            batch_size_train=int(data["batch_size_train"]),
            batch_size_infer=int(data["batch_size_infer"]),
            max_seq_length=int(data["max_seq_length"]),
            learning_rate=float(data["learning_rate"]),
            lora_rank=int(data["lora_rank"]),
            lora_alpha=int(data["lora_alpha"]),
            lora_dropout=float(data["lora_dropout"]),
            quant_bits=int(data["quant_bits"]),
            kd_lambda=float(data["kd_lambda"]),
            kd_warmup_rounds=int(data["kd_warmup_rounds"]),
            kd_temperature=float(data["kd_temperature"]),
            aggregation_mode=str(data["aggregation_mode"]),
            aggregation_temperature=float(data["aggregation_temperature"]),
            public_kd_pool_size=int(data["public_kd_pool_size"]),
            public_eval_holdout_size=int(data["public_eval_holdout_size"]),
            private_dataset_max_size=int(data.get("private_dataset_max_size", 2000)),
            seed_base=int(data["seed_base"]),
            checkpoint_dir=str(data["checkpoint_dir"]),
            log_dir=str(data["log_dir"]),
            hf_token_env_var=str(data["hf_token_env_var"]),
            active_client_ids=active_clients,
            num_classes=int(data.get("num_classes", 6)),
            sparse_training=sparse_training_cfg,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> Config:
        """Load configuration from a YAML file.

        Args:
            yaml_path: Filepath to YAML configuration file.

        Returns:
            Config: Populated dataclass instance.

        Raises:
            FileNotFoundError: If the YAML configuration file does not exist.
            ValueError: If required fields are missing or YAML is empty/invalid.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML content in {path}; expected key-value mapping.")

        return cls.from_dict(data)


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """Load configuration from specified path or default config.yaml in the module directory.

    Args:
        config_path: Optional path to config.yaml. Defaults to 'config.yaml' located
                     in the same directory as this module.

    Returns:
        Config: Populated and validated Config dataclass.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    return Config.from_yaml(config_path)


if __name__ == "__main__":
    # Self-test when executed directly
    cfg = load_config()
    print("Successfully loaded Config:")
    for field in fields(cfg):
        val = getattr(cfg, field.name)
        print(f"  {field.name}: {val} ({type(val).__name__})")
    print(f"  hf_token: {cfg.hf_token} (from env: {cfg.hf_token_env_var})")
