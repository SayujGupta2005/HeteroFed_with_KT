"""Stand-alone model verification and pre-fetching utility.

Executes a pre-flight audit of all configured client LLM backbones:
1. Validates Hugging Face repository accessibility (zero 401/403/404 errors).
2. Downloads and caches tokenizers and 4-bit quantized base model weights.
3. Tests PEFT LoRA adapter injection and classification head initialization.
4. Ensures clean VRAM release between checks.

Usage:
    python federated_emotion/models/verify_models.py [--config federated_emotion/config.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_current_dir = Path(__file__).resolve().parent
_pkg_dir = _current_dir.parent
_parent_dir = _pkg_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

from federated_emotion.config import load_config
from federated_emotion.models.wrapper import (
    CLIENT_MODELS,
    preload_client_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-flight verification and pre-caching for federated client models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    all_client_ids = list(range(1, config.num_clients + 1))
    preload_client_models(all_client_ids, config)


if __name__ == "__main__":
    main()
