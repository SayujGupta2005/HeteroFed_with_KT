# Heterogeneous Federated Knowledge Distillation for Emotion Classification

A federated learning pipeline where **10 heterogeneous 3B-parameter LLMs** collaboratively learn emotion classification **without sharing model weights or private data**. Instead of traditional parameter averaging (FedAvg), clients communicate via soft label distributions over a shared public dataset — enabling cross-architecture knowledge transfer through **Federated Knowledge Distillation (Fed-KD)**.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Central Aggregation Server                    │
│  • Collects client logits over public KD pool                   │
│  • Accuracy-weighted softmax consensus: Σ wₖ · pₖ(x)           │
│  • Broadcasts consensus soft labels to all clients              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Consensus Probabilities
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Client 1 │     │ Client 2 │     │ Client K │
   │ Qwen-3B  │     │ LLaMA-3B │ ... │ OpenELM  │
   │ go_emot. │     │tweet_eval│     │poem_sent.│
   └──────────┘     └──────────┘     └──────────┘
   Loss = L_CE + λ · T² · KL(student ‖ teacher_consensus)
```

### Why Knowledge Distillation instead of FedAvg?

In standard FedAvg, the server averages model parameters: `θ_global = Σ wₖ · θₖ`. This **requires all clients to share the exact same architecture**. Our clients have completely different backbones (Qwen, LLaMA, Phi, StableLM, OpenELM) with different hidden dimensions, vocabularies, and parameter counts — parameter averaging is mathematically impossible.

Fed-KD solves this by communicating **output probability distributions** instead of weights. Each client runs inference on a shared public dataset and sends its logits to the server. The server aggregates these into a consensus teacher distribution, which guides all clients via KL divergence loss.

## Client Registry

| Client | Model Backbone | Private Dataset | HF Config |
|--------|---------------|-----------------|-----------|
| 1 | `Qwen/Qwen2.5-3B` | go_emotions | simplified |
| 2 | `Qwen/Qwen2.5-3B-Instruct` | tweet_eval | emotion |
| 3 | `meta-llama/Llama-3.2-3B` | sem_eval_2018_task1 | subtask5.english |
| 4 | `meta-llama/Llama-3.2-3B-Instruct` | silicone | dyda_e |
| 5 | `microsoft/Phi-3.5-mini-instruct` | silicone | meld_e |
| 6 | `stabilityai/stablelm-3b-4e1t` | silicone | iemocap |
| 7 | `openlm-research/open_llama_3b_v2` | empathetic_dialogues | — |
| 8 | `togethercomputer/RedPajama-INCITE-3B-Base` | emo | — |
| 9 | `apple/OpenELM-3B` | xed_en_fi | en_annotated |
| 10 | `Qwen/Qwen2.5-3B` | poem_sentiment | — |

All datasets are harmonized to 6 canonical emotion classes: `sadness(0)`, `joy(1)`, `love(2)`, `anger(3)`, `fear(4)`, `surprise(5)`.

## Setup Instructions

### Prerequisites

- **Python 3.12+**
- **NVIDIA GPU** with CUDA support (minimum 4GB VRAM with 4-bit quantization)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/Vikhyat-Chauhan/HeteroFed_with_KT.git
cd HeteroFed_with_KT
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r federated_emotion/requirements.txt
```

**Critical:** `bitsandbytes` is required for 4-bit NF4 quantization. Without it, 3B models will attempt to load in full precision (~6GB per model) and will OOM on most consumer GPUs.

Verify bitsandbytes is working:
```bash
python -c "import bitsandbytes; print('bnb version:', bitsandbytes.__version__)"
```

### 4. Set Up Hugging Face Authentication

Some models (e.g., `meta-llama/Llama-3.2-3B`) are gated and require a Hugging Face token:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model license at [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)
3. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

**Option A — Environment variable:**
```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"

# Linux/macOS
export HF_TOKEN="hf_your_token_here"
```

**Option B — Direct in config:**
Set `hf_token_env_var` in your YAML config to your token string directly:
```yaml
hf_token_env_var: "hf_your_token_here"
```

## Running the Pipeline

### Smoke Test (Recommended First Run)

Start with the smoke test configuration (1 round, 2 clients, tiny dataset slices) to verify everything works:

```bash
python federated_emotion/main.py --config federated_emotion/config_smoke.yaml
```

> **Note:** The first run will download model weights from Hugging Face Hub (~6GB for Qwen2.5-3B). This may take several minutes depending on your internet connection.

### Full Training Run

```bash
python federated_emotion/main.py --config federated_emotion/config.yaml
```

### Command-Line Overrides

```bash
python federated_emotion/main.py \
  --config federated_emotion/config.yaml \
  --num_rounds 5 \
  --local_epochs 3 \
  --num_clients 10
```

### Resume from Checkpoints

If a run is interrupted, resume without re-training completed rounds:

```bash
python federated_emotion/main.py --config federated_emotion/config.yaml --resume
```

## Configuration

All hyperparameters are in `federated_emotion/config.yaml`:

```yaml
# Core FL Settings
num_clients: 10              # Total federated clients
num_rounds: 3                # Communication rounds
local_epochs: 2              # Local training epochs per client per round
active_client_ids: [1, 2]    # (Optional) Subset of clients to run

# Model & Training
batch_size_train: 8
batch_size_infer: 16
max_seq_length: 128
learning_rate: 2e-4
lora_rank: 8                 # LoRA adapter rank
lora_alpha: 16               # LoRA scaling factor
lora_dropout: 0.05
quant_bits: 4                # 4-bit NF4 quantization

# Knowledge Distillation
kd_lambda: 0.5               # Weight of KD loss in composite objective
kd_warmup_rounds: 1          # Rounds before KD activates (0 = from round 1)
kd_temperature: 2.0          # Softmax temperature for logit softening

# Aggregation
aggregation_mode: "accuracy_weighted"  # or "uniform"
aggregation_temperature: 1.0

# Data
public_kd_pool_size: 500     # Public KD pool subset size
public_eval_holdout_size: 200
private_dataset_max_size: 2000

# Paths
seed_base: 42
checkpoint_dir: "./checkpoints"
log_dir: "./logs"
hf_token_env_var: "HF_TOKEN" # Env var name or direct token
```

## Output Files

After a run completes, the following files are generated:

| File | Description |
|------|-------------|
| `logs/summary.csv` | Accuracy comparison matrix: rows = clients, columns = rounds, with net gain |
| `logs/detailed_metrics.csv` | **Comprehensive CSV**: model, dataset, CE/KD/total loss, KD active flag, training steps, eval accuracy, correct/total counts per client per round |
| `logs/round_metrics.jsonl` | Raw per-client per-round metrics in JSON Lines format |
| `logs/metrics_history.json` | Aggregated per-round metrics history |
| `logs/round_N_avg_soft_labels.npy` | Consensus teacher soft label distributions per round |
| `checkpoints/client_N/round_M/` | LoRA adapter weights + classification head + cached logits per client per round |

### Sample `detailed_metrics.csv` Columns

```
round, client_id, model_id, dataset_name, num_private_examples,
local_epochs, num_train_steps, avg_ce_loss, avg_kd_loss, avg_total_loss,
kd_active, eval_accuracy, eval_accuracy_pct, eval_correct, eval_total,
num_kd_pool, num_eval_holdout
```

## Project Structure

```
HeteroFed_with_KT/
├── federated_emotion/
│   ├── __init__.py              # Package init
│   ├── config.py                # Config dataclass with YAML loading & validation
│   ├── config.yaml              # Full training configuration
│   ├── config_smoke.yaml        # Smoke test configuration (1 round, 2 clients)
│   ├── main.py                  # Orchestration loop & CLI entrypoint
│   ├── client.py                # Client local training, KD loss, eval, logit extraction
│   ├── server.py                # Server aggregation (uniform / accuracy-weighted)
│   ├── eval_utils.py            # Metrics, summary tables, CSV export
│   ├── requirements.txt         # Python dependencies
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py           # Dataset loading, label harmonization, partitioning
│   └── models/
│       ├── __init__.py
│       └── wrapper.py           # FederatedClassifier, LoRA, quantization, checkpointing
├── pipeline_comprehensive_architecture_report.md  # Detailed architecture & math report
├── .gitignore
└── README.md
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: bitsandbytes` | Not installed | `pip install bitsandbytes` |
| `quantization: None-bit` in logs | bitsandbytes import failed | Reinstall: `pip install bitsandbytes --force-reinstall` |
| `CUDA out of memory` | Model too large for VRAM | Ensure `quant_bits: 4` in config; reduce `batch_size_train` |
| `401 Unauthorized` for Llama models | Missing HF token or license | Set `HF_TOKEN` env var; accept license on HuggingFace |
| Download stuck at 0% | Stale `.incomplete` files in HF cache | Delete `~/.cache/huggingface/hub/models--*/blobs/*.incomplete` |
| `ReadTimeoutError` during download | Slow connection | Retry; or set `HF_HUB_DOWNLOAD_TIMEOUT=60` env var |

## Hardware Requirements

| Config | Min VRAM | Notes |
|--------|----------|-------|
| 4-bit quantization (default) | **4 GB** | RTX 3050/3060/4060 and above |
| 8-bit quantization | **8 GB** | RTX 3070/4070 and above |
| Full precision (no quantization) | **12+ GB** | Not recommended for consumer GPUs |

Models are loaded and freed sequentially (one client at a time), so VRAM requirement is for a single 3B model, not all 10 simultaneously.

## License

This project is for academic/research purposes as part of a B.Tech Project (BTP).
