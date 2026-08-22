# Heterogeneous Federated Knowledge Distillation for Emotion Classification

A federated learning pipeline where **10 heterogeneous LLMs (4B–14B parameters)** collaboratively learn emotion classification **without sharing model weights or private data**. Instead of traditional parameter averaging (FedAvg), clients communicate via soft label distributions over a shared public dataset — enabling cross-architecture knowledge transfer through **Federated Knowledge Distillation (Fed-KD)**.

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
   │OpenChat7B│     │Zephyr 7B │ ... │ DSR1-14B │
   │ go_emot. │     │tweet_eval│     │poem_sent.│
   └──────────┘     └──────────┘     └──────────┘
   Loss = L_CE + λ · T² · KL(student ‖ teacher_consensus)
```

### Why Knowledge Distillation instead of FedAvg?

In standard FedAvg, the server averages model parameters: `θ_global = Σ wₖ · θₖ`. This **requires all clients to share the exact same architecture**. Our clients have completely different backbones (Mistral, Qwen, Phi, DeepSeek) with different hidden dimensions, vocabularies, and parameter counts — parameter averaging is mathematically impossible.

Fed-KD solves this by communicating **output probability distributions** instead of weights. Each client runs inference on a shared public dataset and sends its logits to the server. The server aggregates these into a consensus teacher distribution, which guides all clients via KL divergence loss.

## Client Registry

The pipeline uses a diverse set of **ungated, open-weight** models ranging from **3.8B to 14B parameters**, ensuring true architectural heterogeneity and accessibility without requiring gated model access.

| Client | Model Backbone | Size | Private Dataset | HF Config |
|--------|---------------|------|-----------------|-----------|
| 1 | `openchat/openchat-3.5-0106` | 7B | go_emotions | simplified |
| 2 | `HuggingFaceH4/zephyr-7b-beta` | 7B | tweet_eval | emotion |
| 3 | `Qwen/Qwen2.5-7B` | 7B | sem_eval_2018_task1 | subtask5.english |
| 4 | `Qwen/Qwen2.5-7B-Instruct` | 7B | silicone | dyda_e |
| 5 | `microsoft/Phi-3.5-mini-instruct` | 3.8B | silicone | meld_e |
| 6 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | silicone | iemocap |
| 7 | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | 8B | empathetic_dialogues | — |
| 8 | `mistralai/Mistral-Nemo-Base-2407` | 12B | emo | — |
| 9 | `Qwen/Qwen2.5-14B` | 14B | xed_en_fi | en_annotated |
| 10 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14B | poem_sentiment | — |

> **Note:** All models are ungated and do not require a Hugging Face license agreement. The default fallback model for client IDs beyond the registry is `Qwen/Qwen2.5-7B`.

### Model Families Represented

- **Mistral**: OpenChat 3.5, Zephyr 7B, Mistral Nemo 12B
- **Qwen**: Qwen2.5 7B, Qwen2.5 7B-Instruct, Qwen2.5 14B
- **Phi**: Phi-3.5-mini-instruct (3.8B)
- **DeepSeek R1 Distill**: Qwen-7B, Llama-8B, Qwen-14B

### Dataset Harmonization

All 10 private datasets are harmonized to **6 canonical emotion classes**: `sadness(0)`, `joy(1)`, `love(2)`, `anger(3)`, `fear(4)`, `surprise(5)`.

## Key Features

### Cross-Dataset Performance Matrix

After **every communication round**, the pipeline evaluates each client model against **all** client private holdout datasets and the public evaluation holdout. This produces a `Model × Dataset` performance matrix that enables tracking of:

- How well each model generalizes across datasets it was **not** trained on
- Whether federated distillation improves cross-dataset transfer over rounds
- Comparative performance of smaller vs. larger models

The matrix is printed to the console and exported as a CSV file to both `logs/` and `results/run_<timestamp>/`.

### Model Pre-Loading Phase

Before training begins, the pipeline runs a dedicated **pre-loading and verification phase** (`[3/4]`) that:

1. Downloads and caches all model checkpoints and tokenizers from Hugging Face Hub
2. Initializes each backbone to verify architecture compatibility (quantization, LoRA targets)
3. Immediately frees VRAM after verification to preserve memory for training

This ensures training logs are clean (no mid-round download progress bars) and any issues (authentication, architecture, network) are caught upfront.

### Timestamped Run Output

Each run creates a dedicated output directory (`results/run_YYYYMMDD_HHMMSS/`) containing:

- **`run_info.txt`**: Full experiment manifest with hardware info, hyperparameters, and client-model-dataset assignments
- **`summary.csv`**: Final accuracy matrix across rounds
- **`round_metrics.jsonl`**: Per-client, per-round detailed metrics
- **`round_N_avg_soft_labels.npy`**: Consensus teacher distributions per round
- **Cross-dataset matrix CSVs**: Performance matrices per round

## Setup Instructions

### Prerequisites

- **Python 3.12+**
- **NVIDIA GPU** with CUDA support (minimum 8GB VRAM recommended with 4-bit quantization for 7B+ models)
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

**Critical:** `bitsandbytes` is required for 4-bit NF4 quantization. Without it, models will attempt to load in full precision and will OOM on most consumer GPUs.

Verify bitsandbytes is working:
```bash
python -c "import bitsandbytes; print('bnb version:', bitsandbytes.__version__)"
```

### 4. Set Up Hugging Face Authentication (Optional)

All default models are ungated and do not require authentication. However, if you customize the client registry to use gated models, you will need a Hugging Face token:

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

### Pre-Flight Verification

Before running the full pipeline, you can independently verify datasets and models:

```bash
# Verify all 10 private datasets load correctly
python federated_emotion/data/verify_datasets.py

# Verify all model backbones download and initialize
python federated_emotion/models/verify_models.py
```

### Smoke Test (Recommended First Run)

Start with the smoke test configuration (1 round, 2 clients, tiny dataset slices) to verify everything works:

```bash
python federated_emotion/main.py --config federated_emotion/config_smoke.yaml
```

> **Note:** The first run will download model weights from Hugging Face Hub. Download size varies per model (2–8GB each). This may take significant time depending on your internet connection.

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

# Classification
num_classes: 6               # Number of canonical emotion classes (default: 6)

# Paths
seed_base: 42
checkpoint_dir: "./checkpoints"
log_dir: "./logs"
hf_token_env_var: "HF_TOKEN" # Env var name or direct token
```

### Customizing the Number of Clients

The `num_clients` parameter controls how many clients participate. When `num_clients` exceeds the 10-entry model registry, additional clients automatically receive the default fallback model (`Qwen/Qwen2.5-7B`). You can also specify a subset of clients using `active_client_ids`.

## Pipeline Phases

The main pipeline executes in 4 phases:

```
[1/4] Loading Public KD and Evaluation Datasets
      → 500 public KD pool + 200 evaluation holdout from dair-ai/emotion

[2/4] Pre-loading Client Private Datasets & Slicing Holdouts
      → 10 heterogeneous emotion datasets, each capped at 2000 examples
      → Private holdouts carved for cross-dataset evaluation matrix

[3/4] Pre-loading & Verifying Backbone Models
      → Downloads, initializes, and verifies all unique model architectures
      → Frees VRAM after verification to preserve memory for training

[4/4] Federated Communication Rounds
      → Per round: local training → inference → aggregation → KD broadcast
      → Cross-dataset performance matrix computed after each round
```

## Output Files

After a run completes, the following files are generated in `results/run_<timestamp>/`:

| File | Description |
|------|-------------|
| `run_info.txt` | Full experiment manifest: hardware, hyperparameters, client-model-dataset assignments |
| `summary.csv` | Accuracy comparison matrix: rows = clients, columns = rounds, with net gain |
| `round_metrics.jsonl` | Per-client per-round metrics in JSON Lines format (loss, accuracy, KD status) |
| `round_N_cross_dataset_matrix.csv` | Model × Dataset performance matrix for round N |
| `round_N_avg_soft_labels.npy` | Consensus teacher soft label distributions for round N |

Additionally, LoRA adapter checkpoints are saved per client per round:

| Path | Description |
|------|-------------|
| `checkpoints/client_N/round_M/adapter/` | LoRA adapter weights |
| `checkpoints/client_N/round_M/cls_head.pt` | Classification head weights |
| `checkpoints/client_N/round_M/logits_kd.npy` | Cached logits for resumption |
| `checkpoints/client_N/round_M/meta.json` | Round metadata (accuracy, client_id) |

### Sample `round_metrics.jsonl` Record

```json
{
  "round": 1,
  "client_id": 1,
  "model_id": "openchat/openchat-3.5-0106",
  "dataset_name": "go_emotions",
  "eval_accuracy": 0.425,
  "eval_correct": 85,
  "eval_total": 200,
  "avg_ce_loss": 1.823,
  "avg_kd_loss": 0.0,
  "avg_total_loss": 1.823,
  "kd_active": false,
  "num_private_examples": 1950,
  "num_train_steps": 244,
  "local_epochs": 2
}
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
│   ├── eval_utils.py            # Metrics, summary tables, cross-dataset matrix, CSV export
│   ├── requirements.txt         # Python dependencies
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py           # Dataset loading, label harmonization, partitioning
│   │   └── verify_datasets.py   # Pre-flight dataset health & integrity audit
│   └── models/
│       ├── __init__.py
│       ├── wrapper.py           # FederatedClassifier, LoRA, quantization, pre-loading
│       └── verify_models.py     # Standalone model backbone verification utility
├── results/                     # Timestamped run output (gitignored)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── run_info.txt
│       ├── summary.csv
│       ├── round_metrics.jsonl
│       └── round_N_cross_dataset_matrix.csv
├── checkpoints/                 # LoRA adapter checkpoints (gitignored)
├── logs/                        # Training logs (gitignored)
├── .gitignore
└── README.md
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: bitsandbytes` | Not installed | `pip install bitsandbytes` |
| `quantization: None-bit` in logs | bitsandbytes import failed | Reinstall: `pip install bitsandbytes --force-reinstall` |
| `CUDA out of memory` | Model too large for VRAM | Ensure `quant_bits: 4` in config; reduce `batch_size_train`; try smaller models |
| `401 Unauthorized` for gated models | Missing HF token or license | Set `HF_TOKEN` env var; accept license on HuggingFace |
| Download stuck at 0% | Stale `.incomplete` files in HF cache | Delete `~/.cache/huggingface/hub/models--*/blobs/*.incomplete` |
| `ReadTimeoutError` during download | Slow connection | Retry; or set `HF_HUB_DOWNLOAD_TIMEOUT=60` env var |
| Pre-load phase fails | Architecture incompatibility | Run `python federated_emotion/models/verify_models.py` for diagnostics |

## Hardware Requirements

| Config | Min VRAM | Notes |
|--------|----------|-------|
| 4-bit quantization (default) | **8 GB** | RTX 3070/4060 and above (for 7B models) |
| 4-bit quantization (14B models) | **12 GB** | RTX 3080/4070 Ti and above |
| 8-bit quantization | **16 GB** | RTX 3090/4080 and above |
| Full precision (no quantization) | **24+ GB** | Not recommended for consumer GPUs |

Models are loaded and freed sequentially (one client at a time), so VRAM requirement is for a **single model** at peak usage, not all 10 simultaneously.

## License

This project is for academic/research purposes as part of a B.Tech Project (BTP).
