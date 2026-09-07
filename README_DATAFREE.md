# Data-Free Federated Distillation (`mode: data_free_fd`)

Heterogeneous federated learning across five 3–4B LLM clients **with no shared corpus of any
kind**. Clients exchange one averaged logit vector per class they hold — roughly 144 bytes per
client per round — and are evaluated on their own held-out slices.

This document covers the data-free mode only. The original public-transfer-set method
(`mode: public_set`) is described in [`README.md`](README.md) and
[`pipeline_comprehensive_architecture_report.md`](pipeline_comprehensive_architecture_report.md);
both modes live in the same codebase and are one config field apart.

---

## 1. Why this mode exists

The original pipeline is FedMD / FedDF / DS-FL lineage: every client answers the same 500
sentences from a shared public pool (`dair-ai/emotion`), and the server averages those answers
into a consensus teacher. That works, but it presupposes a public corpus drawn from the same
task — an assumption that often fails in practice, and one that is worth being able to drop.

Data-free mode removes it. The key observation is that **everything communicated can be keyed by
class instead of by example.**

| | Aligns on | Needs a shared corpus? |
|---|---|---|
`public_set` | *"what do you predict for **this sentence**?"* | **yes** — everyone must see the same sentences |
`data_free_fd` | *"what does **class 3** look like in your model?"* | **no** — class IDs are shared by definition |

A second, incidental benefit: logits live in `[num_classes]` space, which is completely
independent of the backbone. Clients can differ in family, hidden size, and depth with **zero**
alignment machinery — no `feature_dim`, no pooling, no projection layer.

---

## 2. The algorithm

**FD / FedDistill** — Jeong et al., *Communication-Efficient On-Device Machine Learning:
Federated Distillation and Augmentation under Non-IID Private Data* (2018),
[arXiv:1811.11479](https://arxiv.org/abs/1811.11479).

Reference implementation consulted: HtFLlib `flcore/clients/clientfd.py`
(Zhang et al., KDD 2025, [arXiv:2506.03954](https://arxiv.org/abs/2506.03954)).

### One training step

```
private batch  ──►  ONE forward pass through the 4-bit LLM + LoRA  ──►  [B, 6] logits
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
   CE(own true labels)     KL to the federation's    collect into
                           per-class targets          class buckets
            └───────────────────────┬───────────────────────┘   (the upload payload,
                                    ▼                            gathered for free)
        loss = CE  +  fd_lambda · KL          ──►  backward  ──►  step
```

The distillation target for example *i* is the federation's averaged logit vector for example
*i*'s **true class**. Rows whose class is absent from the global set keep the model's own output
and therefore contribute exactly zero to the divergence.

Contrast `public_set`, which needs a **second** forward pass on a separate public batch. Data-free
mode is roughly 2× cheaper per step and drops the 500-example KD-pool inference stage entirely.

### One round

```
ROUND r
├─ for each client k, sequentially (one LLM in VRAM at a time):
│    ├─ local training, `local_epochs` epochs
│    ├─ average the collected logits per class  →  {class: [6]}
│    ├─ evaluate on its OWN held-out slice
│    └─ save LoRA adapter + head
│
├─ SERVER: for each class c, average the vectors from the clients that hold c,
│          weighted by their class-c example counts
│
└─ broadcast → next round's distillation targets
```

`kd_warmup_rounds: 1` means **round 1 trains purely locally**, so round-1 accuracy is a free
local-only reference point. No separate baseline run is needed.

### What actually travels

```python
{0: [6 floats],  1: [6 floats],  ...}     # one vector per class the client holds
[n_0, n_1, ..., n_5]                       # per-class example counts
```

6 classes × 6 floats × 4 bytes = **144 bytes** per client per round. No weights, no gradients,
no data, no public corpus.

---

## 3. The roster

Five 3–4B backbones spanning three architecture families, each paired with a different dataset.

| Client | Backbone | Family | hidden | Dataset |
|---|---|---|---|---|
1 | `microsoft/Phi-3.5-mini-instruct` | Phi-3 | 3072 | `go_emotions` (simplified) |
2 | `Qwen/Qwen2.5-3B-Instruct` | Qwen2.5 | 2048 | `tweet_eval` (emotion) |
3 | `microsoft/Phi-3-mini-4k-instruct` | Phi-3 | 3072 | `sem_eval_2018_task1` (subtask5.english) |
4 | `Qwen/Qwen2.5-3B` | Qwen2.5 | 2048 | `xed_en_fi` (en_annotated) |
5 | `Qwen/Qwen1.5-4B-Chat` | Qwen1.5 | 2560 | `emotion` (dair-ai) |

Backbones and ordering deliberately match the parallel `public_set` experiment, so results from
the two modes are directly comparable: same models, same datasets, same hardware.

### Canonical label space

Six classes, defined by `dair-ai/emotion`'s taxonomy:

```
0 sadness   1 joy   2 love   3 anger   4 fear   5 surprise
```

Every dataset is remapped onto these via `LABEL_MAPS` in `data/loaders.py`; examples with no
clean correspondence are dropped.

### Two notes on dataset selection

**Clients 4 and 5 replaced the old `silicone` entries.** Those entries appeared to work only
because `DATASET_ALIASES` listed `mteb/emotion` as their first candidate — a mirror of the
public evaluation corpus. Clients 4–8 were therefore training on the evaluation set, and because
`aggregation_mode: "accuracy_weighted"` their inflated accuracy inflated their vote in the
consensus, spreading the contamination to every other client. The alias is gone, those datasets
now fail to load rather than silently substituting wrong data, and `load_private_dataset()`
refuses any resolution to a public-corpus path.

**Client 5 uses `dair-ai/emotion` directly.** This is legitimate *only* in data-free mode, where
no public pool is loaded, so the corpus is nothing but that one client's private data. The guard
enforces the distinction: it **raises** if you try this under `mode: public_set`, and prints an
explanatory `[NOTE]` under `data_free_fd`. It is also the only source natively carrying all six
canonical labels, which lifts `love` from one usable holder to two.

---

## 4. Class coverage — read this before interpreting any result

Both modes aggregate **per class**, so a class that few clients hold has a weak or nonexistent
consensus. Measured coverage of the active roster (2000-example cap per client, 2026-09):

```
client  dataset            sadness    joy   love  anger   fear  surprise   total
1       go_emotions            310    331    475    472    153       259    2000
2       tweet_eval             581    511      0!   908      0!        0!   2000
3       sem_eval               363    751     13~   617    228        28    2000
4       xed_en_fi              310    477      0!   611    321       281    2000
5       emotion (dair-ai)      560    692    180    270    223        75    2000
        ──────────────────────────────────────────────────────────────────────────
        TOTAL                 2124   2762    668   2878    925       643   10000
        holders                  5      5      3      5      4         4
        usable (>=20)            5      5      2      5      4         4

! = class absent    ~ = fewer than 20 examples
```

`love` is the weak column: three holders, only two of them usable. Expect it to be the worst
class, and **report per-class F1 rather than accuracy alone** — a `love` failure is invisible in
the average.

Client 3 is also severely imbalanced internally (751 `joy` against 13 `love`, a 58× ratio). That
is within-client skew rather than a coverage gap, and no aggregation rule fixes it; it is simply
what `sem_eval_2018_task1` looks like once mapped onto six classes.

The pipeline prints this table at startup. To inspect it without spending GPU time:

```bash
python -m federated_emotion.analyze_coverage --clients 1 2 3 4 5
```

CPU only, no models loaded. Writes `coverage_report.json`.

### On the datasets that are *not* in the roster

Six candidates fail to load, all for the same reason — they ship a Python loading script, which
`datasets` v3+ rejects outright:

```
Dataset scripts are no longer supported, but found <name>.py
```

That covers `silicone/{dyda_e, meld_e, iemocap}`, `empathetic_dialogues`, `emo`, and
`daily_dialog`. The first five previously *appeared* to work only because `DATASET_ALIASES`
redirected them to `mteb/emotion`, i.e. the evaluation corpus — see §3. `daily_dialog` is the
upstream source of `silicone/dyda_e` and was trialled as its replacement; it fails identically,
which is why `xed_en_fi` and `dair-ai/emotion` took slots 4 and 5 instead.

They remain in `CLIENT_DATASETS` (IDs 6–12) with their label maps intact, so the roster's
provenance stays legible and any of them can be reinstated if a parquet mirror appears.

---

## 5. Two deliberate improvements over the reference

Both are switchable, so each is a clean ablation row.

### Count-weighted aggregation — `fd_weight_by_count: true`

The reference takes an unweighted mean across the clients holding each class. That gives a client
with 13 examples of a class the same vote as one with 475. Weighting by count fixes it:

```
'love'      client 1  475 ex →  71.1%   (unweighted: 33.3%)
            client 3   13 ex →   1.9%   (unweighted: 33.3%)   ← 13 examples, equal vote
            client 5  180 ex →  26.9%   (unweighted: 33.3%)

'surprise'  client 1  259 ex →  40.3%   (unweighted: 25.0%)
            client 3   28 ex →   4.4%   (unweighted: 25.0%)
            client 4  281 ex →  43.7%   (unweighted: 25.0%)
            client 5   75 ex →  11.7%   (unweighted: 25.0%)
```

In both settings, a client holding **zero** examples of a class contributes nothing to it —
`average_class_logits` only emits vectors for classes actually observed.

Set `fd_weight_by_count: false` to reproduce the reference exactly.

### Proper temperature-scaled KL

The reference computes `nn.CrossEntropyLoss(output, logit_target)` where `logit_target` holds
**raw, unnormalised logits**. PyTorch then treats them as class probabilities even though they
may be negative and do not sum to one. It trains, but it is not a valid divergence. This
implementation uses

```
KL( softmax(target/T) ‖ softmax(output/T) ) · T²
```

with the `T²` factor restoring the gradient magnitude that dividing by `T` removes
(Hinton, Vinyals & Dean, 2015).

---

## 6. Configuration

Full reference: `federated_emotion/config.yaml`. The fields that matter here:

```yaml
mode: "data_free_fd"          # or "public_set"

num_clients: 5
active_client_ids: [1, 2, 3, 4, 5]
num_rounds: 5
local_epochs: 3
batch_size_train: 8
batch_size_infer: 16
max_seq_length: 128
learning_rate: 2e-4

quant_bits: 4                 # real NF4. Any value other than 4 or 8 disables quantization
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
optimizer_8bit: true

sparse_training:
  enabled: false              # when true, reduced_rank OVERRIDES lora_rank for all clients
  reduced_rank: 2

fd_lambda: 1.0                # weight of the per-class distillation term
fd_temperature: 2.0
fd_weight_by_count: true

kd_warmup_rounds: 1           # round 1 is local-only → your free baseline
local_holdout_size: 200       # per-client eval slice; the ONLY eval set in this mode
private_dataset_max_size: 2000
```

Ignored in this mode: `kd_lambda`, `kd_temperature`, `aggregation_mode`,
`aggregation_temperature`, `public_kd_pool_size`, `public_eval_holdout_size`.

### Settings worth being careful about

| Field | Note |
|---|---|
`quant_bits` | Must be **4** or 8 for NF4. `16` silently disables quantization; the manifest now says so explicitly instead of printing the meaningless `"16-bit NF4"`. |
`sparse_training.enabled` | When true, `reduced_rank` replaces `lora_rank` everywhere. The manifest reports the *effective* rank — it previously printed `lora_rank` regardless, misreporting every sparse run. |
`local_holdout_size` | This is the only evaluation set each client has. The former hard-coded 50 gave ~8 examples per class over 6 classes — far too few to read. |
`fd_lambda` | `0` reduces the run to pure local training on every round, i.e. a full local baseline. |

---

## 7. Running

```bash
cd HeteroFed_with_KT

# optional, minutes, no GPU: verify which datasets load and how classes are covered
python -m federated_emotion.analyze_coverage --clients 1 2 3 4 5 6

# the run
python -m federated_emotion.main

# resume from checkpoints (restores per-class logits and counts, not just accuracy)
python -m federated_emotion.main --resume
```

CLI overrides: `--config`, `--num_rounds`, `--local_epochs`, `--num_clients`, `--resume`.

Requires `HF_TOKEN` in the environment (or set `hf_token_env_var` to a literal `hf_...` token).

### Cost

```
1800 train examples ÷ batch 8 = 225 steps × 3 epochs = 675 steps per client-round
× 5 clients × 5 rounds                                = 16,875 forward passes
```

One forward per step, not two, and no KD-pool inference stage. On a 16 GB card with 4-bit NF4 and
rank-8 LoRA, expect roughly **1.5–2 h per full run**, one model resident at a time.

---

## 8. Output

```
results/run_<timestamp>/
├── run_info.txt                              experiment manifest
├── round_metrics.jsonl                       per client, per round
├── round_N_global_class_logits.json          the aggregated consensus (this mode's artifact)
├── cross_eval_round_N.csv                    client × dataset matrix (final round)
├── detailed_metrics.csv, summary.csv
└── timing_summary.{json,txt}                 per-stage wall-clock

checkpoints/client_<id>/round_<n>/
├── adapter_model.safetensors, adapter_config.json
├── head.pt
├── class_logits.json                         this client's upload payload
└── meta.json                                 eval_accuracy, eval_source, class_counts
```

`round_N_avg_soft_labels.npy` is a `public_set` artifact and does not appear in this mode.

### What to look at

1. **Round 1 vs round 5 accuracy.** Round 1 is local-only. If round 5 is not better, distillation
   is not helping and the cause is worth finding before anything else.
2. **The cross-evaluation matrix.** Every client scored on every other client's holdout. Success
   means client 1 improves on *client 3's* data beyond what it could reach alone — accuracy it
   could only have acquired through the federation.
3. **The diagonal vs. off-diagonal gap.** A client scoring high on its own data and near-random
   everywhere else indicates a **label-mapping inconsistency**, not a federation failure. Watch
   for accuracy *below* chance (16.67% for 6 classes) — that is a systematic sign inversion, not
   noise.
4. **Per-class F1, especially `love`.** The average will hide it.
5. **`Client_Class_Logits` in the timing summary.** Should be near zero. If it is not, the payload
   is being recomputed instead of collected during training.

---

## 9. Design notes and limitations

**No global model.** Like `public_set`, this mode produces five separately improved clients, not
one central model. The server holds only a dict of 6 vectors. Evaluation is per-client.

**No public data anywhere in the algorithm.** `eval_accuracy` comes from each client's own
held-out slice, and FD's aggregation is count-weighted rather than accuracy-weighted, so no
public corpus influences training or aggregation even indirectly.

**Logits are averaged across local epochs.** `class_logit_store` accumulates one entry per
example *per epoch*, so the uploaded vector averages over a changing model — early-epoch logits
from a weaker model are mixed in. The reference behaves the same way. Collecting only during the
final epoch would give fresher targets and is a two-line change; left as a future ablation.

Note that `class_counts` is computed from the dataset via `compute_class_counts()`, **not** from
`len(class_logit_store[c])`, which would be `local_epochs ×` the true count.

**Class coverage is the binding constraint, not compute.** With only three holders for `love`,
there is very little consensus to form for that class. Adding clients that hold the rare classes
would help more than any change to the aggregation rule.

**Heterogeneity is free here, which is also a limitation.** Because only `[num_classes]`-dim
logits are shared, nothing about the backbones' representations is ever aligned or compared. That
is what makes the method trivially architecture-agnostic, and also what caps how much can be
transferred. Feature-space methods (FedKD's `W_h` projection, prototypes) transfer more but
require alignment machinery — the natural next step if this mode plateaus.

---

## 10. Files

| File | Role |
|---|---|
`federated_emotion/datafree.py` | FD primitives: collect, average, aggregate, distil, persist |
`federated_emotion/analyze_coverage.py` | standalone class-coverage diagnostic (CPU only) |
`federated_emotion/client.py` | per-client round; branches on `config.is_data_free` |
`federated_emotion/main.py` | orchestration, mode dispatch, local holdouts, manifest |
`federated_emotion/config.py` | `SUPPORTED_MODES`, `fd_*`, `local_holdout_size` |
`federated_emotion/data/loaders.py` | roster, label maps, leak guard, `compute_class_counts` |
`federated_emotion/models/wrapper.py` | 3–4B backbone registry, 4-bit NF4 + LoRA wrapper |
`federated_emotion/server.py` | `public_set` aggregation only; untouched by this mode |

---

## 11. References

- **FD / FedDistill** — Jeong et al., 2018. [arXiv:1811.11479](https://arxiv.org/abs/1811.11479)
- **HtFLlib** — Zhang et al., KDD 2025. [arXiv:2506.03954](https://arxiv.org/abs/2506.03954) ·
  [code](https://github.com/TsingZ0/HtFLlib). Source of the reference FD implementation and of the
  text benchmark showing FD to be the strongest data-free method on text (91.35% on AG News).
- **FedMD** — Li & Wang, 2019. [arXiv:1910.03581](https://arxiv.org/abs/1910.03581).
  The `public_set` mode's lineage.
- **KD with temperature** — Hinton, Vinyals & Dean, 2015.
  [arXiv:1503.02531](https://arxiv.org/abs/1503.02531). Origin of the `T²` factor.
- **FedKD** — Wu et al., *Nature Communications* 13, 2032 (2022).
  [paper](https://www.nature.com/articles/s41467-022-29763-x). Feature-space alternative; the
  candidate next step.
- **QLoRA** — Dettmers et al., 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
  4-bit NF4 quantization.
