# Pruned-BERT Data-Free Federation with Function-Consistent Feature Distillation

This branch (`feature/pruned-bert-fcfd`) replaces the five heterogeneous 3–4B LLM clients of
`feature/data-free-mode` with **one global `bert-base-uncased` that the server prunes into five
client models of different capacity**. Clients then exchange per-class *averaged outputs*
(intermediate features + logits) and distil toward the server's aggregate using **FCFD**
(Liu et al., *Function-Consistent Feature Distillation*, ICLR 2023).

## Pipeline

```
SERVER (round 0)
  bert-base-uncased ──magnitude pruning──► 25% │ 25% │ 50% │ 75% │ 100%   → client_k/round_0

dair-ai/emotion train ──Dirichlet(α)──► 5 non-IID client shards (+ local holdout each)
dair-ai/emotion test  ──────────────► shared global test set (evaluation only)

ROUND t
  CLIENT k:  CE (+ FCFD toward g^{t-1} if t > warmup)
             upload  P_k = { logits[c], features[layer][c] }   for every class c it holds
  SERVER:    g^t[c] = Σ_k w_k^c · P_k[c]     (per class, per layer)
             send g^t to all clients
```

## 1. Server-side pruning (data-free)

Pruning happens once, on the server, without any data. For each encoder layer:

* **Attention heads** are scored by the L1 norm of their slices of Q, K, V (rows) and the
  attention output projection (columns). The top `capacity × 12` heads are kept, and the rest
  are removed with HF `BertModel.prune_heads`.
* **FFN neurons** are scored by the L1 norm of their row in `intermediate.dense` plus their
  column in `output.dense`. The top `capacity × 3072` neurons are kept by physically slicing
  both linear layers.

The hidden size (768) and the embeddings are never pruned. As a result:

| capacity | heads/layer | FFN/layer | encoder params | total params |
|---|---|---|---|---|
| 0.25 | 3 | 768 | 25.0% | 41.5% |
| 0.50 | 6 | 1536 | 50.0% | 61.0% |
| 0.75 | 9 | 2304 | 75.0% | 80.5% |
| 1.00 | 12 | 3072 | 100% | 100% |

Every client still produces 768-d hidden states. That is the property that makes per-class
**feature** prototypes comparable across clients of different size. With the old LLM roster,
hidden sizes differed (2048/2560/3072), so only logits could be exchanged.

The capacities are set by `client_capacities`, in the same order as `active_client_ids`.

## 2. What is uploaded: per-class averaged outputs

During the last local epoch, each training forward pass also buckets, by true label:

* the logit vector, and
* the masked-mean-pooled hidden state at every layer in `fcfd_positions` (default `[4, 8, 12]`).

At the end of the round the client averages each bucket and uploads
`P_k = { logits[c] ∈ R^6, features[l][c] ∈ R^768 }`. This tells the server what class *c* looks
like to that client, both in feature space and at the output, without revealing any sample.
Payload size is `|classes| × (6 + |positions| × 768)` floats, about 54 KiB for 6 classes and
3 positions.

## 3. Server aggregation

For every class *c*, and separately for the logits and each feature position, the server takes
a weighted mean over the clients that hold *c*. It reuses `aggregate_class_logits` from the
data-free branch unchanged:

    w_k^c ∝ (n_k^c)^count_exponent · (r_k^c)^reliability_exponent

Here `n_k^c` is client k's class-c count and `r_k^c` is its shrunk per-class accuracy on its own
holdout (see `README_DATAFREE.md`). The result `g = { logits[c], features[l][c] }` is sent to
every client and persisted as `logs/round_N_global_payload.json`.

## 4. Client objective: FCFD toward the aggregate

The paper distils a student toward a teacher *network*. Here the teacher is the federation's
per-class aggregate `g`, so every teacher quantity is looked up by the sample's label `y`:

    L = CE(s(x), y)
      + λ_kd  · T² · KL( σ(g.logits[y]/T) ‖ σ(s(x)/T) )                         (L_kd)
      + λ_app · Σ_{l∈K} ‖ B_st^l(f_s^l(x)) − g.features[l][y] ‖²                   (L_app)
      + λ_func· Σ_{l∈S} T² · KL( σ(g.logits[y]/T) ‖ σ(S_{>l}(inject(h_s^l(x), B_ts^l(g.features[l][y])))/T) )   (L_func′)

* `f_s^l(x)` is the pooled student feature at layer *l*, and `h_s^l(x)` is the token sequence.
* `B_st^l` and `B_ts^l` are identity-initialised 768×768 bridges, one pair per position. They
  are trained with the student and checkpointed with it (`extra.pt`).
* `S_{>l}` is the **student's own later layers** plus its classifier head (`forward_from_layer`).
* `S` is a random subset of `fcfd_num_paths` positions, re-sampled every step (paper Eq. 9).
* `inject` places the bridged class prototype into the student's layer-*l* token sequence so
  that the student's later layers can process it.

**Appearance vs. function.** `L_app` pulls the student's features toward the class prototype in
L2, which treats all 768 directions equally. `L_func′` asks a different question: *if the
student's later layers read the federation's class-c prototype, do they output the federation's
class-c prediction?* Minimising it makes the student's later layers decode the shared prototype
the same way the federation does. That is the function-consistency idea of FCFD: features
matter through how later layers process them, not just through their L2 distance.

### Deviations from the paper

1. **No teacher network.** The teacher is a set of per-class averages, not a model, so the
   paper's `L_func` term (student feature → *teacher's* later layers) has no network to run
   through. Only the student-lens term `L_func′` is used. This also means no client has to
   carry layers bigger than its own pruned model.
2. **Class-level targets.** Teacher features and logits are per-class means rather than
   per-sample outputs, which is what makes the scheme data-free.
3. **Pooled features.** Feature matching uses the masked mean over tokens, because a class
   average over variable-length sequences is only defined after pooling.
4. **Bridges are linear (768→768).** The paper uses conv+BN for CNNs. Here hidden sizes already
   match, so the bridges start as identity.

## Data

`dair-ai/emotion` with 6 labels: sadness, joy, love, anger, fear and surprise.
* The train split (16k) is partitioned across clients by a per-class Dirichlet(`dirichlet_alpha`)
  split, re-drawn until every client has at least `min_client_samples` examples. Each shard is
  then capped at `private_dataset_max_size`.
* The tail of each shard (`local_holdout_size`) is that client's local holdout. This is used for
  reliability weighting.
* The test split (capped at `test_set_size`) is a shared global test set, used for evaluation only.

## Running

```bash
pip install -r federated_emotion/requirements.txt
python federated_emotion/main.py                                          # full run (config.yaml)
python federated_emotion/main.py --config federated_emotion/config_smoke.yaml   # 2-round smoke test
python federated_emotion/main.py --resume                                 # resume from checkpoints
```

## Key config fields

| field | meaning |
|---|---|
| `bert_model_name` | global model to prune |
| `client_capacities` | encoder keep ratio per client |
| `dirichlet_alpha`, `min_client_samples` | non-IID strength and minimum shard size |
| `fcfd_positions` | encoder layers whose pooled outputs are averaged, uploaded and distilled |
| `fcfd_num_paths` | positions sampled per step for `L_func′` |
| `fcfd_kd_weight`, `fcfd_app_weight`, `fcfd_func_weight` | λ_kd, λ_app, λ_func |
| `fd_temperature` | T |
| `kd_warmup_rounds` | rounds of pure local training before distillation starts |
| `fd_weight_by_count`, `fd_use_reliability`, ... | aggregation weighting (unchanged from data-free branch) |

`mode: public_set` and the LLM wrapper (`models/wrapper.py`) are kept for history, but the
pruned-BERT pipeline only runs `mode: data_free_fd`.
