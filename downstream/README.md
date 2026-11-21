# Downstream tasks

After pretraining DUMoE or a baseline, freeze the encoder and evaluate two probes on the **same** windows (`dataset/ids/windows.csv`).

```
downstream/
  windows_io.py              # shared window-table I/O (not Task-A-only)
  export_window_samples.py   # optional: cache → csv/jsonl
  taskA/                     # interest classification
  taskB/                     # post interaction
```

| | Task A | Task B |
| --- | --- | --- |
| Question | What is this user’s interest in window \(t\)? | Will the user post this candidate in window \(t\)? |
| User vector | \(\mathbf{z}_t\) (current window) | \(\mathbf{z}_{t-1}\) (previous window) |
| Label / target | 15-way `label_id` | binary: own post vs other-user post in the same time range |
| Head | 2-layer MLP classifier | user/item projection + MLP scorer |
| Metrics | Hit@1, NDCG@3, Recall, KL | Acc, Recall, F1, BCE |
| Paper probe | seeds `42,52,62`, label smoothing `0.1` | cap 10 pos / 10 neg per window; `tweet_reduce_dim=256` |

## Run (from repo root)

```bash
# Task A
python downstream/taskA/run_classification.py \
  --samples-csv dataset/ids/windows.csv \
  --embedding-files outputs/user_window_repr.pt \
  --seeds 42,52,62 --probe-label-smoothing 0.1

# Task B
python downstream/taskB/run_interaction.py \
  --samples-csv dataset/ids/windows.csv \
  --embedding-index-json outputs/embedding_index.json \
  --tweet-emb-shards-dir dataset/local/embeddings_pt \
  --seeds 42,52,62 --tweet-reduce-dim 256 \
  --max-pos-per-window-record 10 --max-neg-per-window-record 10
```

Or: `bash scripts/run_downstream.sh`.

Export DUMoE window vectors: `python downstream/taskA/export_from_checkpoint.py --checkpoint path/to/best_model.pth`.

Protocol details: [`../docs/downstream_protocol.md`](../docs/downstream_protocol.md).
