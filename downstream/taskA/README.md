# Task A — window interest classification

Frozen user vector \(\mathbf{z}_t\) → one of 15 interest classes.

| File | Role |
| --- | --- |
| `run_classification.py` | Paper probe (Hit@1 / NDCG@3 / Recall / KL) |
| `export_from_checkpoint.py` | Encode windows from a DUMoE `.pth`, optionally run the probe |
| `export_embeddings.py` | Encode from a sample cache |
| `train_classifier.py` | Train `ClassifierHead` on exported embeddings |
| `heads.py` | Two-layer MLP |
| `build_sample_cache.py` / `build_msg_table_chunks.py` | Optional caches |

Input labels: `dataset/ids/windows.csv` (`user_id`, `window_idx`, `label_id`, `split_tag`).
