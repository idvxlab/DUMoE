# Downstream protocol (Task A + Task B)

Code lives under [`downstream/`](../downstream/README.md). Both probes freeze the encoder, use the same windows, and the same `split_tag` as pretraining.

| | Task A | Task B |
| --- | --- | --- |
| Folder | `downstream/taskA/` | `downstream/taskB/` |
| Name | Window interest classification | Window-aligned message interaction |
| Input | frozen \(\mathbf{z}_t\) | \(\mathbf{z}_{t-1}\) + candidate tweet UNITE vector |
| Output | 15-way `label_id` | binary (own post vs other-user post) |
| Script | `taskA/run_classification.py` | `taskB/run_interaction.py` |
| Metrics | Hit@1, NDCG@3, Recall, KL | Acc, Recall, F1, BCE |

Shared: newest-first windows of 30 posts; split by `window_idx` (0–7 train, 8–9 val, 10–11 test). Window I/O: `downstream/windows_io.py`.

## Task A

- Head: two-layer MLP (`taskA/heads.py`).
- Default probe: label smoothing `0.1`; seeds `42,52,62`.
- Export vectors: `taskA/export_from_checkpoint.py`.

## Task B

- Positives: the user’s posts in window \(t\).
- Negatives: other users’ posts in the same time range, 1:1 (paper cap 10 pos / 10 neg).
- Tweet dim often reduced (`tweet_reduce_dim=256`).

## Fairness

Same window keys, splits, and probe hyperparameters for DUMoE and baselines unless a table says otherwise.
