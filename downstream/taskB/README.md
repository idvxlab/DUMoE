# Task B — window-aligned message interaction

Frozen user vector from window \(t-1\) + candidate tweet UNITE vector → whether the tweet is a real post of that user in window \(t\).

| File | Role |
| --- | --- |
| `run_interaction.py` | Paper binary interaction probe |

- Positives: the user’s posts in window \(t\).
- Negatives: other users’ posts in the same time range (1:1; paper cap 10/10).
- Metrics: Acc, Recall, F1, BCE.

Needs local tweet shards (`dataset/local/embeddings_pt`) and window embeddings exported for Task A (same \(\mathbf{z}\)).
