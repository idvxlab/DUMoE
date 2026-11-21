# DUMoE

Official code and dataset release for **DUMoE** (*Drift-aware User Modeling with Mixture-Of-Experts*).

> **Paper:** [Drift-Aware Multimodal User Representation Learning via Multi-Scale Temporal Modeling and Sparse Mixture-of-Experts](https://arxiv.org/abs/2608.25773)  
> **Authors:** Ziqing Qian, Haohang Chen, Shengqi Dang, Yuhan Xiong, Canyu Shen, Jiaying Lei, Nan Cao  
> **arXiv:** [2608.25773](https://arxiv.org/abs/2608.25773)

---

## Paper overview

Understanding user preferences from noisy, temporally evolving social media behavior is hard because of **interest drift**: preferences shift over time, show both short- and long-term patterns, and span multiple co-existing interests.

**DUMoE** is a unified framework for drift-aware multimodal user representation learning. It has two main parts:

1. **Temporal dynamics-aware backbone** — fuses static profile signals, short-term behavior (LSTM), and long-term dependencies (Transformer) into one user vector.  
2. **Sparse MoE interest adapter** — specialized experts model distinct interest subspaces; a gating network selects and aggregates a sparse subset of experts per user.

Training uses a **three-stage** schedule (backbone → expert specialization → gating) with a multitask objective that jointly optimizes interest classification and next-item interaction:

\[
L = L_{\mathrm{cls}} + 1.5\, L_{\mathrm{rec}}
\]

We also release a large multimodal dataset from **X** (formerly Twitter): **14,015** users, **7,685,700** posts, **2,890,668** posts with images, covering **15** interest domains (Nov 2025 – Mar 2026). Public files provide **user/post ids and window labels only** (no post text or images), following X redistribution rules.

If anything about the dataset is unclear, please contact us.

---

## Repository layout

| Path | Role |
| --- | --- |
| [`dumoe/`](dumoe/) | Model + **staged** (main) / joint training + ablations |
| [`downstream/`](downstream/README.md) | Frozen probes: **Task A** (interest classification) / **Task B** (interaction) |
| [`baselines/project/`](baselines/project/) | Baseline pretraining + Exp1–3 |
| [`dataset/`](dataset/README.md) | Stats, labels, preprocessing, splits, id zip |
| [`scripts/`](scripts/) | One-command launchers |
| [`configs/`](configs/) | Paper hparams + path template |
| [`docs/downstream_protocol.md`](docs/downstream_protocol.md) | Task A / Task B protocol |

**Not shipped:** post text, images, usernames, UNITE embeddings, checkpoints, cookies, API keys.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/idvxlab/DUMoE.git
cd DUMoE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Unpack id tables

```bash
unzip dataset/ids/dumoe_ids.zip -d dataset/ids
```

| File | Rows | Contents |
| --- | ---: | --- |
| `users.csv` | 14,015 | numeric `user_id` |
| `posts.csv` | 7,685,700 | `user_id,post_id` (newest → oldest) |
| `windows.csv` | 156,757 | `user_id,window_idx,label_id,split_tag` |

Schema previews: `dataset/ids/*.sample.csv`. Full dataset docs: [`dataset/README.md`](dataset/README.md).

### 3. Prepare local artifacts

Hydrate posts via the [official X API](https://developer.x.com/) if you need text/images, then build UNITE embeddings locally. Point paths in:

```bash
cp configs/paths.example.env configs/paths.env
# edit configs/paths.env → user profile .pt, tweet emb shards, samples, labels cache
source configs/paths.env
```

Place large local files under `dataset/local/` (gitignored).

### 4. Train DUMoE (paper main = staged)

```bash
bash scripts/run_dumoe_staged.sh
```

This runs Stage1 → Stage2 → Stage3 with both heads (`cls_weight=1.0`, `rec_weight=1.5`). Hyperparameters: [`configs/paper_hparams.yaml`](configs/paper_hparams.yaml).

Optional ablation without stage-wise training:

```bash
bash scripts/run_dumoe_joint.sh
```

### 5. Train baselines

```bash
bash scripts/run_baselines.sh
```

Models: SASRec, BERT4Rec, MIND, PeterRec, PTUM, UniSRec, HORAE, DisMIR, HM4SR (25 epochs by default).

### 6. Downstream evaluation (frozen encoder)

```bash
bash scripts/run_downstream.sh          # Task A then Task B
# or separately:
bash scripts/run_taskA.sh               # 15-way window classification
bash scripts/run_taskB.sh               # next-window interaction
```

| | Task A | Task B |
| --- | --- | --- |
| Goal | Predict window interest (15 classes) | Predict whether a candidate post is the user’s in window \(t\) |
| Input | \(\mathbf{z}_t\) | \(\mathbf{z}_{t-1}\) + tweet UNITE vector |
| Metrics | Hit@1, NDCG@3, Recall, KL | Acc, Recall, F1, BCE |

Windows are newest-first, size **30**; split by index **0–7** train / **8–9** val / **10–11** test. Details: [`downstream/README.md`](downstream/README.md).

### 7. Ablations

See [`dumoe/ablation/README.md`](dumoe/ablation/README.md) (adapter / gate / branch removals, staged vs joint).

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{qian2026dumoe,
  title={Drift-Aware Multimodal User Representation Learning via Multi-Scale Temporal Modeling and Sparse Mixture-of-Experts},
  author={Qian, Ziqing and Chen, Haohang and Dang, Shengqi and Xiong, Yuhan and Shen, Canyu and Lei, Jiaying and Cao, Nan},
  journal={arXiv preprint arXiv:2608.25773},
  year={2026}
}
```

## License

Code: MIT (`LICENSE`).  
Dataset ids: research use; any hydrated content remains under **X’s terms**.
