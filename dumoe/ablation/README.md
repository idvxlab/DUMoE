# Ablations

Paper **Full / Ours** is **staged** (`none` + Stage1→2→3). Joint is the
ablation without stage-wise training.

## Variants (`--ablation`)

| Name | Flag |
| --- | --- |
| Full | `none` |
| A1_NoAdapter | `no_adapter` |
| A2_NoGating | `no_gating` |
| A3_OnlyUser | `only_user` |
| A4_OnlyShort | `only_short` |
| A5_OnlyLong | `only_long` |

Default staged setup in `run_ablation_suite.py`: batch 32, stage epochs 10/4/15, seeds `42,2023,3407`.

## Run (from this folder)

```bash
python run_ablation_suite.py
python summarize_ablation_results.py

# staged vs joint
python run_training_strategy_ablation.py
python run_strategy_ablation_downstream.py

# controlled / representation variants
python controlled/run_ablation_controlled_multitask.py
python representation_variants_multitask/run_representation_variants_multitask.py
```

Write outputs under a local `results/` directory (gitignored via repo-level patterns).
