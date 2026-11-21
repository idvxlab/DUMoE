# Multitask Representation Variants Runner

This folder runs 7 multitask variants and exports user-window embeddings.

## Variants
- Full
- NoAdapter
- NoGating
- OnlyUser
- OnlyShort
- OnlyLong
- JointAllModules

Each variant is trained with the multitask pipeline and produces:
- best checkpoint: best_model.pth
- embeddings: dynamic_user_embeddings.pt
- summary.json (per seed)

A registry is written to:
- representation_registry.csv
- representation_registry.json

## Usage
Run from this folder:

```bash
/home/idvxlab/.conda/envs/unite/bin/python run_representation_variants_multitask.py --seeds 42,2023,3407 \
  --device cuda:1 --num-workers 8 --pin-memory --persistent-workers \
  --use-cached-labels --use-cached-intersection --use-cached-needed-msg-ids
```

Common options:
- --batch-size, --export-batch-size
- --stage1-epochs, --stage2-epochs, --stage3-epochs
- --joint-epochs, --lr-joint
- --rec-weight, --cls-weight, --rec-num-neg
- --skip-existing
- --skip-variants Full,NoAdapter

Notes:
- The runner uses cached labels/intersection/needed_msg_ids when enabled.
- The pipeline export uses the multitask model and supports base/adapter/thresholded modes.
