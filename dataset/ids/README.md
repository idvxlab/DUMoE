# Identifier files

Canonical release: **`dumoe_ids.zip`** (~72 MB). Full tables are **not** checked into git unpacked; unzip when you need them:

```bash
# from the DUMoE repo root
unzip dataset/ids/dumoe_ids.zip -d dataset/ids
```

| File | Columns | Rows |
| --- | --- | ---: |
| `users.csv` | `user_id` | 14,015 |
| `posts.csv` | `user_id,post_id` | 7,685,700 |
| `windows.csv` | `user_id,window_idx,label_id,split_tag` | 156,757 |

- `user_id` / `post_id` are **numeric X ids** (not @handles).
- `posts.csv` is **newest → oldest** within each user. Count matches the paper table (**7,685,700**): unique post ids used by the model (each labeled window plus up to 300-post long context).
- `windows.csv` has interest labels (`label_id`) and `train` / `val` / `test` for windows 0–11.

Schema previews (committed): `users.sample.csv`, `posts.sample.csv`, `windows.sample.csv`.

## Labels

Released labels are in **`windows.csv`**.

| column | meaning |
| --- | --- |
| `user_id` | numeric user |
| `window_idx` | 0 = newest window |
| `label_id` | 0–14 (see `../labels/interest_categories.json`) |
| `split_tag` | `train` / `val` / `test` |

Not in the zip: tweet text, images, VLM chat logs. Hydrate via the X API if needed; re-label with `../preprocessing/labeling/label_interest_windows.py`.
