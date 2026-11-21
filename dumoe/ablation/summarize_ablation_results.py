#!/usr/bin/env python3
import csv
import statistics
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
INPUT_CSV = OUTPUT_DIR / "run_records.csv"
SUMMARY_CSV = OUTPUT_DIR / "ablation_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "ablation_summary.md"

METRIC_KEYS = [
    "top1_acc",
    "top3_acc",
    "top5_acc",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "recall_macro",
]


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _fmt(mean_v, std_v):
    return f"{mean_v:.4f}+-{std_v:.4f}"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    with INPUT_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("variant", ""), row.get("ablation", ""))
        grouped.setdefault(key, []).append(row)

    out_rows = []
    for (variant, ablation), items in grouped.items():
        out = {
            "variant": variant,
            "ablation": ablation,
            "n_seeds": len(items),
        }
        for mk in METRIC_KEYS:
            vals = [_to_float(x.get(mk, "")) for x in items]
            vals = [x for x in vals if x is not None]
            if vals:
                mean_v = statistics.mean(vals)
                std_v = statistics.stdev(vals) if len(vals) > 1 else 0.0
                out[f"{mk}_mean"] = f"{mean_v:.6f}"
                out[f"{mk}_std"] = f"{std_v:.6f}"
                out[f"{mk}_mean_std"] = _fmt(mean_v, std_v)
            else:
                out[f"{mk}_mean"] = ""
                out[f"{mk}_std"] = ""
                out[f"{mk}_mean_std"] = ""
        out_rows.append(out)

    out_rows.sort(key=lambda x: x["variant"])

    fieldnames = ["variant", "ablation", "n_seeds"]
    for mk in METRIC_KEYS:
        fieldnames.extend([f"{mk}_mean", f"{mk}_std", f"{mk}_mean_std"])

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    lines = []
    lines.append("| Model Variant | Ablation | Seeds | Top1 | Top3 | Top5 | NDCG@1 | NDCG@3 | NDCG@5 | Recall_macro |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in out_rows:
        lines.append(
            "| {variant} | {ablation} | {n_seeds} | {top1_acc_mean_std} | {top3_acc_mean_std} | {top5_acc_mean_std} | {ndcg@1_mean_std} | {ndcg@3_mean_std} | {ndcg@5_mean_std} | {recall_macro_mean_std} |".format(**row)
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
