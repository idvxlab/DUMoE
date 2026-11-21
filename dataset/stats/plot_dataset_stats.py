#!/usr/bin/env python3
"""Plot domain-level dataset statistics from dataset_stats.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["domain"].lower() == "total":
                continue
            rows.append(
                (
                    row["domain"],
                    int(row["num_users"]),
                    int(row["num_posts"]),
                    int(row["num_images"]),
                )
            )
    return rows


def human_readable(v, _):
    abs_v = abs(float(v))
    if abs_v >= 1_000_000:
        x = v / 1_000_000
        return f"{int(round(x))}M" if abs(x - round(x)) < 1e-6 else f"{x:.2f}".rstrip("0").rstrip(".") + "M"
    if abs_v >= 1_000:
        x = v / 1_000
        return f"{int(round(x))}K" if abs(x - round(x)) < 1e-6 else f"{x:.2f}".rstrip("0").rstrip(".") + "K"
    return str(int(round(v)))


def main() -> None:
    base = Path(__file__).resolve().parent
    rows = load_rows(base / "dataset_stats.csv")
    rows.sort(key=lambda x: x[2], reverse=True)
    domains = [r[0] for r in rows]
    users = [r[1] for r in rows]
    posts = [r[2] for r in rows]
    images = [r[3] for r in rows]
    x = np.arange(len(domains))
    scale = 1.30

    fig, ax1 = plt.subplots(figsize=(10, 3))
    ax2 = ax1.twinx()
    ax2.set_zorder(1)
    ax1.set_zorder(2)
    ax1.patch.set_alpha(0)

    bar_color = "#93c5fd"
    ax2.bar(x, users, width=0.54, color=bar_color, alpha=0.55, label="# Users", zorder=1)
    ax2.set_ylabel("# Users")
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(500))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(human_readable))
    ax2.set_ylim(0, max(users) * 1.28)

    tweet_color = "#2563eb"
    image_color = "#f59e0b"
    ax1.plot(x, posts, linestyle="None", marker="_", markersize=12.0 * scale, markeredgewidth=2.1 * scale, color=tweet_color, zorder=3)
    ax1.plot(x, images, linestyle="None", marker="_", markersize=12.0 * scale, markeredgewidth=2.1 * scale, color=image_color, zorder=3)
    ax1.plot(x, posts, linestyle="None", marker="o", markersize=2.8 * scale, markerfacecolor="white", markeredgecolor=tweet_color, label="# Posts", zorder=4)
    ax1.plot(x, images, linestyle="None", marker="D", markersize=2.8 * scale, markerfacecolor="white", markeredgecolor=image_color, label="# Images", zorder=4)
    ax1.set_ylabel("# Posts / # Images")
    ax1.set_ylim(0, max(posts) * 1.12)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(250_000))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(human_readable))
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=24, ha="right")

    ax1.legend(
        [ax2.patches[0], Line2D([0], [0], color=tweet_color, marker="o", markerfacecolor="white"), Line2D([0], [0], color=image_color, marker="D", markerfacecolor="white")],
        ["# Users", "# Posts", "# Images"],
        loc="upper right",
        ncol=3,
        framealpha=0.88,
    )
    ax1.text(
        0.5,
        0.94,
        f"Total Users: {sum(users):,}\nTotal Posts: {sum(posts):,}\nTotal Images: {sum(images):,}",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=7.8 * scale,
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#d1d5db", alpha=0.92),
    )
    fig.tight_layout(pad=0.45)
    fig.savefig(base / "dataset_stats.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(base / "dataset_stats.png", bbox_inches="tight", dpi=300)
    print(f"wrote {base / 'dataset_stats.png'}")


if __name__ == "__main__":
    main()
