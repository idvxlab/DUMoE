"""Repo-relative paths. No lab machine names."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "dataset"
IDS_DIR = DATASET_DIR / "ids"
STATS_DIR = DATASET_DIR / "stats"
LABELS_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = REPO_ROOT / "outputs"
CACHE_DIR = REPO_ROOT / "cache"
LOCAL_DATA_DIR = DATASET_DIR / "local"
DOWNSTREAM_DIR = REPO_ROOT / "downstream"


def p(*parts: str) -> str:
    return str(REPO_ROOT.joinpath(*parts))
