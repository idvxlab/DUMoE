#!/usr/bin/env python3
"""Train / validation / test split used in the paper.

Posts of each user are ordered by time (newest first). Consecutive posts are
grouped into non-overlapping windows of ``WINDOW_SIZE`` posts. Splits are
assigned by window index, not by randomly partitioning users:

  window_idx 0 .. 7   -> train   (8 windows)
  window_idx 8 .. 9   -> val     (2 windows)
  window_idx 10 .. 11 -> test    (2 windows)

Windows with index >= 12 are unused.
"""
from __future__ import annotations

WINDOW_SIZE = 30
TRAIN_WINDOWS = 8
VAL_WINDOWS = 2
TEST_WINDOWS = 2


def split_tag_from_window(
    window_idx: int,
    train_windows: int = TRAIN_WINDOWS,
    val_windows: int = VAL_WINDOWS,
    test_windows: int = TEST_WINDOWS,
) -> str:
    idx = int(window_idx)
    if idx < 0:
        return "other"
    if idx < train_windows:
        return "train"
    if idx < train_windows + val_windows:
        return "val"
    if idx < train_windows + val_windows + test_windows:
        return "test"
    return "other"


def window_idx_from_position(position: int, window_size: int = WINDOW_SIZE) -> int:
    """``position`` is 0-based in the newest-first post sequence."""
    if int(position) < 0:
        raise ValueError("position must be >= 0")
    return int(position) // int(window_size)
