#!/usr/bin/env python3
"""Multimodal Twitter user interest labeling with sliding windows."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


CATEGORIES = [
    "Entertainment", "Sports", "Technology", "Business", "Education", "Gaming", "Politics", "Art", "Culture", "Science", "Food", "Daily Life", "Fashion & Beauty", "Health & Fitness", "News"
]
SYSTEM_PROMPT = "You are a precise content classification assistant that jointly analyzes text and images to assign interest labels."
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = "gpt-5.1"
CACHE_FORMAT_VERSION = 2
CACHE_META_KEY = "__meta__"
CACHE_INDEX_KEY = "index"


@dataclass
class LabelConfig:
    group_size: int = 30
    img_detail: str = "low"
    skip_retweets: bool = False
    dry_run: bool = False
    max_retries: int = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multimodal Twitter interest labeling")
    p.add_argument("--tweet-root", default="dataset/local/tweets", help="Tweet CSV root")
    p.add_argument("--img-root", default="dataset/local/images", help="Tweet image root")
    p.add_argument("--img-extra-roots", default="dataset/local/images_extra", help="Extra image roots, semicolon-separated")
    p.add_argument("--output", default="dataset/local/labeled_results.json", help="Merged output json path")
    p.add_argument("--per-user-dir", default="dataset/local/labeled_users", help="Per-user result directory")
    p.add_argument("--log-file", default="outputs/labeler.log", help="Detailed runtime log file")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL), help="OpenAI-compatible base URL")
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key (prefer env OPENAI_API_KEY)")
    p.add_argument("--max-users", type=int, default=50, help="Max number of users to process")
    p.add_argument("--group-size", type=int, default=30, help="Tweets per group")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--dry-run", action="store_true", help="Skip API calls and emit DRY_RUN labels")
    p.add_argument("--compare-models", action="store_true", help="Enable model comparison mode")
    p.add_argument("--test-models", nargs="*", default=["gpt-5.1", "gpt-5.2", "gpt-5.4"], help="Models for comparison")
    p.add_argument("--test-users", type=int, default=8, help="How many users for comparison")
    p.add_argument("--img-detail", choices=["low", "auto", "high"], default="low", help="Image detail in API payload")
    p.add_argument("--skip-retweets", action="store_true", help="Skip retweets")
    p.add_argument("--image-index-cache", default="dataset/local/image_index.json", help="Cache file for tweet_id -> image paths")
    p.add_argument("--build-index-only", action="store_true", help="Build image index then exit")
    p.add_argument("--force-rebuild-index", action="store_true", help="Force rebuild index cache")
    p.add_argument("--full-image-index", action="store_true", help="Build full image index for all users")
    p.add_argument("--max-size", type=int, default=512, help="Max image side length for base64 encoding")
    p.add_argument("--quality", type=int, default=75, help="JPEG quality for encoded images")
    p.add_argument("--merge-save-every", type=int, default=50, help="How many users between merged output saves (per-user files are always saved)")
    p.add_argument("--shard-count", type=int, default=1, help="Total shards for parallel runs (1 = no sharding)")
    p.add_argument("--shard-index", type=int, default=0, help="Shard index in [0, shard-count)")
    return p.parse_args()


def get_image_roots(img_root: str, img_extra_roots: str) -> list[str]:
    roots: list[str] = []
    for p in [img_root] + [x.strip() for x in img_extra_roots.split(";") if x.strip()]:
        if p and p not in roots:
            roots.append(p)
    return roots


def log_line(msg: str, log_path: str | None = None) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} UTC] {msg}"
    print(line)
    if log_path:
        try:
            p = Path(log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


@lru_cache(maxsize=2048)
def encode_image_to_base64(img_path: str, max_size: int = 512, quality: int = 75) -> str | None:
    try:
        with Image.open(img_path) as img:
            img.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[WARN] image encode failed: {img_path} | {exc}")
        return None


def build_image_index(
    img_roots: list[str],
    allowed_users: set[str] | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, list[str]]:
    log = logger or print
    index: dict[str, list[str]] = {}

    for root_path in img_roots:
        root = Path(root_path)
        if not root.exists():
            log(f"[WARN] image root missing: {root_path}")
            continue

        first_level_dirs = [p for p in root.iterdir() if p.is_dir()]
        # supports two layouts：
        # 1) root/batch_xxx_img/<user>/<tweetid_idx>.jpg
        # 2) root/<user>/<tweetid_idx>.jpg (e.g. batch_1000_8_img)
        if any(d.name.startswith("batch_") or d.name.endswith("_img") for d in first_level_dirs):
            batch_dirs = sorted(first_level_dirs)
        else:
            batch_dirs = [root]

        log(f"[INFO] scanning image roots: {root_path}, num_batches={len(batch_dirs)}")
        for i, batch_dir in enumerate(batch_dirs, start=1):
            if allowed_users is None:
                user_dirs = [p for p in batch_dir.iterdir() if p.is_dir()]
            else:
                user_dirs = [batch_dir / u for u in allowed_users if (batch_dir / u).is_dir()]

            for user_dir in user_dirs:
                try:
                    for img_file in user_dir.iterdir():
                        if not img_file.is_file():
                            continue
                        if img_file.suffix.lower() != ".jpg":
                            continue
                        stem = img_file.stem
                        if "_" not in stem:
                            continue
                        tweet_id = stem.rsplit("_", 1)[0]
                        index.setdefault(tweet_id, []).append(str(img_file))
                except Exception as exc:
                    log(f"[WARN] failed to scan directory: {user_dir} | {exc}")
            log(f"[INFO] image index progress: root={root.name} {i}/{len(batch_dirs)} batch, hit per-user directory={len(user_dirs)}, tweet_id={len(index)}")

    for tid in index:
        index[tid].sort()
    return index


def load_or_build_image_index(
    img_roots: list[str],
    cache_path: str,
    logger: Callable[[str], None],
    allowed_users: set[str] | None = None,
    force_rebuild: bool = False,
) -> dict[str, list[str]]:
    cache = Path(cache_path)

    def _load_cache_payload(payload: Any) -> tuple[dict[str, list[str]], dict[str, Any]]:
        if isinstance(payload, dict) and CACHE_INDEX_KEY in payload and isinstance(payload.get(CACHE_INDEX_KEY), dict):
            idx = payload.get(CACHE_INDEX_KEY, {})
            meta = payload.get(CACHE_META_KEY, {}) if isinstance(payload.get(CACHE_META_KEY), dict) else {}
            return {str(k): list(v) for k, v in idx.items()}, meta
        if isinstance(payload, dict):
            return {str(k): list(v) for k, v in payload.items()}, {}
        return {}, {}

    if allowed_users is None and cache.exists() and not force_rebuild:
        try:
            size_bytes = cache.stat().st_size
            with cache.open("r", encoding="utf-8") as f:
                data = json.load(f)
            index, meta = _load_cache_payload(data)
            if index:
                logger(f"[INFO] loaded image index cache: {cache_path}, size={size_bytes} bytes, keys={len(index)}")
                if meta:
                    logger(
                        "[INFO] cache meta: "
                        f"version={meta.get('format_version')}, "
                        f"built_at={meta.get('built_at_utc')}, "
                        f"img_root={meta.get('img_root')}, "
                        f"batch_count={meta.get('batch_count')}, "
                        f"tweet_id_count={meta.get('tweet_id_count')}"
                    )
                    old_roots = meta.get("img_roots", [])
                    cur_roots = [str(x) for x in img_roots]
                    if isinstance(old_roots, list) and old_roots and old_roots != cur_roots:
                        logger("[WARN] cache img_roots inconsistent with current args，tip --force-rebuild-index")
                else:
                    logger("[WARN] detected legacycache（without meta），rebuild into the new cache format")
                if len(index) < 100:
                    logger("[WARN] cachetoo few entries（<100），rebuild cache if results look wrong")
                return index
        except Exception as exc:
            logger(f"[WARN] failed to read cache，rebuild instead: {exc}")

    mode = "full" if allowed_users is None else f"local(num_users={len(allowed_users)})"
    logger(f"[INFO] building image index: {mode}")
    index = build_image_index(img_roots, allowed_users=allowed_users, logger=logger)

    if allowed_users is not None:
        logger("[INFO] local index mode：do not overwritefullcache")
        return index

    try:
        batch_count = 0
        for rp in img_roots:
            root = Path(rp)
            if root.exists():
                batch_count += len([p for p in root.iterdir() if p.is_dir()])

        payload = {
            CACHE_META_KEY: {
                "format_version": CACHE_FORMAT_VERSION,
                "built_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "img_root": img_roots[0] if img_roots else "",
                "img_roots": img_roots,
                "batch_count": batch_count,
                "tweet_id_count": len(index),
            },
            CACHE_INDEX_KEY: index,
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger(f"[INFO] image index cached to: {cache_path} (version={CACHE_FORMAT_VERSION}, keys={len(index)})")
    except Exception as exc:
        logger(f"[WARN] failed to write index cache: {exc}")
    return index


def load_and_group_tweets(csv_path: str, group_size: int = 30, skip_retweets: bool = False) -> list[list[dict[str, Any]]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "publish_time" not in df.columns:
        raise ValueError(f"CSV missing publish_time column: {csv_path}")

    if "tweet_id" not in df.columns:
        for col in df.columns:
            if col.strip("\ufeff") == "tweet_id":
                df = df.rename(columns={col: "tweet_id"})
                break

    if skip_retweets and "is_retweet" in df.columns:
        vals = df["is_retweet"].astype(str).str.lower()
        df = df[~vals.isin({"true", "1", "yes"})]

    df["dt"] = pd.to_datetime(
        df["publish_time"],
        format="%a %b %d %H:%M:%S +0000 %Y",
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["dt"]).sort_values("dt", ascending=False).reset_index(drop=True)

    groups: list[list[dict[str, Any]]] = []
    for i in range(0, len(df), group_size):
        groups.append(df.iloc[i : i + group_size].to_dict("records"))
    return groups


def get_labelable_windows(groups: list[list[dict[str, Any]]]) -> list[tuple[int, list[list[dict[str, Any]]]]]:
    windows: list[tuple[int, list[list[dict[str, Any]]]]] = []
    for t in range(len(groups)):
        windows.append((t, groups[t : t + 3]))
    return windows


def build_multimodal_prompt(
    window_groups: list[list[dict[str, Any]]],
    group_indices: list[int],
    img_detail: str,
    max_size: int,
    quality: int,
) -> tuple[list[dict[str, Any]], int]:
    categories_str = ", ".join(CATEGORIES)
    content: list[dict[str, Any]] = []
    content.append(
        {
            "type": "text",
            "text": (
                "You are a precise content classification assistant. Jointly analyze the "
                "user's posts (text + images) across up to three temporal windows and "
                "decide the single interest category that best matches the user right now.\n\n"
                f"Allowed categories (15):\n{categories_str}\n\n"
                "Window layout:\n"
                f"- Window A (group {group_indices[0]})\n"
                f"- Window B (group {group_indices[1] if len(group_indices) > 1 else 'N/A'})\n"
                f"- Window C (group {group_indices[2] if len(group_indices) > 2 else 'N/A'})\n\n"
                "Window contents follow:"
            ),
        }
    )

    labels = ["A", "B", "C"]
    image_count = 0
    for label, group in zip(labels, window_groups):
        content.append({"type": "text", "text": f"\n--- Window {label} ---"})
        for tweet in group:
            text_line = str(tweet.get("full_text", "")).strip()
            if text_line:
                content.append({"type": "text", "text": f"Post: {text_line}"})
            for img_path in tweet.get("image_paths", []):
                b64 = encode_image_to_base64(img_path, max_size=max_size, quality=quality)
                if not b64:
                    continue
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": img_detail,
                        },
                    }
                )
                image_count += 1

    content.append({"type": "text", "text": "\nReply with exactly one category name from the list above. Do not add any other text."})
    return content, image_count


def normalize_label(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    for cat in CATEGORIES:
        if cat in text:
            return cat
    return ""


def is_valid_label(label: str) -> bool:
    return label in CATEGORIES


class InsufficientBalanceError(RuntimeError):
    pass


def is_insufficient_balance_error(msg: str) -> bool:
    s = (msg or "").lower()
    return (
        "pre-deduct balance failed" in msg
        or "remaining balance" in msg
        or "balance" in msg
        or ("403" in s and ("quota" in s or "balance" in msg))
    )


def sanitize_windows(windows: Any, total_groups: int) -> list[dict[str, Any]]:
    if not isinstance(windows, list):
        return []

    out: list[dict[str, Any]] = []
    seen_t: set[int] = set()
    for w in windows:
        if not isinstance(w, dict):
            continue
        try:
            t = int(w.get("group_t"))
        except Exception:
            continue
        if t < 0 or t >= total_groups or t in seen_t:
            continue
        label = str(w.get("label", "")).strip()
        if not is_valid_label(label):
            continue
        seen_t.add(t)
        out.append(w)

    out.sort(key=lambda x: int(x.get("group_t", 0)))
    return out


def is_user_fully_labeled(one: dict[str, Any]) -> bool:
    total_groups = one.get("total_groups")
    if not isinstance(total_groups, int) or total_groups <= 0:
        return False
    valid = sanitize_windows(one.get("windows", []), total_groups)
    return len(valid) == total_groups


def load_user_result(per_user_dir: str, username: str) -> dict[str, Any] | None:
    p = Path(per_user_dir) / f"{username}.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def is_image_limit_error(msg: str) -> bool:
    s = msg.lower()
    return "image" in s and ("limit" in s or "too many" in s or "maximum" in s)


def is_risk_error(msg: str) -> bool:
    return "content exists risk" in msg.lower()


def should_retry_network(msg: str) -> bool:
    s = msg.lower()
    keys = ["timeout", "timed out", "connection", "temporarily", "rate limit", "429", "502", "503", "504"]
    return any(k in s for k in keys)


def call_model_with_retry(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_retries: int,
) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=20,
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            msg = str(exc)
            if is_insufficient_balance_error(msg):
                raise InsufficientBalanceError(msg[:240])
            if is_risk_error(msg):
                return "SKIPPED_RISK"
            if attempt < max_retries and should_retry_network(msg):
                sleep_s = min(2**attempt, 8)
                print(f"[WARN] network/rate-limit error，{sleep_s}s  then retry ({attempt}/{max_retries})")
                time.sleep(sleep_s)
                continue
            return f"Error: {msg[:160]}"
    return "Error: unknown"


def flatten_window_tweets(window: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group in window:
        out.extend(group)
    return out


def attach_images(window: list[list[dict[str, Any]]], image_index: dict[str, list[str]]) -> int:
    image_count = 0
    for group in window:
        for tweet in group:
            tid = str(tweet.get("tweet_id", "")).strip()
            paths = image_index.get(tid, [])
            tweet["image_paths"] = paths
            image_count += len(paths)
    return image_count


def cap_images_before_request(window: list[list[dict[str, Any]]], max_images: int = 50) -> int:
    # Tweets are sorted by publish_time descending, so earlier images are newer.
    tweets = flatten_window_tweets(window)
    refs: list[tuple[dict[str, Any], str]] = []
    for tw in tweets:
        for img in tw.get("image_paths", []):
            refs.append((tw, img))

    if len(refs) <= max_images:
        return len(refs)

    for tw in tweets:
        tw["image_paths"] = []

    for tw, img in refs[:max_images]:
        tw["image_paths"].append(img)

    return max_images


def truncate_images_for_retry(window: list[list[dict[str, Any]]], keep_ratio: float, max_images: int = 50) -> int:
    tweets = flatten_window_tweets(window)
    all_items: list[tuple[dict[str, Any], list[str]]] = []
    for tw in tweets:
        imgs = list(tw.get("image_paths", []))
        if imgs:
            all_items.append((tw, imgs))

    total_imgs = sum(len(imgs) for _, imgs in all_items)
    keep = max(1, int(total_imgs * keep_ratio)) if total_imgs > 0 else 0
    keep = min(keep, max_images)

    kept = 0
    for tw, imgs in all_items:
        if kept >= keep:
            tw["image_paths"] = []
            continue
        remaining = keep - kept
        tw["image_paths"] = imgs[:remaining]
        kept += len(tw["image_paths"])

    return kept


def label_user(
    csv_path: str,
    image_index: dict[str, list[str]],
    client: Any,
    model: str,
    cfg: LabelConfig,
    max_size: int,
    quality: int,
    logger: Callable[[str], None],
    existing_user: dict[str, Any] | None = None,
) -> dict[str, Any]:
    groups = load_and_group_tweets(csv_path, group_size=cfg.group_size, skip_retweets=cfg.skip_retweets)
    username = Path(csv_path).parent.name

    result: dict[str, Any] = {
        "csv_path": csv_path,
        "total_tweets": sum(len(g) for g in groups),
        "total_groups": len(groups),
        "windows": [],
    }

    if not groups:
        logger(f"[user {username}] empty groups, skip")
        return {username: result}

    restored = sanitize_windows((existing_user or {}).get("windows", []), len(groups))
    done_t = {int(w.get("group_t", -1)) for w in restored}
    if restored:
        logger(f"[user {username}] resume: finished windows={len(restored)}/{len(groups)}")
    result["windows"] = list(restored)

    logger(f"[user {username}] groups={len(groups)}，tweets={result['total_tweets']}")
    for t, window in get_labelable_windows(groups):
        if t in done_t:
            continue
        tweet_count = sum(len(g) for g in window)
        if tweet_count == 0:
            logger(f"[user {username}] window t={t} tweets=0，skip output")
            continue

        found_count = attach_images(window, image_index)
        window_indices = list(range(t, t + len(window)))
        logger(
            f"[user {username}] window t={t} groups={window_indices} tweets={tweet_count} images_found={found_count}"
        )

        kept_count = cap_images_before_request(window, max_images=50)
        if found_count > kept_count:
            logger(f"[WARN] window t={t} too many images, capped before request: found={found_count}, kept={kept_count}")

        if cfg.dry_run:
            label = ""
            encoded_count = kept_count
        else:
            content, encoded_count = build_multimodal_prompt(
                window,
                window_indices,
                img_detail=cfg.img_detail,
                max_size=max_size,
                quality=quality,
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]

            raw = call_model_with_retry(client, model, messages, cfg.max_retries)

            if isinstance(raw, str) and is_image_limit_error(raw):
                truncated = truncate_images_for_retry(window, keep_ratio=0.5, max_images=50)
                content2, encoded_count = build_multimodal_prompt(
                    window,
                    window_indices,
                    img_detail=cfg.img_detail,
                    max_size=max_size,
                    quality=quality,
                )
                raw = call_model_with_retry(
                    client,
                    model,
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content2},
                    ],
                    cfg.max_retries,
                )
                logger(f"[WARN] image limit hit; retry with fewer images, kept= {truncated}")

            label = normalize_label(raw)

        if label not in CATEGORIES:
            logger(f"[WARN] [user {username}] window t={t} label not in 15 classes, dropped")
            continue

        logger(f"[user {username}] window t={t} label={label}")

        result["windows"].append(
            {
                "group_t": t,
                "window": window_indices,
                "window_size": len(window),
                "tweet_count": tweet_count,
                "found_image_count": found_count,
                "encoded_image_count": encoded_count,
                "image_count": encoded_count,
                "label": label,
            }
        )

    result["windows"] = sanitize_windows(result.get("windows", []), len(groups))
    return {username: result}


def list_all_csvs(tweet_root: str) -> list[str]:
    root = Path(tweet_root)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.glob("batch_*/*/*_full_tweets.csv"))


def load_existing_results(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"[WARN] failed to read existing results，will create new output: {exc}")
    return {}


def save_results(path: str, data: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def save_user_result(per_user_dir: str, username: str, data: dict[str, Any]) -> None:
    out_dir = Path(per_user_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{username}.json"
    tmp = out.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(out)


def load_processed_users_from_dir(per_user_dir: str) -> set[str]:
    p = Path(per_user_dir)
    if not p.exists():
        return set()
    users: set[str] = set()
    for f in p.glob("*.json"):
        try:
            with f.open("r", encoding="utf-8") as fh:
                one = json.load(fh)
            if isinstance(one, dict) and is_user_fully_labeled(one):
                users.add(f.stem)
        except Exception:
            continue
    return users


def choose_users(
    csv_paths: list[str],
    max_users: int,
    seed: int,
    processed: set[str],
    shard_count: int = 1,
    shard_index: int = 0,
) -> list[str]:
    candidates = [p for p in csv_paths if Path(p).parent.name not in processed]
    if shard_count > 1:
        shard_filtered: list[str] = []
        for p in candidates:
            username = Path(p).parent.name
            bucket = int(hashlib.md5(username.encode("utf-8")).hexdigest(), 16) % shard_count
            if bucket == shard_index:
                shard_filtered.append(p)
        candidates = shard_filtered
    random.Random(seed).shuffle(candidates)
    return candidates[:max_users] if max_users > 0 else candidates


def build_client(api_key: str, base_url: str) -> Any:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed，please run: pip install openai")
    if not api_key:
        raise RuntimeError("API key missing; pass --api-key or set OPENAI_API_KEY")
    return OpenAI(api_key=api_key, base_url=base_url)


def classify_window_for_model(
    window: list[list[dict[str, Any]]],
    group_t: int,
    model: str,
    client: Any,
    cfg: LabelConfig,
    max_size: int,
    quality: int,
    logger: Callable[[str], None] | None = None,
) -> str:
    # Work on a shallow-cloned window so truncation retry won't affect other models.
    cloned_window: list[list[dict[str, Any]]] = []
    for group in window:
        cloned_group: list[dict[str, Any]] = []
        for tw in group:
            tw2 = dict(tw)
            tw2["image_paths"] = list(tw.get("image_paths", []))
            cloned_group.append(tw2)
        cloned_window.append(cloned_group)

    window_indices = list(range(group_t, group_t + len(window)))
    found_count = sum(len(tw.get("image_paths", [])) for tw in flatten_window_tweets(cloned_window))
    kept_count = cap_images_before_request(cloned_window, max_images=50)
    if logger and found_count > kept_count:
        logger(f"[CMP] model={model} windowt={group_t} pre-cap images: found={found_count}, kept={kept_count}")

    content, _ = build_multimodal_prompt(cloned_window, window_indices, cfg.img_detail, max_size=max_size, quality=quality)
    raw = call_model_with_retry(
        client,
        model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        cfg.max_retries,
    )

    if isinstance(raw, str) and is_image_limit_error(raw):
        kept = truncate_images_for_retry(cloned_window, keep_ratio=0.5, max_images=50)
        content2, _ = build_multimodal_prompt(
            cloned_window,
            window_indices,
            cfg.img_detail,
            max_size=max_size,
            quality=quality,
        )
        raw = call_model_with_retry(
            client,
            model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content2},
            ],
            cfg.max_retries,
        )
        if logger:
            logger(f"[CMP] model={model} windowt={group_t} too many images，retry after truncation，kept images={kept}")

    return normalize_label(raw)


def run_model_comparison(
    users: list[str],
    image_index: dict[str, list[str]],
    models: list[str],
    client: Any,
    cfg: LabelConfig,
    output_path: str,
    max_size: int,
    quality: int,
    logger: Callable[[str], None] | None = None,
) -> None:
    rows: list[dict[str, Any]] = []

    for i, csv_path in enumerate(users, start=1):
        username = Path(csv_path).parent.name
        groups = load_and_group_tweets(csv_path, group_size=cfg.group_size, skip_retweets=cfg.skip_retweets)
        if not groups:
            continue

        for t in range(2, min(5, len(groups))):
            window = groups[t : t + 3]
            attach_images(window, image_index)

            labels: dict[str, str] = {}
            for model in models:
                if cfg.dry_run:
                    labels[model] = "DRY_RUN"
                else:
                    labels[model] = classify_window_for_model(
                        window,
                        group_t=t,
                        model=model,
                        client=client,
                        cfg=cfg,
                        max_size=max_size,
                        quality=quality,
                        logger=logger,
                    )

            uniq = set(labels.values())
            row = {
                "user_id": username,
                "window_t": t,
                "all_match": len(uniq) == 1,
            }
            for model in models:
                row[f"{model}_label"] = labels.get(model, "")
            rows.append(row)
            print(f"[CMP] user={username} window={t} consistent={row['all_match']} labels={labels}")

        print(f"[CMP] progress: {i}/{len(users)}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[INFO] model comparison written to: {output_path}")


def main() -> int:
    args = parse_args()
    logger = lambda msg: log_line(msg, args.log_file)

    if args.shard_count < 1:
        logger("[ERROR] --shard-count must be >= 1")
        return 1
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        logger("[ERROR] --shard-index must satisfy 0 <= shard-index < shard-count")
        return 1

    cfg = LabelConfig(
        group_size=args.group_size,
        img_detail=args.img_detail,
        skip_retweets=args.skip_retweets,
        dry_run=args.dry_run,
        max_retries=3,
    )

    csv_paths = list_all_csvs(args.tweet_root)
    if not csv_paths:
        logger(f"[ERROR] not found CSV: {args.tweet_root}")
        return 1

    # Index-only mode: build cache and exit.
    if args.build_index_only:
        img_roots = get_image_roots(args.img_root, args.img_extra_roots)
        logger(f"[INFO] image roots this run: {img_roots}")
        build_allowed_users = None if args.full_image_index else {Path(p).parent.name for p in csv_paths}
        image_index = load_or_build_image_index(
            img_roots,
            args.image_index_cache,
            logger=logger,
            allowed_users=build_allowed_users,
            force_rebuild=args.force_rebuild_index,
        )
        logger(f"[INFO] image index ready，tweet_id count={len(image_index)}")
        logger("[INFO] --build-index-only mode finished")
        return 0

    existing = load_existing_results(args.output)
    processed = load_processed_users_from_dir(args.per_user_dir)
    logger(f"[INFO] finished users（merged aggregate results+per-user directory）={len(processed)}")

    selected = choose_users(
        csv_paths,
        args.max_users,
        args.seed,
        processed,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    if not selected:
        logger("[INFO] no pending users (maybe all done)")
        return 0
    logger(f"[INFO] users to process this run={len(selected)}")

    cmp_users = selected[: args.test_users] if args.compare_models else []
    index_users = {Path(p).parent.name for p in (cmp_users if args.compare_models else selected)}
    index_allowed_users = None if args.full_image_index else index_users
    img_roots = get_image_roots(args.img_root, args.img_extra_roots)
    logger(f"[INFO] image roots this run: {img_roots}")
    image_index = load_or_build_image_index(
        img_roots,
        args.image_index_cache,
        logger=logger,
        allowed_users=index_allowed_users,
        force_rebuild=args.force_rebuild_index,
    )
    logger(f"[INFO] image index ready，tweet_id count={len(image_index)}")

    client = None
    if not args.dry_run:
        client = build_client(args.api_key, args.base_url)

    if args.compare_models:
        run_model_comparison(
            users=cmp_users,
            image_index=image_index,
            models=args.test_models,
            client=client,
            cfg=cfg,
            output_path=str(Path(args.output).with_name("model_comparison_results.json")),
            max_size=args.max_size,
            quality=args.quality,
            logger=logger,
        )
        logger("[INFO] model comparison done")
        return 0

    total = len(selected)
    merged = dict(existing)
    for i, csv_path in enumerate(selected, start=1):
        username = Path(csv_path).parent.name
        try:
            existing_user = load_user_result(args.per_user_dir, username)
            one = label_user(
                csv_path=csv_path,
                image_index=image_index,
                client=client,
                model=args.model,
                cfg=cfg,
                max_size=args.max_size,
                quality=args.quality,
                logger=logger,
                existing_user=existing_user,
            )
            save_user_result(args.per_user_dir, username, one[username])
            if is_user_fully_labeled(one[username]):
                merged[username] = one[username]
            else:
                merged.pop(username, None)
        except InsufficientBalanceError as exc:
            logger(f"[ERROR] APIinsufficient balance，stopping: {exc}")
            save_results(args.output, merged)
            return 2
        except Exception as exc:
            logger(f"[ERROR] user processing failed: {username} | {exc}")
            # do not write error object; keep resumable state，wait then retry。

        # Per-user file is the primary checkpoint. Merged JSON is flushed in batches for performance.
        if i % args.merge_save_every == 0 or i == total:
            save_results(args.output, merged)
            logger(f"[INFO] mergedresults saved: {i}/{total}")

        if i % 10 == 0 or i == total:
            pct = 100.0 * i / total
            logger(f"[INFO] progress: {i}/{total} ({pct:.2f}%)")

    logger(f"[INFO] all done，output: {args.output}，per-user directory: {args.per_user_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
