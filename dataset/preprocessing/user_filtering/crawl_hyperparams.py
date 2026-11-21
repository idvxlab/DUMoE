#!/usr/bin/env python3
"""Timeline crawl hyper-parameters used when collecting user histories.

Account cookies / tokens are intentionally omitted from this release.
"""

MAX_USER_CONCURRENCY = 5
GLOBAL_DL_CONCURRENCY = 24
MAX_CRAWL_LIMIT = 30
MIN_TEXT_LENGTH = 50
LANG_FILTER = "en"

TOPICS = (
    "art",
    "business",
    "culture",
    "education",
    "entertainment",
    "fashion",
    "food",
    "gaming",
    "health",
    "daily_life",
    "news",
    "politics",
    "science",
    "sports",
    "technology",
)
