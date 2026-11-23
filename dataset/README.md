# DUMoE dataset

**Dataset description.** We constructed a large-scale multimodal user dataset from publicly available posts on X using the official X API between November 2025 and March 2026. The dataset comprises 14,015 users and 7,685,700 posts, including 2,890,668 posts associated with images. The collected data cover 15 interest domains: Entertainment, Sports, Technology, Business, Education, Gaming, Politics, Art, Culture, Science, Food, Daily Life, Fashion & Beauty, Health & Fitness, and News. Each user is represented by chronologically collected posts, providing textual and visual content for studying temporal user interests and multimodal user representation. To facilitate reproducibility while respecting the platform's content redistribution requirements, the publicly released dataset is designed to provide post and user identifiers rather than the original post text, images, or other hydrated X content. Researchers can retrieve the corresponding publicly available content through the official X API, subject to X's applicable terms and policies.

**If you have any questions about the dataset details, please feel free to contact us.**

## Download the ids

```bash
# from the DUMoE repo root
unzip dataset/ids/dumoe_ids.zip -d dataset/ids
```

| File inside the zip | What it is |
| --- | --- |
| `users.csv` | Column `user_id`: X **numeric** user id. No usernames. |
| `posts.csv` | Columns `user_id,post_id`. Posts of each user are listed **newest first**. |
| `windows.csv` | Columns `user_id,window_idx,label_id,split_tag`. **This is the labeling release** (one interest label per 30-post window). |

Tiny previews (open these in the browser / editor): [`ids/users.sample.csv`](ids/users.sample.csv), [`ids/posts.sample.csv`](ids/posts.sample.csv), [`ids/windows.sample.csv`](ids/windows.sample.csv).

There is **no post text and no image files** here. To get the actual posts:

1. Use `user_id` / `post_id` with the [official X API](https://developer.x.com/).  
2. Follow X’s developer terms.  
3. Keep the result on your machine (do not re-upload full tweet dumps to this repo).

## Numbers

| | Count |
| --- | ---: |
| Users | 14,015 |
| Posts | 7,685,700 |
| Posts with images | 2,890,668 |
| Interest classes | 15 |
| Collection time | November 2025 – March 2026 |

Per-class counts: [`stats/dataset_stats.csv`](stats/dataset_stats.csv). Figure: [`stats/dataset_stats.png`](stats/dataset_stats.png).

## 15 interest labels

`label_id` in `windows.csv` matches this list (`labels/interest_categories.json`):

| id | name |
| --- | --- |
| 0 | Entertainment |
| 1 | Sports |
| 2 | Technology |
| 3 | Business |
| 4 | Education |
| 5 | Gaming |
| 6 | Politics |
| 7 | Art |
| 8 | Culture |
| 9 | Science |
| 10 | Food |
| 11 | Daily Life |
| 12 | Fashion & Beauty |
| 13 | Health & Fitness |
| 14 | News |

## How we built windows and splits

Think of one user as a timeline of posts.

1. Sort that user’s posts **from new to old**.  
2. Cut them into blocks of **30 posts**. Block `0` is the newest 30, block `1` the next 30, and so on.  
3. Assign the split from the block index (same user can appear in train, val, and test at different times):

| `window_idx` | Split (`split_tag`) |
| --- | --- |
| 0, 1, 2, 3, 4, 5, 6, 7 | `train` |
| 8, 9 | `val` |
| 10, 11 | `test` |
| 12 and above | unused |

The function is `splits/assign_split.py`.

## How we chose users and labels

**Users.** We started from topic seed accounts, crawled English timelines, and kept non-celebrity accounts (**followers ≤ 5,000**). Collection roughly covers late 2025 to March 2026. Scripts: `preprocessing/user_filtering/`.

**Windows of 30 posts.** `preprocessing/windowing/group_tweets_by_date.py`.

**Labels.** A vision-language model looks at each window `t` together with the next two windows (`t`, `t+1`, `t+2`) — text plus images — and must answer with **exactly one** of the 15 names above. The exact English prompts are in `preprocessing/prompts/`. The runnable script is `preprocessing/labeling/label_interest_windows.py`.


