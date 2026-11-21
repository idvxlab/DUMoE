# Preprocessing scripts

These are the scripts we actually used. Start from [`../README.md`](../README.md) if you just need the story.

| Step | Script |
| --- | --- |
| Split seed users into 15 topics | `user_filtering/prepare_seed_author_lists.py` |
| Crawl settings (English, min length 50) | `user_filtering/crawl_hyperparams.py` |
| Keep users with ≤ 5000 followers | `user_filtering/filter_users.py` |
| Keep posts in the collection dates | `user_filtering/batch_migrate_and_filter.py` |
| Sort newest-first, groups of 30 | `windowing/group_tweets_by_date.py` |
| VLM window labels | `labeling/label_interest_windows.py` |
| Prompts the VLM saw | `prompts/` |
