# Labeling prompts

Copied from `../labeling/label_interest_windows.py`.

**System** (`system_prompt.txt`): tell the model it must classify from text and images.

**User** (`user_prompt_template.txt`): show up to three windows (A/B/C) of 30 posts each, attach images, ask for **one** category name from `categories.txt`.

The model reply is kept only if it matches one of those 15 English names.
