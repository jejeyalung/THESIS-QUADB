"""
Step 1: preprocess raw reviews (works on ANY csv with a 'review' column —
training data with labels, or a business owner's raw unlabeled export).

What this does:
  1. Load a CSV that has at least a 'review' column
  2. Clean the review text: lowercase, collapse repeated characters,
     standardize a few common Taglish spelling variants
  3. [TEMPLATE, NOT YET APPLIED] Flag/remove spam or irrelevant reviews
  4. Save review + review_clean only — no label handling here.
     Label mapping is training-data-specific and lives in the sentiment
     analysis script, not here, so this script stays reusable for the
     practical pipeline (business owner CSVs have no sentiment column).

Run it:
    python3 src/preprocess.py --input data/SentiTaglish_ProductsAndServices.csv --output data/senti_taglish_clean.csv
    python3 src/preprocess.py --input data/some_business_reviews.csv --output data/some_business_reviews_clean.csv
"""

import argparse
import re
import unicodedata
import pandas as pd

# --- Taglish spelling normalization ------------------------------------------
# Starter set. Extend this as you read more of the actual reviews.
SPELLING_MAP = {
    "d2": "dito", "dto": "dito",
    "un": "yun",
    "sya": "siya", "cya": "siya",
    "nde": "hindi", "hnd": "hindi",
    "wla": "wala", "wlang": "walang",
    "eto": "ito",
    "pde": "pwede", "pwd": "pwede",
    "gud": "good",
    "thnx": "thanks", "tnx": "thanks",
    "salamt": "salamat",
}

_REPEATED_CHAR = re.compile(r"(.)\1{2,}")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _REPEATED_CHAR.sub(r"\1\1", text)          # "gandaaaa" -> "gandaa"
    tokens = text.split()
    tokens = [SPELLING_MAP.get(t, t) for t in tokens]
    text = " ".join(tokens)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ------------------------------------------------------------------------
# TEMPLATE — spam / irrelevant review filtering
#
# Table 1 of the proposal names "Removal of spam, irrelevant reviews, etc."
# as a data-quality-filtering step but never defines what counts as spam or
# how it's detected. Nothing below is applied yet — this is scaffolding to
# fill in once you decide what "spam" means for this dataset (e.g. run the
# audit script first and look at what actually shows up).
#
# Ideas to evaluate, uncomment/implement once decided:
#
# def is_too_short(text: str, min_words: int = 3) -> bool:
#     """Flag reviews with fewer than min_words words (e.g. 'ok', 'nice po')."""
#     return len(text.split()) < min_words
#
# def is_duplicate(df: pd.DataFrame, text_col: str = "review_clean") -> pd.Series:
#     """Flag exact-duplicate reviews (copy-pasted spam, bulk-review farms)."""
#     return df.duplicated(subset=[text_col], keep="first")
#
# def is_low_alpha_ratio(text: str, min_ratio: float = 0.5) -> bool:
#     """Flag reviews that are mostly non-alphabetic (emoji spam, symbol spam)."""
#     if not text:
#         return True
#     alpha_chars = sum(c.isalpha() for c in text)
#     return (alpha_chars / len(text)) < min_ratio
#
# def flag_spam(df: pd.DataFrame) -> pd.DataFrame:
#     """Apply the checks above and add a boolean 'is_spam' column instead of
#     silently dropping rows — review the flagged rows manually before
#     deciding to actually filter them out."""
#     df = df.copy()
#     df["is_spam"] = (
#         df["review_clean"].apply(is_too_short)
#         | is_duplicate(df)
#         | df["review_clean"].apply(is_low_alpha_ratio)
#     )
#     return df
# ------------------------------------------------------------------------


def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if "review" not in df.columns:
        raise KeyError(
            f"Expected a 'review' column in {input_path}. "
            f"Found columns: {list(df.columns)}"
        )

    print(f"Loaded {len(df)} rows from {input_path}")

    df["review_clean"] = df["review"].apply(clean_text)
    before = len(df)
    df = df[df["review_clean"] != ""]  # drop rows that are empty after cleaning
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows that were empty after cleaning")

    # this is where spam and irrelevant shit if template is implemented

    out = df[["review", "review_clean"]]
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out)} cleaned rows to {output_path}")

    print("\nSample before/after:")
    for _, row in out.sample(min(3, len(out)), random_state=42).iterrows():
        print("RAW :", row["review"][:100])
        print("CLEAN:", row["review_clean"][:100])
        print()

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/SentiTaglish_ProductsAndServices.csv")
    parser.add_argument("--output", default="data/senti_taglish_clean.csv")
    args = parser.parse_args()

    preprocess(args.input, args.output)