"""
Step 1: preprocess raw reviews (works on ANY csv with a 'review' column —
training data with labels, or a business owner's raw unlabeled export).

What this does:
  1. Load a CSV that has at least a 'review' column
  2. Clean the review text with clean_text() — the SAME cleaning used to train
     the Stage 1 / Stage 2 models:
       a. Unicode normalization (NFKC)
       b. Lowercasing
       c. Repeated-character collapse (3+ of the same char -> 2)
       d. Taglish spelling normalization (16 common variants, exact word match)
       e. Whitespace cleanup
  3. Drop rows that are empty after cleaning
  4. Save review + review_clean only — no label handling here.
     Label mapping is training-data-specific and lives in the training
     notebooks, so this script stays reusable for the practical pipeline
     (business owner CSVs have no sentiment column).

Run it:
    python src/preprocess.py --input data/SentiTaglish_ProductsAndServices.csv --output data/senti_taglish_clean.csv
    python src/preprocess.py --input data/some_business_reviews.csv --output data/some_business_reviews_clean.csv

IMPORTANT: clean_text() must stay identical to clean_text() in the four
training notebooks. The deployed models only saw text cleaned this way.
Changing it means re-cleaning the training data and retraining both stages.

clean_text_extended() below holds additional rules that were explored but
NOT used to train the models. It is kept for future work only and is never
called by the CLI unless --extended is passed explicitly.
"""

import argparse
import re
import unicodedata

import pandas as pd


# =============================================================================
# Training cleaner — DO NOT CHANGE (must match the training notebooks exactly)
# =============================================================================
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

_REPEATED_CHAR = re.compile(r"(.)\1{2,}")   # "gandaaaa" -> "gandaa", "!!!!" -> "!!"
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Cleaning used for training and inference of the Stage 1 / Stage 2 models."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _REPEATED_CHAR.sub(r"\1\1", text)
    tokens = [SPELLING_MAP.get(t, t) for t in text.split()]
    text = " ".join(tokens)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# =============================================================================
# EXPERIMENTAL extended cleaner — future work, NOT used by the trained models
# =============================================================================
# Larger Taglish map chosen from frequent variants in the SentiTaglish reviews.
# Removed on purpose: "q" (usually "ko", not "ako"), "d" (also sizes/grades),
# and ambiguous short forms like "ml", "mg", "tsk", "rn".
EXTENDED_SPELLING_MAP = {
    **SPELLING_MAP,
    # demonstratives / location
    "yon": "yun",
    "ung": "yung", "yong": "yung", "yng": "yung",
    "ganon": "ganun", "gnun": "ganun",
    # pronouns / particles
    "syang": "siyang",
    "nya": "niya", "nyo": "niyo",
    "aq": "ako",
    "poh": "po",
    "saken": "sakin", "sken": "sakin", "skn": "sakin",
    # negation
    "nd": "hindi", "ndi": "hindi", "hndi": "hindi", "hinde": "hindi", "hnde": "hindi",
    "dko": "hindi ko",
    # common short forms
    "lng": "lang",
    "nlng": "nalang", "nlang": "nalang", "nalng": "nalang",
    "namn": "naman", "nmn": "naman", "nman": "naman",
    "kc": "kasi", "kse": "kasi", "kase": "kasi", "ksi": "kasi",
    "sna": "sana",
    "pra": "para",
    "tlga": "talaga", "tlaga": "talaga", "talga": "talaga", "tlg": "talaga",
    "tpos": "tapos", "tps": "tapos", "tas": "tapos",
    "mganda": "maganda", "mgnda": "maganda", "magnda": "maganda",
    "panget": "pangit",
    "kht": "kahit",
    "bkt": "bakit",
    "dhl": "dahil",
    "dw": "daw",
    "dpt": "dapat",
    "dn": "din",
    "kng": "kung",
    "kya": "kaya",
    "tska": "tsaka",
    # English slang / texting
    "pls": "please",
}

# Whole-word match that also works with attached punctuation ("sya,", "lng.").
# Not matched when glued to word chars, apostrophes, slashes or hyphens.
_EXT_SPELLING_RE = re.compile(
    r"(?<![\w'/-])("
    + "|".join(re.escape(k) for k in sorted(EXTENDED_SPELLING_MAP, key=len, reverse=True))
    + r")(?![\w'/-])"
)
_INVISIBLE = re.compile(r"[​-‍⁠﻿￿￼]")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_STRETCHED_END = re.compile(r"\b(?!paa\b)([a-z]+)([aui])\2\b")   # "gandaa" -> "ganda"
_SPACED_DOTS = re.compile(r"\.(?:\s*\.)+")                        # ". . ." -> ".."
_GLUED_PUNCT = re.compile(r"(?<=[a-z])([.,!?]+)(?=[a-z])(?!(?:com|ph|net|org|co)\b)")


def clean_text_extended(text: str) -> str:
    """Experimental cleaner. Models were NOT trained on this output."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = text.translate(_QUOTES)
    text = text.lower()
    text = _URL.sub(" ", text)
    text = _REPEATED_CHAR.sub(r"\1\1", text)
    text = _STRETCHED_END.sub(r"\1\2", text)
    text = _SPACED_DOTS.sub("..", text)
    text = _GLUED_PUNCT.sub(r"\1 ", text)
    text = _EXT_SPELLING_RE.sub(lambda m: EXTENDED_SPELLING_MAP[m.group(1)], text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# =============================================================================
# TEMPLATE — spam / irrelevant review filtering (NOT applied)
# =============================================================================
# The training data was not spam-filtered. Kept as scaffolding for future work.
#
# def is_too_short(text: str, min_words: int = 3) -> bool:
#     return len(text.split()) < min_words
#
# def is_duplicate(df: pd.DataFrame, text_col: str = "review_clean") -> pd.Series:
#     return df.duplicated(subset=[text_col], keep="first")
#
# def is_low_alpha_ratio(text: str, min_ratio: float = 0.5) -> bool:
#     if not text:
#         return True
#     return sum(c.isalpha() for c in text) / len(text) < min_ratio
#
# def flag_spam(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["is_spam"] = (
#         df["review_clean"].apply(is_too_short)
#         | is_duplicate(df)
#         | df["review_clean"].apply(is_low_alpha_ratio)
#     )
#     return df


def preprocess(input_path: str, output_path: str, extended: bool = False) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    if "review" not in df.columns:
        raise KeyError(
            f"Expected a 'review' column in {input_path}. "
            f"Found columns: {list(df.columns)}"
        )

    print(f"Loaded {len(df)} rows from {input_path}")

    if extended:
        print("WARNING: using clean_text_extended(). The trained models were NOT "
              "trained on this cleaning — use for experiments only.")
        cleaner = clean_text_extended
    else:
        cleaner = clean_text

    df["review_clean"] = df["review"].apply(cleaner)
    before = len(df)
    df = df[df["review_clean"] != ""]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows that were empty after cleaning")

    out = df[["review", "review_clean"]]
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out)} cleaned rows to {output_path}")

    print("\nSample before/after:")
    for _, row in out.sample(min(3, len(out)), random_state=42).iterrows():
        print("RAW  :", str(row["review"])[:100])
        print("CLEAN:", row["review_clean"][:100])
        print()

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/SentiTaglish_ProductsAndServices.csv")
    parser.add_argument("--output", default="data/senti_taglish_clean.csv")
    parser.add_argument("--extended", action="store_true",
                        help="Use the experimental cleaner (models were NOT trained on it).")
    args = parser.parse_args()

    preprocess(args.input, args.output, extended=args.extended)