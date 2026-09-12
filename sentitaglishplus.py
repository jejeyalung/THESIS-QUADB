"""
Taglish Review Analyzer - CLI tool

Takes a CSV of raw customer reviews (just needs a 'review' column, no labels)
and produces sentiment + dissatisfaction-category predictions for each one.

Pipeline:
  1. Clean text (same preprocessing used during training - see preprocess.py)
  2. Stage 1 model predicts sentiment: negative / neutral / positive / mixed
  3. For reviews predicted negative, Stage 2 model predicts which of 12
     dissatisfaction categories apply (a review can have more than one)
  4. Writes an output CSV with all predictions, plus a console summary

Usage:
    python3 cli_analyze.py --input reviews.csv --output results.csv

Requires trained models to already exist locally:
    models/stage1_sentiment/     (config.json, model.safetensors, tokenizer files)
    models/stage2_dissatisfaction/  (same structure)
Download these from Kaggle after training - see notebooks/ for how they were made.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import clean_text

STAGE1_MODEL_DIR = "models/stage1_sentiment"
STAGE2_MODEL_DIR = "models/stage2_dissatisfaction"
THRESHOLDS_PATH = "config/stage2_thresholds.json"

STAGE1_LABELS = ["negative", "neutral", "positive", "mixed"]
CATEGORY_COLUMNS = ["def", "dam", "perf", "pmq", "lst", "var", "mi",
                    "auth", "pss", "del", "pack", "val"]

# Categories known to be unreliable due to limited training data. Flagged
# in output rather than silently presented as equally trustworthy.
LOW_CONFIDENCE_CATEGORIES = {"auth"}

MAX_LENGTH = 128


def load_thresholds(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # strip metadata keys, keep only actual category thresholds
    return {k: v for k, v in data.items() if k in CATEGORY_COLUMNS}


def load_model(model_dir: str, device: str):
    if not Path(model_dir).exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Download the trained model files from Kaggle and place them here first."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_stage1(reviews_clean: list, tokenizer, model, device: str) -> list:
    """Returns a list of predicted sentiment label strings."""
    predictions = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(reviews_clean), batch_size):
            batch = reviews_clean[i:i + batch_size]
            inputs = tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(STAGE1_LABELS[i] for i in pred_ids)
    return predictions


def predict_stage2(reviews_clean: list, tokenizer, model, device: str, thresholds: dict) -> list:
    """Returns a list of dicts, one per review, mapping category -> 0/1."""
    results = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(reviews_clean), batch_size):
            batch = reviews_clean[i:i + batch_size]
            inputs = tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            for row_probs in probs:
                row_result = {}
                for j, cat in enumerate(CATEGORY_COLUMNS):
                    row_result[cat] = int(row_probs[j] >= thresholds[cat])
                results.append(row_result)
    return results


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print("\nSentiment breakdown:")
    counts = df["predicted_sentiment"].value_counts()
    for label, count in counts.items():
        pct = 100 * count / len(df)
        print(f"  {label:10s}: {count:5d} ({pct:.1f}%)")

    negative_df = df[df["predicted_sentiment"] == "negative"]
    if len(negative_df) == 0:
        print("\nNo negative reviews found - no dissatisfaction breakdown to show.")
        return

    print(f"\nDissatisfaction categories among {len(negative_df)} negative reviews:")
    cat_counts = []
    for cat in CATEGORY_COLUMNS:
        count = int(negative_df[cat].sum())
        pct = 100 * count / len(negative_df)
        flag = "  [LOW CONFIDENCE - limited training data]" if cat in LOW_CONFIDENCE_CATEGORIES else ""
        cat_counts.append((cat, count, pct, flag))

    cat_counts.sort(key=lambda x: -x[1])
    for cat, count, pct, flag in cat_counts:
        print(f"  {cat:6s}: {count:5d} ({pct:.1f}%){flag}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Taglish product reviews for sentiment and dissatisfaction reasons.")
    parser.add_argument("--input", required=True, help="Path to input CSV with a 'review' column")
    parser.add_argument("--output", required=True, help="Path to write the output CSV")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading input from {args.input} ...")
    df = pd.read_csv(args.input)
    if "review" not in df.columns:
        raise KeyError(f"Expected a 'review' column. Found: {list(df.columns)}")

    print(f"Loaded {len(df)} reviews")
    df["review_clean"] = df["review"].apply(clean_text)
    before = len(df)
    df = df[df["review_clean"] != ""].reset_index(drop=True)
    if before != len(df):
        print(f"Dropped {before - len(df)} rows that were empty after cleaning")

    thresholds = load_thresholds(THRESHOLDS_PATH)

    print("\nLoading Stage 1 (sentiment) model...")
    tok1, model1 = load_model(STAGE1_MODEL_DIR, device)
    print("Running sentiment predictions...")
    df["predicted_sentiment"] = predict_stage1(df["review_clean"].tolist(), tok1, model1, device)

    print("\nLoading Stage 2 (dissatisfaction) model...")
    tok2, model2 = load_model(STAGE2_MODEL_DIR, device)

    for cat in CATEGORY_COLUMNS:
        df[cat] = ""

    negative_mask = df["predicted_sentiment"] == "negative"
    negative_indices = df[negative_mask].index.tolist()

    if negative_indices:
        print(f"Running dissatisfaction predictions on {len(negative_indices)} negative reviews...")
        negative_texts = df.loc[negative_indices, "review_clean"].tolist()
        stage2_results = predict_stage2(negative_texts, tok2, model2, device, thresholds)
        for idx, result in zip(negative_indices, stage2_results):
            for cat, val in result.items():
                df.at[idx, cat] = val
    else:
        print("No negative reviews found, skipping Stage 2.")

    output_cols = ["review", "predicted_sentiment"] + CATEGORY_COLUMNS
    df[output_cols].to_csv(args.output, index=False)
    print(f"\nSaved results to {args.output}")

    print_summary(df)


if __name__ == "__main__":
    main()
