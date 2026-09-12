# SentiTaglish+

Sentiment-Based Detection of Customer Dissatisfaction in Code-Switched
Taglish E-Commerce Reviews Using XLM-RoBERTa.

An enhanced version of ccosme's SentiTaglish sentiment model, extended with
a second stage that classifies *why* a negative review is negative.

## What it does

Given a CSV of raw customer reviews, SentiTaglish+ automatically:

1. Classifies each review's sentiment as **negative**, **neutral**,
   **positive**, or **mixed**.
2. For every review classified as negative, identifies which specific
   dissatisfaction category or categories apply (a review can match more
   than one - for example, a review can be both "wrong item sent" and
   "poor seller service" at once).

This turns a pile of unstructured customer reviews into a structured
breakdown of what customers are unhappy about and how often, without
reading each review by hand.

## How it works

Two models run in sequence:

- **Stage 1 (Sentiment):** a fine-tuned XLM-RoBERTa model trained on 10,510
  labeled Taglish product reviews, classifying overall sentiment.
- **Stage 2 (Dissatisfaction):** a second fine-tuned XLM-RoBERTa model,
  trained only on the negative reviews from Stage 1's training set,
  classifying each into one or more of 12 dissatisfaction categories.

Both models share the same text-cleaning step before predicting, so input
text is processed the same way it was during training.

## Requirements

- Python 3.10 or later
- The trained model files for both stages (see **Setup** below)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

The trained models are not included in this repository (the files are too
large for GitHub). Before running the tool, place the trained model files
in the following folders:

```
models/
  stage1_sentiment/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
  stage2_dissatisfaction/
    config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
```

If you don't have these yet, train them yourself using the notebooks in
`/kaggle` (or `/notebooks`), or request the files (shared separately, e.g.
via Google Drive, due to file size).

## Input format

The input file must be a CSV with a column named exactly `review`
(lowercase). No other columns are required.

```csv
review
"ang bagal ng delivery, sobrang tagal bago dumating"
"maganda naman yung product, sulit sa presyo"
"sira agad, isang gamit lang nasira na"
```

Notes:
- The column must be named `review`, not `Review`, `reviews`, or anything else.
- Rows with a blank review (or a review that becomes blank after cleaning)
  are automatically skipped and reported in the console output.
- No sentiment or category labels are needed - this is for unlabeled,
  real-world review data.

## Usage

```bash
python3 src/sentitaglishplus.py --input reviews.csv --output results.csv
```

| Flag | Required | Description |
|---|---|---|
| `--input` | Yes | Path to the input CSV (must contain a `review` column) |
| `--output` | Yes | Path where the results CSV will be written |

Running the tool prints a console summary (sentiment breakdown, then a
ranked list of dissatisfaction categories among the negative reviews) in
addition to writing the output file.

## Output format

The output CSV contains the original review, its predicted sentiment, and
one column per dissatisfaction category:

| Column | Description |
|---|---|
| `review` | The original review text, unchanged |
| `predicted_sentiment` | One of `negative`, `neutral`, `positive`, `mixed` |
| `def` – `val` (12 columns) | `1` if that category applies, `0` if not. Left blank for reviews that are not negative. |

### Dissatisfaction category reference

| Code | Category |
|---|---|
| `def` | Product Defect or Malfunction |
| `dam` | Damaged on Arrival |
| `perf` | Poor Product Performance |
| `pmq` | Poor Material or Build Quality |
| `lst` | Listing/Description Mismatch |
| `var` | Wrong Item or Variant Sent |
| `mi` | Missing Item, Part, or Accessory |
| `auth` | Counterfeit or Authenticity Concern |
| `pss` | Poor Seller Service |
| `del` | Delivery or Logistics Issue |
| `pack` | Poor Packaging |
| `val` | Poor Value for Money |

A negative review can have more than one category set to `1`.

## Known limitations

- **`auth` (Counterfeit/Authenticity) predictions are unreliable.** This
  category had only 61 labeled training examples (1.8% of the training
  data), which was not enough for the model to learn a reliable pattern.
  The tool still outputs a prediction for this category, but it should not
  be trusted the same way as the other 11.
- The model is trained specifically on Taglish (Tagalog-English
  code-switched) e-commerce reviews. Performance on purely English,
  purely Tagalog, or non-e-commerce text is not guaranteed.

## Project structure

```
THESIS-QUADB/
  config/
    stage2_thresholds.json    per-category decision thresholds for Stage 2
  data/                       training datasets
  kaggle/ (or notebooks/)     training notebooks for both stages
  models/                     trained model files (not in git, see Setup)
  src/
    preprocess.py             shared text-cleaning logic
    sentitaglishplus.py       the CLI tool
  requirements.txt
```

## Credits

Built on top of the SentiTaglish sentiment corpus and methodology from
Cosme, C. J. & De Leon, M. M. (2024), *Sentiment Analysis of Code-Switched
Filipino-English Product and Service Reviews Using Transformers-Based
Large Language Models.*
