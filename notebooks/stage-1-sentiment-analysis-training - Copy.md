# Stage 1: Sentiment Classifier Training (XLM-RoBERTa)

Trains on SentiTaglish_ProductsAndServices.csv to classify reviews as
negative, neutral, positive, or mixed.

**Before running:** set `SMOKE_TEST = True` in the config cell below and
run everything once. It uses a tiny subset and 1 epoch, just to prove the
pipeline works end to end. Once that finishes with no errors, set
`SMOKE_TEST = False` and run the full thing.

**Also check:** the DATA_PATH in the config cell matches wherever your
uploaded dataset actually landed under /kaggle/input/.

## 1. Setup


```python
!pip install -q -U transformers datasets scikit-learn
```

    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    


```python
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

    Torch version: 2.14.0+cu130
    CUDA available: True
    GPU: NVIDIA GeForce RTX 5060
    


```python
import transformers
import torch
import sklearn
import datasets as hf_datasets

try:
    import iterstrat
    iterstrat_version = getattr(iterstrat, "__version__", "unknown (no __version__ attr)")
except ImportError:
    iterstrat_version = "not installed"

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("sklearn:", sklearn.__version__)
print("datasets:", hf_datasets.__version__)
print("iterative-stratification:", iterstrat_version)

# Also worth checking what attention backend the model actually loaded with
from transformers import AutoModelForSequenceClassification
model_check = AutoModelForSequenceClassification.from_pretrained(
    "xlm-roberta-base", num_labels=12, problem_type="multi_label_classification"
)
print("Attention implementation:", model_check.config._attn_implementation)
```

    transformers: 5.17.0
    torch: 2.14.0+cu130
    sklearn: 1.9.1
    datasets: 5.0.1
    iterative-stratification: 0.1.9
    


    Loading weights:   0%|          | 0/197 [00:00<?, ?it/s]


    [transformers] [1mXLMRobertaForSequenceClassification LOAD REPORT[0m from: xlm-roberta-base
    Key                         | Status     | 
    ----------------------------+------------+-
    lm_head.layer_norm.bias     | UNEXPECTED | 
    lm_head.dense.weight        | UNEXPECTED | 
    lm_head.dense.bias          | UNEXPECTED | 
    lm_head.bias                | UNEXPECTED | 
    roberta.pooler.dense.weight | UNEXPECTED | 
    roberta.pooler.dense.bias   | UNEXPECTED | 
    lm_head.layer_norm.weight   | UNEXPECTED | 
    classifier.dense.weight     | MISSING    | 
    classifier.out_proj.bias    | MISSING    | 
    classifier.out_proj.weight  | MISSING    | 
    classifier.dense.bias       | MISSING    | 
    
    Notes:
    - UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    

    Attention implementation: sdpa
    

## 2. Config


```python
# --- CHECK THIS PATH ---
# In Kaggle, uploaded datasets live under /kaggle/input/<dataset-name>/<filename>
# Update this to match yourDATA_PATH = "/kaggle/input/datasets/nahokeel/sentimentanalysis/SentiTaglish_ProductsAndServices.csv" actual dataset name once you've added it as an input.
DATA_PATH = "../data/SentiTaglish_ProductsAndServices.csv"
OUTPUT_DIR = "../models/stage1_sentiment"


SMOKE_TEST = False          # set False for the real full run
SMOKE_TEST_SIZE = 200      # total rows used across all classes in smoke test mode
EPOCHS = 5                 # used only when SMOKE_TEST = False

SEED = 42
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 8

LABEL_NAMES = ["negative", "neutral", "positive", "mixed"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

# Numeric codes in the raw CSV, confirmed against Table 1 in the proposal
SENTIMENT_CODE_MAP = {1: "negative", 2: "neutral", 3: "positive", 4: "mixed"}

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: no GPU detected. Check Notebook options > Accelerator is set to GPU.")
```

    Using device: cuda
    

## 3. Preprocessing


```python
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
    text = _REPEATED_CHAR.sub(r"\1\1", text)
    tokens = text.split()
    tokens = [SPELLING_MAP.get(t, t) for t in tokens]
    text = " ".join(tokens)
    text = _WHITESPACE.sub(" ", text).strip()
    return text

# quick check
print(clean_text("SOBRAAAA ganda ng product!!! salamt seller"))
```

    sobraa ganda ng product!! salamat seller
    

## 4. Load and prepare data


```python
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

df["sentiment_label"] = df["sentiment"].map(SENTIMENT_CODE_MAP)
unmapped = df["sentiment_label"].isna().sum()
if unmapped:
    raise ValueError(f"{unmapped} rows have a sentiment code not in SENTIMENT_CODE_MAP")

df["label"] = df["sentiment_label"].map(LABEL2ID)
df["review_clean"] = df["review"].apply(clean_text)
df = df[df["review_clean"] != ""].reset_index(drop=True)

print(df["sentiment_label"].value_counts())

if SMOKE_TEST:
    df = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, SMOKE_TEST_SIZE // 4)), random_state=SEED)
    ).reset_index(drop=True)
    epochs = 1
    print(f"\n[SMOKE TEST] Using {len(df)} rows, 1 epoch")
else:
    epochs = EPOCHS

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=SEED
)
print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")
```

    Loaded 10510 rows
    sentiment_label
    positive    3443
    negative    3408
    mixed       3397
    neutral      262
    Name: count, dtype: int64
    
    Train: 8408 | Val: 2102
    


```python
print(df["label"].value_counts())
print(f"Train: {len(train_df)} | Val: {len(val_df)}")
print(train_df["label"].value_counts())
print(val_df["label"].value_counts())
```

    label
    2    3443
    0    3408
    3    3397
    1     262
    Name: count, dtype: int64
    Train: 8408 | Val: 2102
    label
    2    2754
    0    2726
    3    2718
    1     210
    Name: count, dtype: int64
    label
    2    689
    0    682
    3    679
    1     52
    Name: count, dtype: int64
    

## 5. Tokenize


```python
import gc, os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DataCollatorWithPadding

N_FOLDS = 5
EPOCHS = 5
MODEL_NAME = "xlm-roberta-base"   # rerun with "bert-base-multilingual-uncased"
TAG = MODEL_NAME.split("/")[-1]
os.makedirs("../results", exist_ok=True)

def to_ds(d):
    ds = Dataset.from_pandas(d[["review_clean", "label"]].reset_index(drop=True))
    return ds.map(lambda b: tokenizer(b["review_clean"], truncation=True, max_length=MAX_LENGTH), batched=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
rows = []

for fold, (tr, va) in enumerate(skf.split(df, df["label"])):
    print(f"\n=== Fold {fold + 1}/{N_FOLDS} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES), id2label=ID2LABEL, label2id=LABEL2ID
    )
    train_ds, val_ds = to_ds(df.iloc[tr]), to_ds(df.iloc[va])

    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{TAG}_fold{fold}",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        num_train_epochs=EPOCHS,
        save_strategy="no",
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      data_collator=DataCollatorWithPadding(tokenizer))
    trainer.train()

    out = trainer.predict(val_ds)
    y, p = out.label_ids, out.predictions.argmax(-1)
    ids = list(range(len(LABEL_NAMES)))

    row = {
        "model": TAG, "fold": fold,
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
        "macro_precision": precision_score(y, p, average="macro", zero_division=0),
        "macro_recall": recall_score(y, p, average="macro", zero_division=0),
    }
    for name, f in zip(LABEL_NAMES, f1_score(y, p, average=None, labels=ids, zero_division=0)):
        row[f"f1_{name}"] = f
    rows.append(row)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

res = pd.DataFrame(rows)
res.to_csv(f"../results/stage1_{TAG}_folds.csv", index=False)
print(res.drop(columns="model").describe().loc[["mean", "std"]].round(4))
```
