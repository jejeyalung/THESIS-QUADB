# Stage 2: Dissatisfaction Classifier Training (XLM-RoBERTa, multi-label)

Trains on SentiTaglish_ProductsAndServices_Dissatisfaction_Classification.csv
to classify WHY a negative review is negative, across 13 categories:
def, dam, perf, pmq, lst, var, mi, auth, pss, del, pack, val

Unlike Stage 1 (one sentiment per review), a review can belong to MULTIPLE
dissatisfaction categories at once (72% of the labeled data has 2+ labels).
This means: different loss function (binary cross-entropy per label, not
softmax over one label), different metrics (per-label F1 + micro/macro
averages, not simple accuracy), and a different split strategy (iterative
stratification for multi-label, not sklearn's standard stratify).

**Before running:** set `SMOKE_TEST = True` first, run everything once to
confirm no errors. Then set `SMOKE_TEST = False` and run the full thing.

**Also check:** DATA_PATH matches wherever your uploaded dataset landed
under /kaggle/input/. Use the copy-path icon in the Data panel to get the
exact path rather than typing it by hand.

## 1. Setup


```python
!pip install -q -U transformers datasets scikit-learn
!pip install -q iterative-stratification
```

    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    WARNING: Ignoring invalid distribution ~ransformers (C:\Users\Julian Edison\AppData\Local\Programs\Python\Python313\Lib\site-packages)
    


```python
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False
    print("iterative-stratification not available, will fall back to random split")

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
    

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


    Loading weights:   0%|          | 0/197 [00:00<?, ?it/s]


    [transformers] [1mXLMRobertaForSequenceClassification LOAD REPORT[0m from: xlm-roberta-base
    Key                         | Status     | 
    ----------------------------+------------+-
    lm_head.dense.bias          | UNEXPECTED | 
    lm_head.layer_norm.weight   | UNEXPECTED | 
    lm_head.bias                | UNEXPECTED | 
    lm_head.dense.weight        | UNEXPECTED | 
    lm_head.layer_norm.bias     | UNEXPECTED | 
    roberta.pooler.dense.bias   | UNEXPECTED | 
    roberta.pooler.dense.weight | UNEXPECTED | 
    classifier.out_proj.bias    | MISSING    | 
    classifier.dense.bias       | MISSING    | 
    classifier.out_proj.weight  | MISSING    | 
    classifier.dense.weight     | MISSING    | 
    
    Notes:
    - UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    

    Attention implementation: sdpa
    

## 2. Config


```python
# --- CHECK THIS PATH ---
# Use the copy-path icon in Kaggle's Data panel to get the exact path.
DATA_PATH = "../data/SentiTaglish_ProductsAndServices_Dissatisfaction_Classification.csv"
OUTPUT_DIR = "../models/stage2_dissatisfaction"


SMOKE_TEST = False
SMOKE_TEST_SIZE = 200
EPOCHS = 10

SEED = 42
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
PREDICTION_THRESHOLD = 0.5   # sigmoid output above this counts as a positive label

CATEGORY_COLUMNS = ["def", "dam", "perf", "pmq", "lst", "var", "mi",
                    "auth", "pss", "del", "pack", "val"]
NUM_LABELS = len(CATEGORY_COLUMNS)
ID2LABEL = {i: name for i, name in enumerate(CATEGORY_COLUMNS)}
LABEL2ID = {name: i for i, name in enumerate(CATEGORY_COLUMNS)}

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: no GPU detected. Check Notebook options > Accelerator is set to GPU.")
```

    Using device: cuda
    

## 3. Text cleaning


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

print(clean_text("SOBRAAAA panget ng quality, sira agad!!"))
```

    sobraa panget ng quality, sira agad!!
    

## 4. Load and prepare data


```python
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows")

# Handle either 'review' or 'a' as the text column name (spreadsheet exports
# sometimes rename the first column oddly)
if "review" not in df.columns and "a" in df.columns:
    df = df.rename(columns={"a": "review"})

missing_cols = [c for c in CATEGORY_COLUMNS if c not in df.columns]
if missing_cols:
    raise KeyError(f"Missing expected category columns: {missing_cols}. Found: {list(df.columns)}")

df["review_clean"] = df["review"].apply(clean_text)
df = df[df["review_clean"] != ""].reset_index(drop=True)

labels_matrix = df[CATEGORY_COLUMNS].values.astype(float)
print(f"\nAfter cleaning: {len(df)} rows")

print("\nPer-category positive counts:")
for i, cat in enumerate(CATEGORY_COLUMNS):
    count = int(labels_matrix[:, i].sum())
    print(f"  {cat:6s}: {count:5d} ({100*count/len(df):.1f}%)")

label_counts_per_row = labels_matrix.sum(axis=1)
print(f"\nRows with 2+ labels: {(label_counts_per_row >= 2).sum()} ({100*(label_counts_per_row>=2).sum()/len(df):.1f}%)")

if SMOKE_TEST:
    df = df.sample(min(len(df), SMOKE_TEST_SIZE), random_state=SEED).reset_index(drop=True)
    labels_matrix = df[CATEGORY_COLUMNS].values.astype(float)
    epochs = 1
    print(f"\n[SMOKE TEST] Using {len(df)} rows, 1 epoch")
else:
    epochs = EPOCHS

# Multi-label stratified split: ensures rare categories (e.g. auth at ~2%)
# appear in both train and val, unlike a plain random split.
if HAS_ITERSTRAT and len(df) > 20:
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(msss.split(df["review_clean"], labels_matrix))
else:
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED
    )
    print("Using random split (iterstrat unavailable or dataset too small for smoke test)")

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
train_labels = labels_matrix[train_idx]
val_labels = labels_matrix[val_idx]

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")
```

    Loaded 3408 rows
    
    After cleaning: 3408 rows
    
    Per-category positive counts:
      def   :   710 (20.8%)
      dam   :   502 (14.7%)
      perf  :   392 (11.5%)
      pmq   :   742 (21.8%)
      lst   :   822 (24.1%)
      var   :   878 (25.8%)
      mi    :   379 (11.1%)
      auth  :    61 (1.8%)
      pss   :  1268 (37.2%)
      del   :   224 (6.6%)
      pack  :   251 (7.4%)
      val   :   474 (13.9%)
    
    Rows with 2+ labels: 2460 (72.2%)
    
    Train: 2708 | Val: 700
    

## 5. Tokenize


```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

def make_dataset(texts, labels):
    enc = tokenizer(
        list(texts), truncation=True, padding="max_length", max_length=MAX_LENGTH
    )
    enc["labels"] = labels.tolist()
    return Dataset.from_dict(enc)

train_ds = make_dataset(train_df["review_clean"], train_labels)
val_ds = make_dataset(val_df["review_clean"], val_labels)
```


    Loading weights:   0%|          | 0/197 [00:00<?, ?it/s]


    [transformers] [1mXLMRobertaForSequenceClassification LOAD REPORT[0m from: xlm-roberta-base
    Key                         | Status     | 
    ----------------------------+------------+-
    lm_head.dense.bias          | UNEXPECTED | 
    lm_head.layer_norm.weight   | UNEXPECTED | 
    lm_head.bias                | UNEXPECTED | 
    lm_head.dense.weight        | UNEXPECTED | 
    lm_head.layer_norm.bias     | UNEXPECTED | 
    roberta.pooler.dense.bias   | UNEXPECTED | 
    roberta.pooler.dense.weight | UNEXPECTED | 
    classifier.out_proj.bias    | MISSING    | 
    classifier.dense.bias       | MISSING    | 
    classifier.out_proj.weight  | MISSING    | 
    classifier.dense.weight     | MISSING    | 
    
    Notes:
    - UNEXPECTED:	can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:	those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
    

## 6. Train


```python
## 6. Train (with class-weighted loss for imbalance)

import torch.nn as nn

# --- Compute per-category pos_weight from the TRAINING split only ---
# pos_weight = (# negatives / # positives) per label, standard formula for
# BCEWithLogitsLoss. This tells the loss "a missed positive on a rare label
# (like auth) costs more than a missed positive on a common one (like pss)."
train_labels_arr = np.array(train_labels).astype(float)
num_pos = train_labels_arr.sum(axis=0)
num_neg = len(train_labels_arr) - num_pos

raw_pos_weight = num_neg / np.clip(num_pos, 1, None)  # avoid div-by-zero

# Optional: cap extreme weights so ultra-rare labels (auth) don't destabilize
# training by making the model over-predict positives just to avoid penalty.
MAX_POS_WEIGHT = 10.0
pos_weight = np.clip(raw_pos_weight, None, MAX_POS_WEIGHT)

print("Per-category pos_weight (train split):")
for cat, raw, capped in zip(CATEGORY_COLUMNS, raw_pos_weight, pos_weight):
    print(f"  {cat:6s}: raw={raw:6.2f}  capped={capped:6.2f}")

pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)


# --- Custom Trainer that overrides the loss with weighted BCE ---
class WeightedBCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= PREDICTION_THRESHOLD).astype(int)
    labels = labels.astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)
    subset_accuracy = accuracy_score(labels, preds)
    h_loss = hamming_loss(labels, preds)

    return {
        "accuracy": subset_accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "subset_accuracy": subset_accuracy,
        "hamming_loss": h_loss,
    }

common_training_kwargs = dict(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=epochs,
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=20,
    seed=SEED,
    report_to="none",
)
try:
    training_args = TrainingArguments(eval_strategy="epoch", **common_training_kwargs)
except TypeError:
    training_args = TrainingArguments(evaluation_strategy="epoch", **common_training_kwargs)

trainer = WeightedBCETrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
```

    Per-category pos_weight (train split):
      def   : raw=  3.77  capped=  3.77
      dam   : raw=  5.74  capped=  5.74
      perf  : raw=  7.62  capped=  7.62
      pmq   : raw=  3.56  capped=  3.56
      lst   : raw=  3.12  capped=  3.12
      var   : raw=  2.86  capped=  2.86
      mi    : raw=  7.94  capped=  7.94
      auth  : raw= 54.27  capped= 10.00
      pss   : raw=  1.67  capped=  1.67
      del   : raw= 14.13  capped= 10.00
      pack  : raw= 12.47  capped= 10.00
      val   : raw=  6.15  capped=  6.15
    



    <div>

      <progress value='3390' max='3390' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [3390/3390 08:45, Epoch 10/10]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>Macro F1</th>
      <th>Micro F1</th>
      <th>Macro Precision</th>
      <th>Macro Recall</th>
      <th>Subset Accuracy</th>
      <th>Hamming Loss</th>
      <th>Runtime</th>
      <th>Samples Per Second</th>
      <th>Steps Per Second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1.000503</td>
      <td>0.909232</td>
      <td>0.067143</td>
      <td>0.383577</td>
      <td>0.411877</td>
      <td>0.313518</td>
      <td>0.699519</td>
      <td>0.067143</td>
      <td>0.330119</td>
      <td>3.025100</td>
      <td>231.395000</td>
      <td>29.090000</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.750533</td>
      <td>0.660942</td>
      <td>0.070000</td>
      <td>0.534856</td>
      <td>0.592674</td>
      <td>0.428656</td>
      <td>0.756682</td>
      <td>0.070000</td>
      <td>0.177381</td>
      <td>2.896000</td>
      <td>241.709000</td>
      <td>30.386000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.643268</td>
      <td>0.571742</td>
      <td>0.140000</td>
      <td>0.602086</td>
      <td>0.655040</td>
      <td>0.504445</td>
      <td>0.777641</td>
      <td>0.140000</td>
      <td>0.138929</td>
      <td>2.942900</td>
      <td>237.860000</td>
      <td>29.902000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.501132</td>
      <td>0.513850</td>
      <td>0.197143</td>
      <td>0.670107</td>
      <td>0.694436</td>
      <td>0.593473</td>
      <td>0.808724</td>
      <td>0.197143</td>
      <td>0.119643</td>
      <td>2.906100</td>
      <td>240.876000</td>
      <td>30.282000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.456905</td>
      <td>0.497795</td>
      <td>0.247143</td>
      <td>0.691725</td>
      <td>0.717655</td>
      <td>0.616696</td>
      <td>0.798280</td>
      <td>0.247143</td>
      <td>0.105476</td>
      <td>2.934400</td>
      <td>238.549000</td>
      <td>29.989000</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.390806</td>
      <td>0.467142</td>
      <td>0.292857</td>
      <td>0.727853</td>
      <td>0.743305</td>
      <td>0.656544</td>
      <td>0.823962</td>
      <td>0.292857</td>
      <td>0.093571</td>
      <td>2.938700</td>
      <td>238.197000</td>
      <td>29.945000</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.350687</td>
      <td>0.466470</td>
      <td>0.332857</td>
      <td>0.727649</td>
      <td>0.748362</td>
      <td>0.656365</td>
      <td>0.824749</td>
      <td>0.332857</td>
      <td>0.091429</td>
      <td>2.875800</td>
      <td>243.413000</td>
      <td>30.600000</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.310411</td>
      <td>0.448131</td>
      <td>0.360000</td>
      <td>0.731448</td>
      <td>0.755231</td>
      <td>0.657143</td>
      <td>0.829154</td>
      <td>0.360000</td>
      <td>0.087738</td>
      <td>3.077300</td>
      <td>227.474000</td>
      <td>28.597000</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.249792</td>
      <td>0.446629</td>
      <td>0.402857</td>
      <td>0.743007</td>
      <td>0.767520</td>
      <td>0.677147</td>
      <td>0.826844</td>
      <td>0.402857</td>
      <td>0.082143</td>
      <td>3.070300</td>
      <td>227.993000</td>
      <td>28.662000</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.223466</td>
      <td>0.446688</td>
      <td>0.417143</td>
      <td>0.743963</td>
      <td>0.769335</td>
      <td>0.681089</td>
      <td>0.821968</td>
      <td>0.417143</td>
      <td>0.080952</td>
      <td>3.172200</td>
      <td>220.666000</td>
      <td>27.741000</td>
    </tr>
  </tbody>
</table><p>



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]





    TrainOutput(global_step=3390, training_loss=0.49304621107107066, metrics={'train_runtime': 527.5382, 'train_samples_per_second': 51.333, 'train_steps_per_second': 6.426, 'total_flos': 1781421777100800.0, 'train_loss': 0.49304621107107066, 'epoch': 10.0})



## 7. Evaluate and save


```python
metrics = trainer.evaluate()
print("Final validation metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# Per-category breakdown, since macro/micro averages hide which categories
# are actually struggling (expect 'auth' to be weakest, given only ~2% of data)
predictions = trainer.predict(val_ds)
probs = 1 / (1 + np.exp(-predictions.predictions))
preds = (probs >= PREDICTION_THRESHOLD).astype(int)
val_labels_arr = np.array(val_labels).astype(int)

print("\nPer-category F1:")
for i, cat in enumerate(CATEGORY_COLUMNS):
    f1 = f1_score(val_labels_arr[:, i], preds[:, i], zero_division=0)
    support = int(val_labels_arr[:, i].sum())
    print(f"  {cat:6s}: F1={f1:.3f}  (support={support})")

if not SMOKE_TEST:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
    print("Download it from the Kaggle 'Output' tab after this session ends.")
else:
    print("\n[SMOKE TEST] Model not saved. Set SMOKE_TEST = False above and re-run for the real training run.")
```



<div>

  <progress value='88' max='88' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [88/88 00:02]
</div>




<table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Epoch</th>
      <th>Accuracy</th>
      <th>Macro F1</th>
      <th>Micro F1</th>
      <th>Macro Precision</th>
      <th>Macro Recall</th>
      <th>Subset Accuracy</th>
      <th>Hamming Loss</th>
      <th>Runtime</th>
      <th>Samples Per Second</th>
      <th>Steps Per Second</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.223466</td>
      <td>0.446688</td>
      <td>10</td>
      <td>0.417143</td>
      <td>0.743963</td>
      <td>0.769335</td>
      <td>0.681089</td>
      <td>0.821968</td>
      <td>0.417143</td>
      <td>0.080952</td>
      <td>3.046600</td>
      <td>229.765000</td>
      <td>28.885000</td>
    </tr>
  </tbody>
</table><p>


    Final validation metrics:
      eval_loss: 0.44668814539909363
      eval_accuracy: 0.41714285714285715
      eval_macro_f1: 0.7439634116097454
      eval_micro_f1: 0.7693351424694709
      eval_macro_precision: 0.6810886577464778
      eval_macro_recall: 0.8219677188316026
      eval_subset_accuracy: 0.41714285714285715
      eval_hamming_loss: 0.08095238095238096
      eval_runtime: 3.0466
      eval_samples_per_second: 229.765
      eval_steps_per_second: 28.885
    





    
    Per-category F1:
      def   : F1=0.755  (support=142)
      dam   : F1=0.739  (support=100)
      perf  : F1=0.701  (support=78)
      pmq   : F1=0.693  (support=148)
      lst   : F1=0.747  (support=164)
      var   : F1=0.874  (support=176)
      mi    : F1=0.734  (support=76)
      auth  : F1=0.593  (support=12)
      pss   : F1=0.822  (support=254)
      del   : F1=0.708  (support=45)
      pack  : F1=0.737  (support=50)
      val   : F1=0.824  (support=95)
    


    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Model saved to ../models/stage2_dissatisfaction
    Download it from the Kaggle 'Output' tab after this session ends.
    


```python
# Threshold sweep - paste this as a new cell after "7. Evaluate and save"
# Reuses `predictions` and `val_labels_arr` already computed in that cell

from sklearn.metrics import f1_score

thresholds_to_try = [0.5, 0.4, 0.3, 0.2, 0.1]

print(f"{'Category':8s} " + " ".join(f"t={t:<5.1f}" for t in thresholds_to_try))
for i, cat in enumerate(CATEGORY_COLUMNS):
    scores = []
    for t in thresholds_to_try:
        preds_t = (probs[:, i] >= t).astype(int)
        f1 = f1_score(val_labels_arr[:, i], preds_t, zero_division=0)
        scores.append(f1)
    print(f"{cat:8s} " + " ".join(f"{s:<7.3f}" for s in scores))
```

    Category t=0.5   t=0.4   t=0.3   t=0.2   t=0.1  
    def      0.755   0.739   0.725   0.655   0.529  
    dam      0.739   0.734   0.715   0.681   0.541  
    perf     0.701   0.663   0.618   0.580   0.455  
    pmq      0.693   0.688   0.643   0.594   0.477  
    lst      0.747   0.713   0.643   0.577   0.486  
    var      0.874   0.857   0.818   0.742   0.574  
    mi       0.734   0.750   0.707   0.611   0.463  
    auth     0.593   0.457   0.360   0.243   0.127  
    pss      0.822   0.797   0.738   0.691   0.621  
    del      0.708   0.693   0.654   0.592   0.419  
    pack     0.737   0.700   0.687   0.631   0.461  
    val      0.824   0.816   0.821   0.743   0.538  
    
