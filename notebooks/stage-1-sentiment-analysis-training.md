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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_NAMES),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

def tokenize_fn(batch):
    return tokenizer(
        batch["review_clean"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

train_ds = Dataset.from_pandas(train_df[["review_clean", "label"]].reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df[["review_clean", "label"]].reset_index(drop=True))

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds = val_ds.map(tokenize_fn, batched=True)
```

    Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    


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
    


    Map:   0%|          | 0/8408 [00:00<?, ? examples/s]



    Map:   0%|          | 0/2102 [00:00<?, ? examples/s]


## 6. Train


```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": f1_macro,
        "macro_precision": precision_macro,
        "macro_recall": recall_macro,
        "micro_f1": f1_micro,
        "micro_precision": precision_micro,
        "micro_recall": recall_micro,
    }

# transformers renamed evaluation_strategy -> eval_strategy in v4.46+.
# Try the new name first, fall back to the old one.
common_training_kwargs = dict(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=epochs,
    save_strategy="epoch",
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()
```



    <div>

      <progress value='5255' max='5255' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [5255/5255 10:17, Epoch 5/5]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>Macro F1</th>
      <th>Macro Precision</th>
      <th>Macro Recall</th>
      <th>Micro F1</th>
      <th>Micro Precision</th>
      <th>Micro Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.524253</td>
      <td>0.516453</td>
      <td>0.819220</td>
      <td>0.654818</td>
      <td>0.709921</td>
      <td>0.647604</td>
      <td>0.819220</td>
      <td>0.819220</td>
      <td>0.819220</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.511934</td>
      <td>0.512190</td>
      <td>0.828259</td>
      <td>0.681842</td>
      <td>0.749612</td>
      <td>0.667907</td>
      <td>0.828259</td>
      <td>0.828259</td>
      <td>0.828259</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.378797</td>
      <td>0.566656</td>
      <td>0.838725</td>
      <td>0.688929</td>
      <td>0.748063</td>
      <td>0.675910</td>
      <td>0.838725</td>
      <td>0.838725</td>
      <td>0.838725</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.450406</td>
      <td>0.739883</td>
      <td>0.833968</td>
      <td>0.677640</td>
      <td>0.718124</td>
      <td>0.667826</td>
      <td>0.833968</td>
      <td>0.833968</td>
      <td>0.833968</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.212692</td>
      <td>0.843250</td>
      <td>0.839201</td>
      <td>0.720302</td>
      <td>0.761560</td>
      <td>0.702868</td>
      <td>0.839201</td>
      <td>0.839201</td>
      <td>0.839201</td>
    </tr>
  </tbody>
</table><p>



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]



    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]





    TrainOutput(global_step=5255, training_loss=0.421880201063873, metrics={'train_runtime': 634.7474, 'train_samples_per_second': 66.231, 'train_steps_per_second': 8.279, 'total_flos': 2765346848808960.0, 'train_loss': 0.421880201063873, 'epoch': 5.0})



## 7. Evaluate and save


```python
metrics = trainer.evaluate()
print("Final validation metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

if not SMOKE_TEST:
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to {OUTPUT_DIR}")
    print("Download it from the Kaggle 'Output' tab on the right after this session ends.")
else:
    print("\n[SMOKE TEST] Model not saved. Set SMOKE_TEST = False above and re-run for the real training run.")
```



<div>

  <progress value='263' max='263' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [263/263 00:07]
</div>




<table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Epoch</th>
      <th>Accuracy</th>
      <th>Macro F1</th>
      <th>Macro Precision</th>
      <th>Macro Recall</th>
      <th>Micro F1</th>
      <th>Micro Precision</th>
      <th>Micro Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.212692</td>
      <td>0.843250</td>
      <td>5</td>
      <td>0.839201</td>
      <td>0.720302</td>
      <td>0.761560</td>
      <td>0.702868</td>
      <td>0.839201</td>
      <td>0.839201</td>
      <td>0.839201</td>
    </tr>
  </tbody>
</table><p>


    Final validation metrics:
      eval_loss: 0.8432499170303345
      eval_accuracy: 0.8392007611798288
      eval_macro_f1: 0.7203018366388686
      eval_macro_precision: 0.7615604387453773
      eval_macro_recall: 0.7028683095553009
      eval_micro_f1: 0.8392007611798288
      eval_micro_precision: 0.8392007611798288
      eval_micro_recall: 0.8392007611798288
    


    Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s]


    
    Model saved to ../models/stage1_sentiment
    Download it from the Kaggle 'Output' tab on the right after this session ends.
    
