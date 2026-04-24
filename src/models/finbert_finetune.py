"""
Fine-tune ProsusAI/finbert on the project's training split.

Training data strategy:
  - Silver labels: VADER applied to tweets_train.parquet (~28k tweets)
  - Gold labels:   human_labels/lebeled.csv (200 tweets, repeated 5x to up-weight)
  - Val set:       10% stratified split from combined data
  - "No Opinion" rows are dropped (FinBERT has no such class)

Label mapping (matches ProsusAI/finbert output order):
  Buy  → positive (id 0)
  Sell → negative (id 1)
  Hold → neutral  (id 2)
"""
import logging
import os
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

PROCESSED_DIR  = Path("data/processed")
HUMAN_LABELS   = Path("human_labels/lebeled.csv")
OUTPUT_DIR     = Path("models/finbert_finetuned")
BASE_MODEL     = "ProsusAI/finbert"

LABEL2ID = {"Buy": 0, "Sell": 1, "Hold": 2}
ID2LABEL = {0: "Buy", 1: "Sell", 2: "Hold"}
GOLD_REPEAT = 5   # oversample human labels this many times


class _TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _load_live_human_labels() -> pd.DataFrame:
    """Pull human labels collected via the dashboard from Supabase.

    Multiple annotators may label the same tweet — resolve by majority vote;
    drop ties and any label not in LABEL2ID. Returns a DataFrame with
    columns ['Tweet', 'label']. Returns empty DataFrame if Supabase is
    unreachable or the table is missing.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not (url and key):
        log.info("SUPABASE_URL/KEY not set — skipping live human labels.")
        return pd.DataFrame(columns=["Tweet", "label"])

    try:
        from supabase import create_client
        client = create_client(url, key)
        rows = client.table("human_labels_live").select("tweet_id,tweet,label").execute().data or []
    except Exception as e:
        log.warning("Could not fetch live human labels from Supabase: %s", e)
        return pd.DataFrame(columns=["Tweet", "label"])

    if not rows:
        return pd.DataFrame(columns=["Tweet", "label"])

    by_tweet: dict[int, dict] = {}
    for r in rows:
        tid = r["tweet_id"]
        bucket = by_tweet.setdefault(tid, {"tweet": r["tweet"], "labels": []})
        bucket["labels"].append(r["label"])

    out = []
    for tid, bucket in by_tweet.items():
        valid = [lbl for lbl in bucket["labels"] if lbl in LABEL2ID]
        if not valid:
            continue
        counts = Counter(valid).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            continue  # tie — drop
        out.append({"Tweet": bucket["tweet"], "label": counts[0][0]})

    df = pd.DataFrame(out, columns=["Tweet", "label"])
    log.info("Loaded %d live human labels (after majority vote) from Supabase", len(df))
    return df


def _vader_label(text: str) -> str | None:
    vader = SentimentIntensityAnalyzer()
    tokens = str(text).split()
    if len(tokens) < 3:
        return None
    s = vader.polarity_scores(str(text))
    if s["compound"] >= 0.05:
        return "Buy"
    if s["compound"] <= -0.05:
        return "Sell"
    return "Hold"


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1":       f1_score(labels, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }


class FinBERTFineTuner:
    """Fine-tunes ProsusAI/finbert and saves the best checkpoint."""

    def _build_training_df(self) -> pd.DataFrame:
        # --- silver labels from train split ---
        train_df = pd.read_parquet(PROCESSED_DIR / "tweets_train.parquet")
        train_df = train_df.reset_index()
        log.info("Labelling %d train tweets with VADER (silver labels)...", len(train_df))
        train_df["label"] = train_df["Tweet"].apply(_vader_label)
        silver = train_df[["Tweet", "label"]].dropna()

        # --- gold labels from human annotation (CSV seed + live dashboard) ---
        gold_frames: list[pd.DataFrame] = []
        if HUMAN_LABELS.exists():
            human_df = pd.read_csv(HUMAN_LABELS)
            if "final_label" in human_df.columns:
                human_df = human_df.rename(columns={"final_label": "label"})
            human_df = human_df[["Tweet", "label"]].dropna()
            human_df = human_df[human_df["label"].isin(LABEL2ID)]
            log.info("Loaded %d gold human labels from CSV seed", len(human_df))
            gold_frames.append(human_df)

        live_df = _load_live_human_labels()
        if not live_df.empty:
            gold_frames.append(live_df)

        gold_parts: list[pd.DataFrame] = []
        if gold_frames:
            gold_all = pd.concat(gold_frames, ignore_index=True).drop_duplicates(
                subset=["Tweet"], keep="last"  # live labels override CSV on conflict
            )
            log.info("Total gold human labels: %d (repeated x%d)", len(gold_all), GOLD_REPEAT)
            gold_parts = [gold_all] * GOLD_REPEAT

        combined = pd.concat([silver, *gold_parts], ignore_index=True)
        combined = combined[combined["label"].isin(LABEL2ID)]
        log.info("Combined training rows: %d", len(combined))
        return combined

    def run(self, epochs: int = 3, batch_size: int = 16, max_samples: int | None = None) -> dict:
        """
        Fine-tune and save model.  Returns {f1, accuracy, model_path}.

        max_samples: if set, stratified-subsample the combined dataset to at most
        this many rows before train/val split. Useful for demo / CPU runs.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        df = self._build_training_df()
        if max_samples is not None and len(df) > max_samples:
            df = df.groupby("label", group_keys=False).apply(
                lambda g: g.sample(
                    n=max(1, int(round(max_samples * len(g) / len(df)))),
                    random_state=42,
                )
            ).reset_index(drop=True)
            log.info("Subsampled training rows to %d (max_samples=%d)", len(df), max_samples)
        texts  = df["Tweet"].astype(str).tolist()
        labels = df["label"].map(LABEL2ID).tolist()

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.1, random_state=42, stratify=labels
        )

        # --- tokenise ---
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        train_enc = tokenizer(train_texts, truncation=True, max_length=128,
                              padding="max_length", return_tensors=None)
        val_enc   = tokenizer(val_texts,   truncation=True, max_length=128,
                              padding="max_length", return_tensors=None)

        train_dataset = _TweetDataset(train_enc, train_labels)
        val_dataset   = _TweetDataset(val_enc,   val_labels)

        # --- model ---
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=3,
            id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

        # device detection: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            device_str = "mps"
        elif torch.cuda.is_available():
            device_str = "cuda"
        else:
            device_str = "cpu"
        log.info("Training device: %s", device_str)

        training_args = TrainingArguments(
            output_dir              = str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs        = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size * 2,
            eval_strategy           = "epoch",
            save_strategy           = "epoch",
            load_best_model_at_end  = True,
            metric_for_best_model   = "eval_f1",
            greater_is_better       = True,
            logging_steps           = 100,
            seed                    = 42,
            fp16                    = torch.cuda.is_available(),
            dataloader_num_workers  = 0,
        )

        trainer = Trainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = val_dataset,
            compute_metrics = _compute_metrics,
            callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
        )

        with mlflow.start_run(run_name="finbert_finetune", nested=True):
            mlflow.log_params({
                "base_model":    BASE_MODEL,
                "epochs":        epochs,
                "batch_size":    batch_size,
                "train_samples": len(train_texts),
                "val_samples":   len(val_texts),
                "gold_repeat":   GOLD_REPEAT,
                "device":        device_str,
            })

            trainer.train()

            metrics = trainer.evaluate()
            val_f1  = metrics.get("eval_f1", 0.0)
            val_acc = metrics.get("eval_accuracy", 0.0)

            mlflow.log_metrics({"val_f1": val_f1, "val_accuracy": val_acc})

            # save best model + tokenizer
            model_path = str(OUTPUT_DIR / "best")
            trainer.save_model(model_path)
            tokenizer.save_pretrained(model_path)
            try:
                mlflow.log_artifact(model_path, artifact_path="finbert_finetuned")
            except Exception as e:
                log.warning("Skipped MLflow artifact upload (non-fatal): %s", e)

            log.info("Fine-tune done. val_f1=%.4f  saved → %s", val_f1, model_path)

        return {"f1": val_f1, "accuracy": val_acc, "model_path": model_path}
