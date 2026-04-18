import random
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mlflow
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

log = logging.getLogger(__name__)
RESULTS_DIR = Path("results")
MODEL_NAME = "ProsusAI/finbert"
CONFIDENCE_THRESHOLD = 0.6
# FinBERT label order: positive, negative, neutral
FINBERT_LABELS = ["positive", "negative", "neutral"]
LABEL_MAP = {"positive": "Buy", "negative": "Sell", "neutral": "Hold"}


class FinBERTModel:
    def __init__(
        self,
        processed_dir: str = "data/processed",
        batch_size: int = 32,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        model_name_or_path: str | None = None,   # pass fine-tuned path to use challenger
    ):
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        source = model_name_or_path or MODEL_NAME
        log.info("FinBERT loading from: %s  device: %s", source, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.to(self.device)
        self.model.eval()

    def _classify_batch(self, texts: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for prob_row in probs:
            max_idx = int(np.argmax(prob_row))
            max_conf = float(prob_row[max_idx])
            finbert_label = FINBERT_LABELS[max_idx]

            if max_conf < self.confidence_threshold:
                label = "No Opinion"
            else:
                label = LABEL_MAP[finbert_label]

            results.append({
                "label": label,
                "confidence": round(max_conf, 4),
                "positive_score": round(float(prob_row[0]), 4),
                "negative_score": round(float(prob_row[1]), 4),
                "neutral_score": round(float(prob_row[2]), 4),
            })
        return results

    def run(self, mlflow_parent_run_id: str | None = None) -> pd.DataFrame:
        RESULTS_DIR.mkdir(exist_ok=True)
        test_df = pd.read_parquet(self.processed_dir / "tweets_test.parquet")
        log.info("FinBERT: classifying %d tweets …", len(test_df))

        with mlflow.start_run(
            run_name="finbert",
            nested=True,
            parent_run_id=mlflow_parent_run_id,
        ):
            mlflow.log_params({
                "model_type": "finbert",
                "dataset_split": "test",
                "ticker_scope": "AAPL,TSLA,TSM",
                "batch_size": self.batch_size,
                "confidence_threshold": self.confidence_threshold,
                "random_seed": SEED,
                "device": str(self.device),
            })

            t0 = time.time()
            all_results = []
            texts = test_df["Tweet"].tolist()

            for i in tqdm(range(0, len(texts), self.batch_size), desc="FinBERT"):
                batch_texts = texts[i: i + self.batch_size]
                all_results.extend(self._classify_batch(batch_texts))

            elapsed = time.time() - t0

            result_df = test_df[["Trading Date", "Stock Name"]].copy()
            result_df.index.name = "tweet_id"
            result_df = result_df.reset_index()
            result_df.rename(
                columns={"Stock Name": "ticker", "Trading Date": "trading_date"}, inplace=True
            )
            for key in ["label", "confidence", "positive_score", "negative_score", "neutral_score"]:
                result_df[key] = [r[key] for r in all_results]

            out_path = RESULTS_DIR / "finbert_results.csv"
            result_df.to_csv(out_path, index=False)
            mlflow.log_artifact(str(out_path))

            total = len(result_df)
            vc = result_df["label"].value_counts(normalize=True)
            mlflow.log_metrics({
                "processing_time_seconds": round(elapsed, 2),
                "total_tweets_processed": total,
                "buy_rate": round(vc.get("Buy", 0), 4),
                "sell_rate": round(vc.get("Sell", 0), 4),
                "hold_rate": round(vc.get("Hold", 0), 4),
                "no_opinion_rate": round(vc.get("No Opinion", 0), 4),
                "avg_confidence": round(float(result_df["confidence"].mean()), 4),
            })

            log.info("FinBERT done in %.1fs. Distribution: %s", elapsed, vc.to_dict())
            return result_df
