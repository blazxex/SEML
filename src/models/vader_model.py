import random
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mlflow
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

log = logging.getLogger(__name__)
RESULTS_DIR = Path("results")


def _map_label(compound: float, token_count: int) -> tuple[str, float]:
    if token_count < 3:
        return "No Opinion", abs(compound)
    if compound >= 0.05:
        return "Buy", compound
    if compound <= -0.05:
        return "Sell", abs(compound)
    return "Hold", 1 - abs(compound)


class VADERModel:
    def __init__(self, processed_dir: str = "data/processed", batch_size: int = 1000):
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.analyzer = SentimentIntensityAnalyzer()

    def _classify_batch(self, texts: list[str]) -> list[tuple]:
        results = []
        for text in texts:
            scores = self.analyzer.polarity_scores(text)
            tokens = text.split()
            label, confidence = _map_label(scores["compound"], len(tokens))
            results.append((label, confidence, scores["compound"]))
        return results

    def run(self, mlflow_parent_run_id: str | None = None) -> pd.DataFrame:
        RESULTS_DIR.mkdir(exist_ok=True)

        test_df = pd.read_parquet(self.processed_dir / "tweets_test.parquet")
        log.info("VADER: classifying %d tweets …", len(test_df))

        with mlflow.start_run(
            run_name="vader",
            nested=True,
            parent_run_id=mlflow_parent_run_id,
        ) as run:
            mlflow.log_params({
                "model_type": "vader",
                "dataset_split": "test",
                "ticker_scope": "AAPL,TSLA,TSM",
                "batch_size": self.batch_size,
                "random_seed": SEED,
            })

            t0 = time.time()
            labels, confidences, compounds = [], [], []

            texts = test_df["Tweet"].tolist()
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i: i + self.batch_size]
                for label, conf, comp in self._classify_batch(batch):
                    labels.append(label)
                    confidences.append(conf)
                    compounds.append(comp)

            elapsed = time.time() - t0

            results = test_df[["Trading Date", "Stock Name"]].copy()
            results.index.name = "tweet_id"
            results = results.reset_index()
            results.rename(columns={"Stock Name": "ticker", "Trading Date": "trading_date"}, inplace=True)
            results["label"] = labels
            results["confidence"] = confidences
            results["compound_score"] = compounds

            out_path = RESULTS_DIR / "vader_results.csv"
            results.to_csv(out_path, index=False)
            try:
                mlflow.log_artifact(str(out_path))
            except Exception as e:
                print(f"[warn] Skipped MLflow artifact upload: {e}")

            total = len(results)
            vc = results["label"].value_counts(normalize=True)
            mlflow.log_metrics({
                "processing_time_seconds": round(elapsed, 2),
                "total_tweets_processed": total,
                "buy_rate": round(vc.get("Buy", 0), 4),
                "sell_rate": round(vc.get("Sell", 0), 4),
                "hold_rate": round(vc.get("Hold", 0), 4),
                "no_opinion_rate": round(vc.get("No Opinion", 0), 4),
            })

            log.info("VADER done in %.1fs. Distribution: %s", elapsed, vc.to_dict())
            return results
