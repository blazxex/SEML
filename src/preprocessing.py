import html
import re
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from langdetect import detect, LangDetectException

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TICKERS = {"AAPL", "TSLA", "TSM"}
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _clean_tweet(text: str) -> str:
    text = html.unescape(str(text))
    text = URL_RE.sub("", text)
    return text.strip()


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


class PreprocessingPipeline:
    def __init__(self, data_dir: str = "cleaned_data", processed_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _load_prices(self) -> pd.DataFrame:
        path = self.data_dir / "cleaned_stock_prices.csv"
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
        df = df[df["Stock Name"].isin(TICKERS)].copy()

        # Forward-fill missing price days per ticker
        df = df.sort_values(["Stock Name", "Date"]).reset_index(drop=True)
        df["Daily Return %"] = df.groupby("Stock Name")["Close"].transform(
            lambda s: s.pct_change() * 100
        )
        return df

    # ------------------------------------------------------------------
    def _load_tweets(self) -> pd.DataFrame:
        frames = []
        for fname in ["vader_ready_tweets.csv", "finbert_ready_tweets.csv"]:
            p = self.data_dir / fname
            if p.exists():
                frames.append(pd.read_csv(p, low_memory=False))

        df = pd.concat(frames, ignore_index=True)

        # Normalise column names
        df.columns = [c.strip() for c in df.columns]

        # Parse dates
        df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)
        df["Trading Date"] = pd.to_datetime(df["Trading Date"], infer_datetime_format=True)

        # Filter to known tickers (some rows have missing Stock Name — fill from text if possible)
        df = df[df["Stock Name"].isin(TICKERS)].copy()

        # Clean tweet text
        log.info("Cleaning tweet text …")
        df["Tweet"] = df["Tweet"].apply(_clean_tweet)

        # Remove duplicates (same trading date + exact tweet text + ticker)
        before = len(df)
        df = df.drop_duplicates(subset=["Trading Date", "Tweet", "Stock Name"])
        log.info("Dropped %d duplicate tweets", before - len(df))

        # English filter
        log.info("Filtering non-English tweets (may take a minute) …")
        mask = df["Tweet"].apply(_is_english)
        df = df[mask].copy()
        log.info("%d English tweets remain", len(df))

        return df

    # ------------------------------------------------------------------
    def _split(self, df: pd.DataFrame):
        dates = df["Trading Date"].sort_values().unique()
        n = len(dates)
        train_end = dates[int(n * 0.60) - 1]
        val_end = dates[int(n * 0.80) - 1]

        train = df[df["Trading Date"] <= train_end]
        val = df[(df["Trading Date"] > train_end) & (df["Trading Date"] <= val_end)]
        test = df[df["Trading Date"] > val_end]
        return train, val, test

    # ------------------------------------------------------------------
    def run(self) -> dict:
        log.info("Loading stock prices …")
        prices = self._load_prices()
        prices.to_parquet(self.processed_dir / "prices_clean.parquet", index=False)
        log.info("Saved prices_clean.parquet (%d rows)", len(prices))

        log.info("Loading tweets …")
        tweets = self._load_tweets()

        train, val, test = self._split(tweets)
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            out = self.processed_dir / f"tweets_{split_name}.parquet"
            split_df.reset_index(drop=True).to_parquet(out, index=False)
            log.info("Saved %s: %d rows", out.name, len(split_df))

        summary = {}
        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            summary[split_name] = {
                ticker: int((split_df["Stock Name"] == ticker).sum())
                for ticker in sorted(TICKERS)
            }

        log.info("Preprocessing complete. Summary: %s", summary)
        return summary
