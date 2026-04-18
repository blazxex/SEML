import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
RESULTS_DIR = Path("results")


class AggregationEngine:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)

    def run(self, results_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        RESULTS_DIR.mkdir(exist_ok=True)

        prices = pd.read_parquet(self.processed_dir / "prices_clean.parquet")
        prices["Date"] = pd.to_datetime(prices["Date"])

        df = results_df.copy()
        df["trading_date"] = pd.to_datetime(df["trading_date"])

        classified = df[df["label"] != "No Opinion"]

        def _agg(group):
            total = len(group)
            classified_group = group[group["label"] != "No Opinion"]
            c = len(classified_group)
            buy = (group["label"] == "Buy").sum()
            sell = (group["label"] == "Sell").sum()
            hold = (group["label"] == "Hold").sum()
            no_op = (group["label"] == "No Opinion").sum()
            sentiment_score = (buy - sell) / c if c > 0 else 0.0
            return pd.Series({
                "tweet_volume": total,
                "buy_count": buy,
                "sell_count": sell,
                "hold_count": hold,
                "no_opinion_count": no_op,
                "buy_pct": buy / total if total > 0 else 0.0,
                "sell_pct": sell / total if total > 0 else 0.0,
                "hold_pct": hold / total if total > 0 else 0.0,
                "no_opinion_pct": no_op / total if total > 0 else 0.0,
                "sentiment_score": sentiment_score,
            })

        agg = (
            df.groupby(["ticker", "trading_date"])
            .apply(_agg)
            .reset_index()
        )

        agg = agg.sort_values(["ticker", "trading_date"])
        agg["rolling_3day_sentiment"] = (
            agg.groupby("ticker")["sentiment_score"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        agg["rolling_7day_sentiment"] = (
            agg.groupby("ticker")["sentiment_score"]
            .transform(lambda s: s.rolling(7, min_periods=1).mean())
        )

        # Merge stock prices
        prices_slim = prices[["Date", "Stock Name", "Close", "Daily Return %", "Intraday Trend"]].copy()
        prices_slim.rename(columns={"Date": "trading_date", "Stock Name": "ticker"}, inplace=True)
        merged = agg.merge(prices_slim, on=["ticker", "trading_date"], how="left")

        out_path = RESULTS_DIR / f"aggregated_{model_name}.csv"
        merged.to_csv(out_path, index=False)
        log.info("Saved %s (%d rows)", out_path, len(merged))
        return merged
