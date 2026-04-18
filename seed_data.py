"""
Migrate local CSV results into Supabase.

Run once after pipeline completes:
    python seed_data.py
    python seed_data.py --models vader finbert_finetuned   # specific models only
    python seed_data.py --clear                             # wipe tables first
"""
import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path("results")
MODELS = ["vader", "finbert", "finbert_finetuned", "gpt"]
BATCH_SIZE = 500   # rows per Supabase upsert call


def _clean(v):
    """Convert NaN / numpy types to JSON-safe Python."""
    if v is None:
        return None
    try:
        if math.isnan(float(v)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


def migrate_aggregated(db, model: str, clear: bool = False):
    path = RESULTS_DIR / f"aggregated_{model}.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return 0

    if clear:
        db.table("aggregated_results").delete().eq("model", model).execute()
        print(f"  [clear] deleted existing aggregated_results for {model}")

    df = pd.read_csv(path, parse_dates=["trading_date"])
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "model":                  model,
            "ticker":                 r["ticker"],
            "trading_date":           r["trading_date"].strftime("%Y-%m-%d"),
            "tweet_volume":           _clean(r.get("tweet_volume")),
            "buy_count":              _clean(r.get("buy_count")),
            "sell_count":             _clean(r.get("sell_count")),
            "hold_count":             _clean(r.get("hold_count")),
            "no_opinion_count":       _clean(r.get("no_opinion_count")),
            "buy_pct":                _clean(r.get("buy_pct")),
            "sell_pct":               _clean(r.get("sell_pct")),
            "hold_pct":               _clean(r.get("hold_pct")),
            "no_opinion_pct":         _clean(r.get("no_opinion_pct")),
            "sentiment_score":        _clean(r.get("sentiment_score")),
            "rolling_3day_sentiment": _clean(r.get("rolling_3day_sentiment")),
            "rolling_7day_sentiment": _clean(r.get("rolling_7day_sentiment")),
            "close_price":            _clean(r.get("Close")),
            "daily_return":           _clean(r.get("Daily Return %")),
            "intraday_trend":         int(_clean(r.get("Intraday Trend")) or 0),
        })

    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        db.table("aggregated_results").upsert(batch, on_conflict="model,ticker,trading_date").execute()
        total += len(batch)
        print(f"  [aggregated/{model}] {total}/{len(rows)} rows upserted")

    return total


def migrate_drift(db, model: str, clear: bool = False):
    path = RESULTS_DIR / f"drift_flags_{model}.csv"
    if not path.exists():
        print(f"  [skip] {path} not found")
        return 0

    if clear:
        db.table("drift_flags").delete().eq("model", model).execute()

    df = pd.read_csv(path, parse_dates=["trading_date"])
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "model":              model,
            "ticker":             r["ticker"],
            "trading_date":       r["trading_date"].strftime("%Y-%m-%d"),
            "drift_flag":         bool(r.get("drift_flag", False)),
            "volume_spike_flag":  bool(r.get("volume_spike_flag", False)),
            "weak_signal_flag":   bool(r.get("weak_signal_flag", False)),
            "divergence_flag":    bool(r.get("divergence_flag", False)),
        })

    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        db.table("drift_flags").upsert(batch, on_conflict="model,ticker,trading_date").execute()
        total += len(batch)
        print(f"  [drift/{model}] {total}/{len(rows)} rows upserted")

    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--clear", action="store_true", help="Delete existing rows before inserting")
    args = p.parse_args()

    from dashboard.supabase_client import get_client
    db = get_client()

    total_agg   = 0
    total_drift = 0

    for model in args.models:
        print(f"\n── {model} ──────────────────────────")
        total_agg   += migrate_aggregated(db, model, args.clear)
        total_drift += migrate_drift(db, model, args.clear)

    print(f"\n✓ Done. {total_agg} aggregated rows + {total_drift} drift rows uploaded.")


if __name__ == "__main__":
    main()
