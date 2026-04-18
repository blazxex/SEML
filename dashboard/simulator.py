"""
Mock real-time simulation worker.
Replays tweets from the test split, classifies with the active model,
and writes aggregated results to Supabase every N seconds.
"""
import logging
import threading
import time
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from dashboard.supabase_client import get_client

log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")
BATCH_SIZE    = 50  # tweets processed per tick

_lock   = threading.Lock()
_thread: threading.Thread | None = None
_stop_event = threading.Event()

# VADER instance (always available, no GPU needed)
_vader = SentimentIntensityAnalyzer()


def _vader_label(text: str) -> str:
    scores = _vader.polarity_scores(str(text))
    tokens = str(text).split()
    if len(tokens) < 3:
        return "No Opinion"
    if scores["compound"] >= 0.05:
        return "Buy"
    if scores["compound"] <= -0.05:
        return "Sell"
    return "Hold"


def _get_label_from_results(model: str, tweet_id: int) -> str:
    """Replay label from pre-computed results CSV for finbert/gpt."""
    path = RESULTS_DIR / f"{model}_results.csv"
    if not path.exists():
        return "No Opinion"
    # Cache in memory to avoid re-reading every tick
    if not hasattr(_get_label_from_results, "_cache"):
        _get_label_from_results._cache = {}
    if model not in _get_label_from_results._cache:
        df = pd.read_csv(path, usecols=["tweet_id", "label"])
        _get_label_from_results._cache[model] = df.set_index("tweet_id")["label"].to_dict()
    cache = _get_label_from_results._cache[model]
    return cache.get(tweet_id, "No Opinion")


def _classify_batch(model: str, batch_df: pd.DataFrame) -> pd.DataFrame:
    batch = batch_df.copy()
    if model == "vader":
        batch["label"] = batch["Tweet"].apply(_vader_label)
    else:
        batch["label"] = batch["tweet_id"].apply(lambda tid: _get_label_from_results(model, tid))
    return batch


def _aggregate_and_save(model: str, batch_df: pd.DataFrame):
    db = get_client()
    for (ticker, trading_date), grp in batch_df.groupby(["Stock Name", "Trading Date"]):
        total = len(grp)
        buy   = (grp["label"] == "Buy").sum()
        sell  = (grp["label"] == "Sell").sum()
        hold  = (grp["label"] == "Hold").sum()
        no_op = (grp["label"] == "No Opinion").sum()
        classified = buy + sell + hold
        sentiment_score = float((buy - sell) / classified) if classified > 0 else 0.0

        db.table("sim_results").insert({
            "ticker":         str(ticker),
            "model":          model,
            "trading_date":   str(trading_date)[:10],
            "label":          "aggregated",
            "sentiment_score": round(sentiment_score, 4),
            "buy_pct":        round(buy / total, 4),
            "sell_pct":       round(sell / total, 4),
            "hold_pct":       round(hold / total, 4),
            "no_opinion_pct": round(no_op / total, 4),
            "tweet_volume":   total,
        }).execute()


def _run_simulation(model: str):
    log.info("Simulator started for model: %s", model)
    db = get_client()

    # Load full test split
    test_df = pd.read_parquet(PROCESSED_DIR / "tweets_test.parquet")
    test_df.index.name = "tweet_id"
    test_df = test_df.reset_index()
    test_df["Trading Date"] = pd.to_datetime(test_df["Trading Date"]).dt.strftime("%Y-%m-%d")

    total = len(test_df)

    while not _stop_event.is_set():
        # Read current position from Supabase
        state = db.table("sim_state").select("position,status,speed_seconds").eq("model", model).execute()
        if not state.data:
            break
        s = state.data[0]
        if s["status"] != "running":
            time.sleep(2)
            continue

        pos   = s["position"]
        speed = s["speed_seconds"]

        if pos >= total:
            db.table("sim_state").update({"status": "finished"}).eq("model", model).execute()
            log.info("Simulation finished for %s", model)
            break

        # Process next batch
        batch = test_df.iloc[pos: pos + BATCH_SIZE].copy()
        classified = _classify_batch(model, batch)
        _aggregate_and_save(model, classified)

        new_pos = pos + len(batch)
        db.table("sim_state").update({
            "position": new_pos,
            "updated_at": "now()",
        }).eq("model", model).execute()

        log.info("Simulator [%s]: %d/%d tweets processed", model, new_pos, total)
        time.sleep(speed)

    log.info("Simulator thread exiting for model: %s", model)


def start(model: str):
    global _thread
    with _lock:
        db = get_client()
        db.table("sim_state").update({"status": "running"}).eq("model", model).execute()

        _stop_event.clear()
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(target=_run_simulation, args=(model,), daemon=True)
            _thread.start()


def pause(model: str):
    db = get_client()
    db.table("sim_state").update({"status": "paused"}).eq("model", model).execute()


def reset(model: str):
    db = get_client()
    db.table("sim_state").update({"status": "paused", "position": 0}).eq("model", model).execute()
    db.table("sim_results").delete().eq("model", model).execute()
    log.info("Simulation reset for %s", model)


def set_speed(model: str, seconds: int):
    db = get_client()
    db.table("sim_state").update({"speed_seconds": max(5, seconds)}).eq("model", model).execute()


def get_state(model: str) -> dict:
    db = get_client()
    result = db.table("sim_state").select("*").eq("model", model).execute()
    return result.data[0] if result.data else {}
