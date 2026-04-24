"""
Generate a small mock tweet stream for the live-simulation demo.

The real tweets_test.parquet has ~9,853 rows from 2022 — too long to walk
through during a presentation. This script writes a curated, realistic-looking
parquet with a configurable size, recent dates, and all three tickers.

Usage:
    python seed_mock_tweets.py                       # 300 tweets, last 15 days
    python seed_mock_tweets.py --count 500 --days 30
    python seed_mock_tweets.py --out data/processed/tweets_test.parquet --overwrite
    python seed_mock_tweets.py --seed-supabase       # also seed empty sim_state rows

After running, point the simulator at the mock file by either:
  (a) replacing data/processed/tweets_test.parquet (use --overwrite), or
  (b) symlinking:  ln -sf tweets_test_mock.parquet data/processed/tweets_test.parquet
"""
import argparse
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

OUT_DEFAULT = Path("data/processed/tweets_test_mock.parquet")
TICKERS = {
    "AAPL": "Apple Inc.",
    "TSLA": "Tesla, Inc.",
    "TSM":  "Taiwan Semiconductor",
}

# Roughly weighted: bullish 40%, bearish 25%, neutral 35%
TEMPLATES = {
    "AAPL": {
        "bullish": [
            "Just bought more $AAPL on this dip. iPhone 17 demand is INSANE 🚀",
            "$AAPL services revenue keeps printing. Apple is a software company now, period.",
            "Tim Cook quietly dropped Vision Pro 2 specs and nobody is talking about it. $AAPL setting up beautifully.",
            "App Store fees barely dented by EU regs. $AAPL moat is just too wide.",
            "If $AAPL holds 230 it's going to 260 by EOM. Setup looks textbook.",
            "Apple buyback alone supports $AAPL above 220. Rate cut and we're at ATH.",
            "China iPhone numbers came in better than the bears wanted. $AAPL squeeze incoming.",
            "$AAPL pulling away from Samsung in premium segment. Pricing power = wide moat.",
        ],
        "bearish": [
            "$AAPL trading at 30x earnings with single-digit growth. Make it make sense.",
            "iPhone upgrade cycle is dead. $AAPL multiple compression coming.",
            "China sales for $AAPL down 20% YoY. This is a falling knife rn.",
            "Warren Buffett trimming $AAPL again should tell you something.",
            "$AAPL Vision Pro is a complete flop. Stock priced for innovation that isn't there.",
            "Services growth slowing, hardware flat. $AAPL is dead money for 2 years.",
        ],
        "neutral": [
            "Earnings on Thursday for $AAPL. Holding through but not adding.",
            "$AAPL options activity unusual — straddle premium up but no clear direction.",
            "Apple announces event for next week. $AAPL chop until the headline.",
            "$AAPL float keeps shrinking. Buybacks > new shares for years now.",
            "Anyone else watching $AAPL 50DMA? Bouncing for 4 sessions in a row.",
            "Apple dividend hike rumor making rounds. $AAPL yield still tiny tho.",
        ],
    },
    "TSLA": {
        "bullish": [
            "$TSLA Robotaxi unveil is going to be a generational catalyst. Loaded calls.",
            "Cybertruck delivery numbers crushing it. $TSLA back to $300 incoming.",
            "FSD v13 is genuinely incredible. $TSLA software moat is real.",
            "$TSLA energy storage growing 90% YoY. Nobody is pricing this in.",
            "Elon back focused on Tesla = $TSLA 🚀 . Distraction trade is over.",
            "Shanghai gigafactory capacity expansion confirmed. $TSLA shorts in pain.",
            "$TSLA delivered record Q with margins UP. The bears are silent rn.",
        ],
        "bearish": [
            "$TSLA P/E still 80+ with deliveries flat. Reality coming for this stock.",
            "BYD outselling Tesla in China by a mile now. $TSLA narrative cracking.",
            "Cybertruck recalls again. $TSLA quality control is a joke.",
            "$TSLA margins compressed for 6 quarters straight. Growth story broken.",
            "Elon spending $TSLA brand equity on politics. Won't end well for shareholders.",
            "Auto industry in cyclical decline. $TSLA priced like it doesn't apply to them.",
        ],
        "neutral": [
            "$TSLA holds 250 and we go higher, breaks and we test 220. Watching closely.",
            "Tesla earnings tonight. $TSLA IV through the roof, premium too rich for me.",
            "$TSLA delivery numbers tomorrow. Whisper at 460k, consensus 445k.",
            "Cybertruck production update due this week. $TSLA chop until we hear.",
            "$TSLA options flow mixed. Big calls AND big puts hitting. No clear lean.",
            "Anyone have a take on $TSLA 200DMA? Hasn't lost it in 8 months.",
        ],
    },
    "TSM": {
        "bullish": [
            "$TSM is the most important company on earth and trades at 18x. Steal.",
            "$TSM Arizona fab finally producing N4. US chip independence trade is on.",
            "AI capex isn't slowing. $TSM is THE picks-and-shovels play.",
            "$TSM 3nm yields above 80% per the latest leak. Apple and NVDA locked in.",
            "Taiwan election risk priced in. $TSM at $180 is generational opportunity.",
            "$TSM raising prices on advanced nodes. Pricing power = monopoly economics.",
        ],
        "bearish": [
            "$TSM China exposure is an underpriced tail risk. One headline = -20%.",
            "Samsung gaining share at 2nm. $TSM moat narrowing for first time in years.",
            "Geopolitics around Taiwan getting worse. $TSM at any P/E is too risky.",
            "$TSM capex blowing out. FCF story falling apart in real time.",
            "Smartphone demand tanking. $TSM mature node revenue collapsing.",
        ],
        "neutral": [
            "$TSM monthly revenue out tomorrow. Watching for AI vs phone mix.",
            "Anyone else find $TSM coverage thin in Western media? Hard to get edge here.",
            "$TSM ADR vs Taipei listing premium widening. Arb getting interesting.",
            "TSMC investor day next month. $TSM probably ranges till then.",
            "$TSM dividend yield creeping up. Quietly becoming an income stock.",
            "Watching $TSM 175 as key support. Holds = base, breaks = retest 160.",
        ],
    },
}

# Distribution per ticker: 40 bullish / 25 bearish / 35 neutral
SENTIMENT_WEIGHTS = [("bullish", 0.40), ("bearish", 0.25), ("neutral", 0.35)]


def _pick_sentiment(rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for name, w in SENTIMENT_WEIGHTS:
        cum += w
        if r <= cum:
            return name
    return "neutral"


def _pick_tweet(ticker: str, sentiment: str, rng: random.Random) -> str:
    pool = TEMPLATES[ticker][sentiment]
    base = rng.choice(pool)
    # Light variation: occasional emoji / cashtag duplication so re-rolls don't look identical
    if rng.random() < 0.15:
        base = base + " " + rng.choice(["#stocks", "#fintwit", "#trading", "📊", "👀", "💎🙌"])
    return base


def generate(count: int, days: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    start = today - timedelta(days=days - 1)

    tickers = list(TICKERS.keys())
    rows = []
    for i in range(count):
        ticker  = rng.choice(tickers)
        days_offset = rng.randint(0, days - 1)
        trading_date = start + timedelta(days=days_offset)
        # Tweet posted some time during that trading day
        ts = trading_date + timedelta(
            hours=rng.randint(13, 23),  # market-hours-ish UTC
            minutes=rng.randint(0, 59),
            seconds=rng.randint(0, 59),
        )
        sentiment = _pick_sentiment(rng)
        tweet = _pick_tweet(ticker, sentiment, rng)
        rows.append({
            "Date":          ts,
            "Tweet":         tweet,
            "Stock Name":    ticker,
            "Company Name":  TICKERS[ticker],
            "Trading Date":  trading_date,
        })

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return df


def maybe_seed_supabase_state():
    """Reset sim_state position to 0 for all models so the demo replays from scratch."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not (url and key):
            print("[supabase] SUPABASE_URL/KEY not set — skipping state reset.")
            return
        client = create_client(url, key)
        for model in ("vader", "finbert", "finbert_finetuned", "gpt"):
            client.table("sim_state").upsert({
                "model": model, "position": 0, "status": "paused", "speed_seconds": 30,
            }, on_conflict="model").execute()
            client.table("sim_results").delete().eq("model", model).execute()
        print("[supabase] sim_state reset to position=0 and sim_results cleared for all models.")
    except Exception as e:
        print(f"[supabase] could not seed state: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count",     type=int,    default=300, help="Number of mock tweets (default 300)")
    p.add_argument("--days",      type=int,    default=15,  help="Span across N recent days (default 15)")
    p.add_argument("--seed",      type=int,    default=42,  help="RNG seed (default 42)")
    p.add_argument("--out",       type=Path,   default=OUT_DEFAULT, help=f"Output parquet path (default {OUT_DEFAULT})")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing file at --out")
    p.add_argument("--seed-supabase", action="store_true", help="Reset sim_state.position=0 & clear sim_results")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing {args.out} — pass --overwrite to confirm.")

    df = generate(args.count, args.days, args.seed)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {len(df)} mock tweets → {args.out}")
    print("Distribution by ticker:")
    print(df["Stock Name"].value_counts().to_string())
    print("Date range: {} → {}".format(df["Trading Date"].min().date(), df["Trading Date"].max().date()))

    if args.seed_supabase:
        maybe_seed_supabase_state()

    real = Path("data/processed/tweets_test.parquet")
    if args.out != real:
        print()
        print("To use this in the live simulator, either:")
        print(f"  (a) python seed_mock_tweets.py --out {real} --overwrite")
        print(f"  (b) cp {args.out} {real}")
        print(f"  (c) ln -sf {args.out.name} {real}")


if __name__ == "__main__":
    main()
