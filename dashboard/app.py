import os
import glob
from pathlib import Path
from functools import wraps

import pandas as pd
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from dotenv import load_dotenv

from dashboard.auth import authenticate, create_user
from dashboard.supabase_client import get_client
import dashboard.simulator as simulator

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

RESULTS_DIR = Path(__file__).parent.parent / "results"
MODELS = ["vader", "finbert", "gpt"]


def _load_csv(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, parse_dates=["trading_date"])
    return None


# Cache aggregated + drift data at startup
_aggregated: dict[str, pd.DataFrame] = {}
_drift: dict[str, pd.DataFrame] = {}

for _model in MODELS:
    _df = _load_csv(f"aggregated_{_model}.csv")
    if _df is not None:
        _aggregated[_model] = _df

    _dd = _load_csv(f"drift_flags_{_model}.csv")
    if _dd is not None:
        _drift[_model] = _dd

_comparison: dict = {}
_agreement_path = RESULTS_DIR / "inter_model_agreement.csv"
if _agreement_path.exists():
    _comparison["agreement"] = pd.read_csv(_agreement_path).to_dict(orient="records")


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ------------------------------------------------------------------
# Auth routes
# ------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if authenticate(username, password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ------------------------------------------------------------------
# Main dashboard
# ------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    available_models = list(_aggregated.keys()) or MODELS
    return render_template("index.html", models=available_models, username=session.get("username"))


# ------------------------------------------------------------------
# Historical sentiment API (from CSV)
# ------------------------------------------------------------------

@app.route("/api/sentiment")
@login_required
def api_sentiment():
    ticker = request.args.get("ticker", "AAPL").upper()
    model = request.args.get("model", "vader").lower()
    date_from = request.args.get("date_from")
    date_to = request.args.get("date_to")

    if model not in _aggregated:
        return jsonify({"error": f"No results for model '{model}'. Run the pipeline first."}), 404

    df = _aggregated[model]
    df = df[df["ticker"] == ticker].copy()

    if date_from:
        df = df[df["trading_date"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["trading_date"] <= pd.to_datetime(date_to)]

    df = df.sort_values("trading_date")

    def clean(series):
        return [None if pd.isna(v) else v for v in series]

    payload = {
        "ticker": ticker,
        "model": model,
        "dates": df["trading_date"].dt.strftime("%Y-%m-%d").tolist(),
        "sentiment_score": clean(df["sentiment_score"].round(4)),
        "close_price": clean(df["Close"]),
        "buy_pct": clean(df["buy_pct"].round(4)),
        "sell_pct": clean(df["sell_pct"].round(4)),
        "hold_pct": clean(df["hold_pct"].round(4)),
        "no_opinion_pct": clean(df["no_opinion_pct"].round(4)),
        "rolling_3day_sentiment": clean(df["rolling_3day_sentiment"].round(4)),
        "rolling_7day_sentiment": clean(df["rolling_7day_sentiment"].round(4)),
        "tweet_volume": clean(df["tweet_volume"]),
    }
    return jsonify(payload)


@app.route("/api/drift")
@login_required
def api_drift():
    ticker = request.args.get("ticker", "AAPL").upper()
    model = request.args.get("model", "vader").lower()

    if model not in _drift:
        return jsonify({"error": f"No drift data for model '{model}'."}), 404

    df = _drift[model]
    df = df[df["ticker"] == ticker].sort_values("trading_date")

    payload = {
        "ticker": ticker,
        "model": model,
        "dates": df["trading_date"].dt.strftime("%Y-%m-%d").tolist(),
        "drift_flag": df["drift_flag"].tolist(),
        "volume_spike_flag": df["volume_spike_flag"].tolist(),
        "weak_signal_flag": df["weak_signal_flag"].tolist(),
        "divergence_flag": df["divergence_flag"].tolist(),
        "any_active": bool(
            df[["drift_flag", "volume_spike_flag", "weak_signal_flag", "divergence_flag"]]
            .any()
            .any()
        ),
    }
    return jsonify(payload)


@app.route("/api/compare")
@login_required
def api_compare():
    summary = []
    for model, df in _aggregated.items():
        row = {"model": model}
        row["total_tweets"] = int(df["tweet_volume"].sum())
        row["avg_sentiment"] = round(float(df["sentiment_score"].mean()), 4)
        row["avg_buy_pct"] = round(float(df["buy_pct"].mean()), 4)
        row["avg_sell_pct"] = round(float(df["sell_pct"].mean()), 4)
        row["avg_no_opinion_pct"] = round(float(df["no_opinion_pct"].mean()), 4)
        summary.append(row)

    return jsonify({
        "models": summary,
        "agreement": _comparison.get("agreement", []),
    })


# ------------------------------------------------------------------
# Simulation control API
# ------------------------------------------------------------------

@app.route("/api/sim/start", methods=["POST"])
@login_required
def sim_start():
    model = request.json.get("model", "vader")
    simulator.start(model)
    return jsonify({"status": "started", "model": model})


@app.route("/api/sim/pause", methods=["POST"])
@login_required
def sim_pause():
    model = request.json.get("model", "vader")
    simulator.pause(model)
    return jsonify({"status": "paused", "model": model})


@app.route("/api/sim/reset", methods=["POST"])
@login_required
def sim_reset():
    model = request.json.get("model", "vader")
    simulator.reset(model)
    return jsonify({"status": "reset", "model": model})


@app.route("/api/sim/speed", methods=["POST"])
@login_required
def sim_speed():
    model = request.json.get("model", "vader")
    seconds = int(request.json.get("seconds", 30))
    simulator.set_speed(model, seconds)
    return jsonify({"status": "updated", "model": model, "speed_seconds": max(5, seconds)})


@app.route("/api/sim/state")
@login_required
def sim_state():
    model = request.args.get("model", "vader")
    state = simulator.get_state(model)
    return jsonify(state)


# ------------------------------------------------------------------
# Live results from Supabase (sim_results table)
# ------------------------------------------------------------------

@app.route("/api/live")
@login_required
def api_live():
    ticker = request.args.get("ticker", "AAPL").upper()
    model = request.args.get("model", "vader").lower()

    db = get_client()
    result = (
        db.table("sim_results")
        .select("trading_date,sentiment_score,buy_pct,sell_pct,hold_pct,no_opinion_pct,tweet_volume")
        .eq("ticker", ticker)
        .eq("model", model)
        .order("trading_date")
        .execute()
    )

    rows = result.data or []

    def _clean(v):
        return None if v is None else v

    payload = {
        "ticker": ticker,
        "model": model,
        "dates": [r["trading_date"] for r in rows],
        "sentiment_score": [_clean(r["sentiment_score"]) for r in rows],
        "buy_pct": [_clean(r["buy_pct"]) for r in rows],
        "sell_pct": [_clean(r["sell_pct"]) for r in rows],
        "hold_pct": [_clean(r["hold_pct"]) for r in rows],
        "no_opinion_pct": [_clean(r["no_opinion_pct"]) for r in rows],
        "tweet_volume": [_clean(r["tweet_volume"]) for r in rows],
    }
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 8080))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
