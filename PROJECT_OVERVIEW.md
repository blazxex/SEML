# Stock Trend Prediction from Twitter Posts
## Project Overview — SE4ML Course Project

> **Status:** Fully implemented and Dockerized.
> Run the entire stack with `docker compose up --build`.

---

## 1. Project Summary

A **Software Engineering for Machine Learning (SE4ML) course project** that builds a complete, end-to-end ML pipeline:

1. Reads raw tweet data about three stock tickers (AAPL, TSLA, TSM)
2. Classifies each tweet into: **Buy / Hold / Sell / No Opinion**
3. Aggregates daily sentiment signals per ticker
4. Correlates sentiment signals with historical stock price movements
5. Detects data drift and triggers automated model retraining via Airflow
6. Tracks all experiments and promotes the best model via MLflow
7. Serves results via a Flask web dashboard

The system is a **decision-support tool for retail investors**, not an automated trading system.
Every output includes: *"For informational purposes only. Not financial advice."*

---

## 2. Running the Project

### Docker (recommended)
```bash
cp .env.example .env       # fill in SUPABASE_URL, SUPABASE_KEY, etc.
docker compose up --build
```

| Service | URL | Notes |
|---|---|---|
| Dashboard | http://localhost:8080 | Flask frontend |
| MLflow UI | http://localhost:5000 | Experiment tracking |
| Airflow UI | http://localhost:8082 | admin / admin |

### Local dev
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run_pipeline.py --models vader finbert gpt --skip_gpt False
gunicorn "dashboard.app:app" --bind 0.0.0.0:8080
```

---

## 3. Datasets (CSV — Already Available)

### `data/raw/stock_prices.csv`
Historical OHLCV data for AAPL, TSLA, TSM.

| Column | Type | Notes |
|---|---|---|
| Date | string | `DD/MM/YYYY` |
| Open/High/Low/Close/Adj Close | float | Price columns |
| Volume | int | Trading volume |
| Stock Name | string | AAPL / TSLA / TSM |
| Daily Return % | float | `(Close - prev_Close) / prev_Close * 100` |
| Intraday Trend | int | 1 = bullish, 0 = bearish/flat |

### `data/raw/tweets.csv`
~53K raw tweets aligned to trading dates.

| Column | Type | Notes |
|---|---|---|
| Date | string | `DD/MM/YYYY HH:MM` |
| Tweet | string | Raw text — may contain URLs, HTML entities, emojis |
| Stock Name | string | AAPL / TSLA / TSM |
| Trading Date | string | `DD/MM/YYYY` — market day assignment |

**Known data issues handled:** HTML entities, embedded URLs, non-English tweets, duplicates.

---

## 4. Three Models

| Model | Type | Cost | Speed |
|---|---|---|---|
| **VADER** | Rule-based lexicon | $0 | ~5 sec (CPU) |
| **FinBERT** | `ProsusAI/finbert` neural | $0 | ~2–4 min (GPU) |
| **GPT-4o-mini** | OpenAI API, few-shot | ~$5–15 | ~5–18 min |

### Label mapping
- **VADER:** `compound ≥ 0.05` → Buy; `≤ -0.05` → Sell; else → Hold; `token count < 3` → No Opinion
- **FinBERT:** `positive→Buy`, `negative→Sell`, `neutral→Hold`; `max softmax < 0.6` → No Opinion
- **GPT:** 8-example few-shot prompt, batches of 20 tweets, temperature=0

---

## 5. Evaluation Metrics

### On human-labeled subset (200 tweets, test split)
- Accuracy, Precision, Recall, F1 (weighted + macro)
- Cohen's Kappa (inter-annotator between 2 team members)

### Sentiment-price correlation
- Pearson r at lag 0/1/2/3 days; per ticker and combined
- p-value + 95% bootstrap CI (n=1000)
- `sentiment_score = (Buy_count − Sell_count) / total_classified`

### Supporting metrics
- No Opinion Rate, Inter-model Agreement Rate, Label Distribution
- Processing time, API cost (GPT only)

---

## 6. Model Selection Criteria (priority order)

| Priority | Criterion | Target |
|---|---|---|
| 1 | Weighted F1 on human labels | > 0.50 |
| 2 | Pearson r (best lag, combined) | higher |absolute| |
| 3 | No Opinion Rate | < 20% |
| 4 | Inter-model agreement | higher |
| 5 | Inference cost | lower |
| 6 | Processing time | lower |

Winner is registered in MLflow as `stock_sentiment_classifier` → stage `Production`.

---

## 7. MLflow Experiment Tracking

**In Docker:** MLflow runs as a dedicated service at `http://mlflow:5000` (SQLite backend + artifact volume).
**Local dev:** file-based `./mlruns/`.

### Experiment structure
```
Experiment: "stock_sentiment_prediction"
└── Parent Run: pipeline_run_{timestamp}
    ├── Child Run: vader
    ├── Child Run: finbert
    └── Child Run: gpt4o_mini
```

### Per-run logged artifacts
- `results/aggregated_{model}.csv`
- `results/drift_flags_{model}.csv`
- `results/classification_report_{model}.txt`
- `results/inter_model_agreement.csv`

---

## 8. Data Drift Detection & Automated Retraining

### Drift flags (`results/drift_flags_{model}.csv`)
| Flag | Trigger |
|---|---|
| `drift_flag` | Any label class shifts > 15 pp vs 7-day rolling baseline |
| `volume_spike_flag` | Tweet volume > mean + 3×std of 7-day window |
| `weak_signal_flag` | No Opinion rate > 35% |
| `divergence_flag` | Sentiment and price move opposite for 5+ consecutive days |

### Airflow DAG: `sentiment_retrain_pipeline`
Schedule: daily at 02:00 UTC. File: `dags/retrain_dag.py`.

```
check_drift (ShortCircuit)
    │  skip if drift ratio < 30%
    ▼
retrain_finbert        ← fine-tune on train split + human labels
    ▼
evaluate_challenger    ← F1 on human-label test set
    ▼
compare_champion       ← load Production model metrics from MLflow
    ▼
promote_if_better      ← transition to Production if challenger wins
```

---

## 9. Web Dashboard

**Stack:** Flask + Chart.js + Bootstrap 5 (dark theme)

### Routes
| Route | Method | Description |
|---|---|---|
| `/` | GET | Main dashboard |
| `/api/sentiment` | GET | `ticker`, `model` → aggregated sentiment + price JSON |
| `/api/drift` | GET | `ticker`, `model` → drift flags JSON |
| `/api/compare` | GET | Model comparison metrics JSON |

### UI components
1. Ticker selector (AAPL / TSLA / TSM)
2. Model selector (VADER / FinBERT / GPT-4o-mini)
3. Date range slider
4. Dual-axis line chart: sentiment score + stock close price
5. Stacked bar chart: daily Buy/Hold/Sell/No Opinion %
6. Drift alert banner (red, when any flag active)
7. Metrics panel: Pearson r, p-value, best lag
8. Footer disclaimer (every page)

Also includes a **live simulator** (`dashboard/simulator.py`) that replays test-split tweets using VADER for real-time demo purposes.

---

## 10. Project Structure

```
project/
├── data/
│   ├── raw/                          # stock_prices.csv, tweets.csv
│   └── processed/                    # parquets (git-ignored)
│
├── src/
│   ├── preprocessing.py              # PreprocessingPipeline
│   ├── aggregation.py                # AggregationEngine
│   ├── evaluation.py                 # EvaluationEngine
│   ├── drift_detection.py            # DriftDetector
│   └── models/
│       ├── vader_model.py            # VADERModel
│       ├── finbert_model.py          # FinBERTModel
│       ├── finbert_finetune.py       # FinBERTFineTuner (used by retrain DAG)
│       └── gpt_model.py              # GPTModel
│
├── dags/
│   └── retrain_dag.py                # Airflow: daily retrain pipeline
│
├── dashboard/
│   ├── app.py                        # Flask app
│   ├── auth.py                       # Login/user management
│   ├── simulator.py                  # Live replay simulator
│   ├── supabase_client.py            # Supabase connector
│   ├── templates/
│   └── static/
│
├── human_labels/
│   └── lebeled.csv                   # 200-tweet human annotations
│
├── results/                          # Auto-created (git-ignored)
├── mlruns/                           # Local MLflow store (git-ignored)
├── logs/airflow/                     # Airflow logs (git-ignored)
│
├── docker-compose.yml                # Unified stack (dashboard + MLflow + Airflow)
├── Dockerfile.dashboard              # Lightweight Flask image
├── Dockerfile.airflow                # Airflow + ML deps image
├── Dockerfile.mlflow                 # MLflow server image
│
├── requirements.txt                  # Full deps (local dev / Airflow)
├── requirements-dashboard.txt        # Lightweight deps (dashboard image only)
├── requirements-airflow.txt          # Airflow image deps
│
├── run_pipeline.py                   # CLI: run full pipeline locally
├── seed_admin.py                     # One-time: create dashboard admin user
├── .env                              # Secrets — never commit
├── .env.example                      # Template
└── supabase_setup.sql                # Supabase schema bootstrap
```

---

## 11. Environment Variables

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here

# Flask
FLASK_SECRET_KEY=change-me-in-production
FLASK_PORT=8080
FLASK_DEBUG=False

# MLflow
# Local:  MLFLOW_TRACKING_URI=./mlruns
# Docker: set automatically to http://mlflow:5000
MLFLOW_TRACKING_URI=./mlruns

# OpenAI (optional — only needed for GPT model)
OPENAI_API_KEY=your_openai_api_key_here

# Ollama (optional — local LLM alternative)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=gemma3:4b
```

---

## 12. Data Split

Time-based only — **no random splitting** (prevents future data leakage).

| Split | Proportion | Purpose |
|---|---|---|
| Train | 60% | Model config, threshold tuning |
| Validation | 20% | Confidence threshold, calibration |
| Test | 20% | Final evaluation (held out) |

---

## 13. Reproducibility

All scripts set seeds at startup:
```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
```
Every MLflow run logs `random_seed: 42`.

---

## 14. Known Limitations

1. No ground-truth trading outcome labels — all evaluation is indirect
2. Dataset is historical and static (Sept 2021 – Sept 2022)
3. GPT-4o-mini behavior may change with future model updates
4. FinBERT attention-based explainability not implemented
5. System does not account for market microstructure or macroeconomic factors
6. Correlation ≠ causation

---

## 15. Grading Checklist (SE4ML Progress Report 3)

- [x] Clear problem statement, ML task type, dataset described
- [x] Three models with rationale and comparison framework
- [x] End-to-end pipeline: data → preprocessing → inference → evaluation → deployment
- [x] Data drift detection (4 flag types)
- [x] Automated retraining via Airflow DAG with champion/challenger pattern
- [x] MLflow experiment tracking + model registry
- [x] Working web dashboard (Flask + Chart.js)
- [x] Batch prediction mode implemented and justified
- [x] Fixed seeds, pinned deps, reproducible runs
- [x] Disclaimer on every dashboard page
- [x] Docker deployment for all services

---

*Last updated: 2026-04-18*
