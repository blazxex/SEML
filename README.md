# Stock Sentiment Analysis — SE4ML Course Project

An end-to-end ML pipeline that classifies stock-related tweets (AAPL, TSLA, TSM) into sentiment signals, correlates them with price movements, and serves results through a Bloomberg-style web dashboard.

> **For informational purposes only. Not financial advice.**

---

## Quick Start

```bash
# 1. Clone and set up environment
cp .env.example .env          # fill in SUPABASE_URL, SUPABASE_KEY

# 2a. Docker (recommended)
docker compose up --build

# 2b. Local
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --models vader finbert gpt --skip_gpt False
python -m dashboard.app
```

| Service | URL | Credentials |
|---|---|---|
| Dashboard | http://localhost:8080 | your account |
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8082 | admin / admin |

---

## Architecture

```
tweets.csv + stock_prices.csv
        │
        ▼
PreprocessingPipeline   (60/20/20 time-based split)
        │
        ├── VADERModel          → results/vader_results.csv
        ├── FinBERTModel        → results/finbert_results.csv
        ├── FinBERTModel (ft)   → results/finbert_finetuned_results.csv
        └── GPTModel            → results/gpt_results.csv
                │
        AggregationEngine       → aggregated_{model}.csv
        DriftDetector           → drift_flags_{model}.csv
        EvaluationEngine        → classification reports + Pearson r
                │
        MLflow Model Registry   (champion/challenger promotion)
                │
        Flask Dashboard         ← reads CSVs / Supabase fallback
        Airflow DAG             ← daily retrain at 02:00 UTC
```

---

## Models

Evaluated on 200 human-labeled tweets (test split). Fine-tuned FinBERT val metrics are from training validation set.

| Model | Macro F1¹ | Weighted F1¹ | Accuracy¹ | Buy% | Hold% | Sell% | Confidence |
|---|---|---|---|---|---|---|---|
| VADER | 0.28 | 0.30 | 34% | 50.9% | 26.2% | 22.8% | — |
| FinBERT (base) | 0.27 | 0.26 | 27% | 9.1% | 71.8% | 9.3% | 81.9% |
| Gemma3:4b (Ollama) | **0.59** | **0.63** | **65%** | 43.2% | 8.3% | 21.7% | — |
| FinBERT (fine-tuned) | 0.893² | —² | 90.6%² | **51.6%** | 24.6% | 21.5% | **96.9%** |

¹ Evaluated on 200 human-labeled tweets.  
² Evaluated on training validation set (VADER silver + gold labels) — not directly comparable.

Fine-tuning training data: ~35k VADER silver labels + 145 human gold labels (×5 oversampled).

---

## Running the Pipeline

```bash
# VADER only (fast, no GPU)
python run_pipeline.py --models vader

# All models + fine-tune FinBERT on project data
python run_pipeline.py --models vader finbert gpt \
    --finetune --skip_gpt False

# Skip preprocessing (parquets already exist)
python run_pipeline.py --models finbert --finetune --skip_preprocessing

# Control fine-tuning epochs
python run_pipeline.py --models finbert --finetune --finetune_epochs 5
```

After the pipeline, upload results to Supabase (for cloud deployment):

```bash
python seed_data.py
python seed_data.py --models finbert_finetuned --clear   # re-upload one model
```

---

## Dashboard Features

- **Historical view** — sentiment score vs. close price (dual-axis), daily Buy/Hold/Sell/No Opinion distribution
- **Click any chart point** → Bloomberg-style day summary panel:
  - Price card (Open / High / Low / Close, daily return %, volume)
  - Sentiment gauge with needle
  - Sentiment meter (−1 Bearish → +1 Bullish)
  - Rolling 3-day / 7-day averages
  - Drift flag status
  - Side-by-side model comparison
- **Live simulation** — replays test-split tweets in real time using VADER, writes to Supabase
- **Model selector** — VADER / FinBERT (Base) / FinBERT (Fine-tuned) / Gemma3:4b (Ollama)
- **Drift alerts** — banner appears when any drift flag is active

---

## Automated Retraining (Airflow)

DAG: `sentiment_retrain_pipeline` — runs daily at 02:00 UTC.

```
check_drift → retrain_finbert → evaluate_challenger → compare_champion → promote_if_better
```

- Triggers if ≥ 30% of recent dates have any drift flag
- Champion/challenger: promotes to MLflow Production only if val F1 improves
- Airflow UI: http://localhost:8082 (admin / admin)

---

## Project Structure

```
├── src/
│   ├── preprocessing.py          # PreprocessingPipeline
│   ├── aggregation.py            # AggregationEngine
│   ├── evaluation.py             # EvaluationEngine
│   ├── drift_detection.py        # DriftDetector
│   └── models/
│       ├── vader_model.py
│       ├── finbert_model.py
│       ├── finbert_finetune.py   # FinBERTFineTuner
│       └── gpt_model.py
├── dags/
│   └── retrain_dag.py            # Airflow DAG
├── dashboard/
│   ├── app.py                    # Flask app + API routes
│   ├── simulator.py              # Live simulation worker
│   ├── templates/index.html
│   └── static/
├── docker-compose.yml            # Unified stack
├── Dockerfile.dashboard
├── Dockerfile.airflow
├── Dockerfile.mlflow
├── run_pipeline.py               # CLI orchestration
├── seed_data.py                  # Migrate CSVs → Supabase
├── seed_admin.py                 # Create dashboard login account
└── supabase_setup.sql            # Database schema
```

---

## Environment Variables

```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key

# Flask
FLASK_SECRET_KEY=change-me-in-production
FLASK_PORT=8080

# MLflow (Docker sets this automatically)
MLFLOW_TRACKING_URI=./mlruns

# OpenAI (only needed for GPT model)
OPENAI_API_KEY=sk-...

# Ollama (optional local LLM)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=gemma3:4b
```

---

## First-Time Setup

```bash
# 1. Create Supabase tables
#    Open supabase_setup.sql → paste into Supabase SQL Editor → Run

# 2. Create your dashboard login account
python seed_admin.py

# 3. Run the pipeline
python run_pipeline.py --models vader finbert --finetune --skip_gpt False

# 4. Upload data to Supabase (for cloud deployment)
python seed_data.py

# 5. Start the dashboard
python -m dashboard.app
```

---

## Data

- **Dataset**: ~53,000 tweets + OHLCV prices for AAPL, TSLA, TSM (Sept 2021 – Sept 2022)
- **Split**: 60% train / 20% val / 20% test — time-based only (no random split)
- **Human labels**: 200 tweets manually annotated for F1 evaluation
- **Known limitations**: static historical dataset; no ground-truth trading labels; LLM output quality depends on Ollama model version; correlation ≠ causation

---

## Tech Stack

Python 3.11 · Flask · Chart.js · Bootstrap 5 · Apache Airflow · MLflow · Supabase · Docker · HuggingFace Transformers · PyTorch
