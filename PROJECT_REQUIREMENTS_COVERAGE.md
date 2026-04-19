# Term Project Requirements Coverage

This document maps each project requirement to the specific files and implementations in this codebase.

---

## Part 2: System Implementation & Demonstration

---

### 1. Use Case & User Interaction

#### 1.1 Real-World ML Use Case
**What we did:** Stock sentiment analysis on ~53,000 tweets across three tickers (AAPL, TSLA, TSM) from September 2021 – September 2022. The system classifies each tweet as **Buy / Hold / Sell / No Opinion** and correlates daily sentiment scores with stock price movements to serve as a decision-support tool for investors.

**Key files:**
- `run_pipeline.py` — master orchestration entry point
- `src/models/vader_model.py` — VADER rule-based classifier
- `src/models/finbert_model.py` — FinBERT transformer inference
- `src/models/finbert_finetune.py` — FinBERT fine-tuning on project data
- `src/models/gpt_model.py` — Gemma3:4b (Ollama) few-shot classification
- `cleaned_data/vader_ready_tweets.csv` — ~26k preprocessed tweets
- `human_labels/lebeled.csv` — 200 hand-annotated tweets (2 annotators)

---

#### 1.2 Basic User Access Control / Authentication
**What we did:** Session-based authentication backed by Supabase PostgreSQL. Passwords are stored as bcrypt hashes (salted). Every dashboard route is protected by a `@login_required` decorator. A first-time admin can be seeded via CLI.

**Key files:**
- `dashboard/auth.py` — `authenticate()`, `create_user()`, `hash_password()`, `check_password()` using `bcrypt`
- `dashboard/app.py` (lines ~1–40) — `login_required` decorator, `/login`, `/logout` routes, `session["logged_in"]`
- `supabase_setup.sql` — `users` table schema (id UUID, username unique, password_hash)
- `seed_admin.py` — CLI to create the first admin user: `python seed_admin.py <username> <password>`

**Flow:**
```
POST /login → auth.authenticate() → bcrypt.checkpw() → session["logged_in"] = True
All other routes → @login_required → redirect to /login if not authenticated
```

---

#### 1.3 Live Front-End User Interface
**What we did:** Flask dashboard served via Gunicorn with a Bootstrap 5 dark theme and Chart.js visualizations. Users can select tickers, models, and date ranges interactively. Clicking a date opens a detailed day panel.

**Key files:**
- `dashboard/app.py` — Flask routes: `/`, `/api/sentiment`, `/api/drift`, `/api/compare`, `/api/day`, `/api/live`, `/api/sim/*`
- `dashboard/templates/index.html` — main dashboard (dual-axis Chart.js, stacked bar chart, sentiment gauge, drift alert banners)
- `dashboard/templates/login.html` — login form
- `dashboard/simulator.py` — thread-based simulation worker replaying test-split tweets in real time via Supabase

**UI features:**
- Ticker selector: AAPL / TSLA / TSM
- Model selector: VADER / FinBERT Base / FinBERT Fine-tuned / Gemma3:4b
- Dual-axis line chart: sentiment score (left) + close price (right)
- Stacked bar chart: daily Buy% / Hold% / Sell% / No Opinion%
- Day-click panel: OHLCV card, sentiment gauge, rolling averages, drift alert banner
- Live simulator: Start / Pause / Reset with adjustable replay speed

---

### 2. Data & Experiment Management

#### 2.1 Clear Data Management Pipeline
**What we did:** A 6-phase pipeline runs end-to-end from raw CSV to registered model.

```
Phase 1: Preprocessing   → src/preprocessing.py
Phase 2: Inference       → src/models/{vader,finbert,gpt}_model.py
Phase 3: Aggregation     → src/aggregation.py
Phase 4: Drift Detection → src/drift_detection.py
Phase 5: Evaluation      → src/evaluation.py
Phase 6: Model Registry  → MLflow (run_pipeline.py)
```

**Key files:**
- `run_pipeline.py` — orchestrates all phases via CLI flags (`--models`, `--skip_preprocessing`, `--finetune`)
- `src/preprocessing.py` — `PreprocessingPipeline`: loads CSVs, cleans tweets (HTML entities, URLs, language detection), deduplicates, applies time-based 60/20/20 split, saves parquet files
- `src/aggregation.py` — `AggregationEngine`: groups by (ticker, trading_date), computes `sentiment_score = (Buy - Sell) / classified_total`, adds 3-day and 7-day rolling averages, merges with stock prices
- `data/processed/` — output parquets: `prices_clean.parquet`, `tweets_train.parquet`, `tweets_val.parquet`, `tweets_test.parquet`

---

#### 2.2 Basic Data Versioning / Dataset Tracking
**What we did:** Raw data and code are version-controlled via Git. ML experiment runs (parameters, metrics, artifacts) are tracked in MLflow with nested run structure.

**Key files:**
- `.gitignore` — excludes generated parquets and result CSVs (raw data in Git, derived data regenerated)
- `mlruns/` — MLflow local artifact store (nested parent/child runs per pipeline execution)
- `run_pipeline.py` — logs `random_seed: 42`, all metrics, and artifact CSVs to MLflow per run
- `seed_data.py` — migrates CSV results to Supabase tables with upsert + conflict handling for cloud persistence

**MLflow experiment structure:**
```
Experiment: "stock_sentiment_prediction"
└── Parent Run: pipeline_run_{YYYYMMDD_HHMMSS}
    ├── Child Run: vader        (metrics + artifacts)
    ├── Child Run: finbert      (metrics + artifacts)
    ├── Child Run: finbert_finetuned
    └── Child Run: gpt4o_mini
```

---

#### 2.3 Reproducible ML Experiments with More Than One Model Run
**What we did:** Four models are run and compared in every pipeline execution. Global seed=42 is set across all libraries. Time-based split (no random shuffle) prevents data leakage.

**Key files:**
- `src/preprocessing.py` (lines 13–17) — `SEED = 42`, sets `random`, `numpy`, `torch`, and `cuda` seeds
- `src/evaluation.py` (line 16) — `SEED = 42` for bootstrap CI reproducibility
- `src/models/finbert_finetune.py` — `Trainer` with deterministic seed; logs all hyperparameters to MLflow

**Models compared per run:**
| Model | Type |
|---|---|
| VADER | Rule-based lexicon |
| FinBERT (Base) | Pre-trained transformer |
| FinBERT (Fine-tuned) | Custom fine-tune on silver + gold labels |
| Gemma3:4b (Ollama) | LLM few-shot classification |

---

### 3. Model Evaluation

#### 3.1 Clear Evaluation Metrics
**What we did:** Two complementary evaluation dimensions — classification quality vs. human labels, and predictive correlation with stock prices.

**Key file:** `src/evaluation.py`

**Classification metrics** (`evaluate_on_human_labels()`, lines 24–72):
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 (weighted and macro)
- Cohen's Kappa (inter-annotator agreement between 2 human annotators)

**Correlation metrics** (`evaluate_sentiment_price_correlation()`, lines 75–137):
- Pearson r: sentiment score vs. Daily Return % at lags 0, 1, 2, 3 days
- p-value and 95% bootstrap confidence interval (n=1000, seed=42)
- Computed per ticker (AAPL, TSLA, TSM) and combined

**Output files:**
- `results/classification_report_{model}.txt` — sklearn text report per model
- `results/inter_model_agreement.csv` — pairwise agreement matrix

---

#### 3.2 Basic Analysis of Model Robustness / Bias
**What we did:** Confidence thresholding to handle uncertain predictions; "No Opinion" class quantifies model uncertainty; inter-model agreement measures consistency.

**Key files:**
- `src/models/finbert_model.py` — confidence threshold of 0.6: predictions below threshold are labeled "No Opinion" rather than forcing a low-confidence class
- `src/evaluation.py` (`compare_models()`, lines 140–172) — model selection penalizes high No Opinion rate as a robustness signal
- `src/evaluation.py` (`inter_model_agreement()`, lines 175–188) — pairwise agreement rates across all model pairs on the same tweet IDs

**Bias awareness:**
- Data covers only 3 stocks (AAPL, TSLA, TSM) — English-language US/Taiwan market bias
- Language filter (`langdetect`) removes non-English tweets, reducing multilingual representation
- Human annotation uses 2 annotators; Cohen's Kappa measures inter-rater reliability
- Footer disclaimer on every dashboard page: *"For informational purposes only. Not financial advice."*

---

#### 3.3 Discussion of Model Limitations
Documented in `PROJECT_OVERVIEW.md`. Key limitations:
- Sentiment-price correlation is weak (finance is noisy; many external factors)
- VADER is domain-agnostic — financial jargon may be misclassified
- FinBERT fine-tuned on silver labels (VADER outputs) inherits VADER's errors
- Gemma3 requires local Ollama — not available in all deployment environments
- No real-time tweet ingestion; system operates on historical data only

---

### 4. Machine Learning System Design

#### 4.1 Overview of ML System Architecture
**What we did:** The system has three clearly separated runtime layers: a batch pipeline, an MLOps orchestration layer, and a serving layer.

```
Batch Pipeline (run_pipeline.py)
    └── Data → Preprocessing → Inference → Aggregation → Drift → Evaluation → Registry

MLOps Layer (Airflow)
    └── Scheduled retrain DAG with champion/challenger promotion

Serving Layer (Flask + MLflow)
    └── Dashboard reads from results CSVs / Supabase; MLflow serves model registry
```

**Key files:**
- `run_pipeline.py` — batch orchestration
- `dags/retrain_dag.py` — Airflow DAG for automated retraining
- `dashboard/app.py` — serving layer

---

#### 4.2 Clear Separation Between Data Processing, Training, and Inference
**What we did:** Each concern is in a dedicated module with no cross-coupling.

| Layer | File | Responsibility |
|---|---|---|
| Data processing | `src/preprocessing.py` | Load, clean, split |
| Aggregation | `src/aggregation.py` | Group by date, compute scores |
| Training | `src/models/finbert_finetune.py` | Fine-tune only, no inference |
| Inference | `src/models/{vader,finbert,gpt}_model.py` | Predict only, no training |
| Evaluation | `src/evaluation.py` | Metrics only, no model code |
| Drift | `src/drift_detection.py` | Flag generation only |

---

#### 4.3 How ML Components Connect to the Software System
**What we did:** Results flow from ML pipeline → CSV/Supabase → Flask API → browser.

```
run_pipeline.py
    ↓ writes
results/aggregated_{model}.csv
results/drift_flags_{model}.csv
    ↓ read by
dashboard/app.py (/api/sentiment, /api/drift, /api/day)
    ↓ consumed by
dashboard/templates/index.html (Chart.js, JavaScript fetch)
```

MLflow model registry (`mlruns/`) is used by:
- `run_pipeline.py` — to register the selected best model
- `dags/retrain_dag.py` — to query champion F1 and promote challenger

---

### 5. MLOps & Deployment

#### 5.1 Working Model Deployment Setup
**What we did:** Full Docker Compose stack with 5 services. Dashboard is served via Gunicorn (production WSGI). MLflow runs as a dedicated tracking server. Airflow handles scheduling.

**Key files:**
- `docker-compose.yml` — unified stack definition
- `Dockerfile.dashboard` — lightweight Python 3.11-slim image, Gunicorn on port 8080
- `Dockerfile.mlflow` — MLflow server on port 5000
- `Dockerfile.airflow` — Apache Airflow 2.9.3 with ML dependencies
- `render.yaml` — Render.com PaaS deployment config (alternative to Docker)

**Service URLs:**
| Service | Port |
|---|---|
| Dashboard | 8080 |
| MLflow UI | 5000 |
| Airflow UI | 8082 (admin/admin) |

---

#### 5.2 Basic Automation / CI/CD for ML Components
**What we did:** Airflow DAG (`dags/retrain_dag.py`) automates the full retrain-evaluate-promote cycle on a daily schedule, triggered only when drift is detected.

**Key file:** `dags/retrain_dag.py`

**DAG: `sentiment_retrain_pipeline`**
```
Schedule: 0 2 * * *  (daily at 02:00 UTC)

check_drift (ShortCircuitOperator)
    ├─ Reads drift_flags_*.csv for last 7 days
    ├─ If ≥30% of dates have any flag → continue
    └─ Otherwise → short-circuit (skip all downstream tasks)

retrain_finbert (PythonOperator, timeout=4h)
    └─ FinBERTFineTuner.run() on train split + human gold labels (5x oversampled)

evaluate_challenger (PythonOperator, timeout=1h)
    └─ FinBERTModel inference on test split → EvaluationEngine → push challenger_f1 to XCom

compare_champion (PythonOperator)
    └─ Load current Production model F1 from MLflow registry → push challenger_wins to XCom

promote_if_better (PythonOperator)
    └─ If challenger_wins: transition new version to Production, archive old champion
```

---

#### 5.3 How the Model Can Be Updated or Retrained
**Two paths:**

**Automatic (via Airflow):**
1. Drift flags accumulate in `results/drift_flags_*.csv`
2. DAG triggers at 02:00 UTC, checks drift ratio
3. If ≥30% drift → retrain → evaluate → promote if better F1

**Manual (via CLI):**
```bash
python run_pipeline.py --models finbert --finetune --finetune_epochs 5
```
This reruns the full pipeline and registers the new model in MLflow.

---

### 6. Testing & Maintainability

#### 6.1 Basic Testing for Data Processing / Model Pipelines
**What we have:**
- `notebooks/model_evaluation.ipynb` — exploratory evaluation notebook with per-model metric comparison, confusion matrices, and correlation plots
- `src/evaluation.py` — `EvaluationEngine` functions act as validation gates: pipeline only continues if metrics are computable
- MLflow artifact logging provides a persistent audit trail of every run's input/output

**Known gap:** No formal `pytest` test suite. This is identified as technical debt (see Section 6.3).

---

#### 6.2 Debugging Approaches
**What we did:**
- Python `logging` module used throughout all pipeline modules (`log.info`, `log.warning`, `log.error`)
- Airflow task logs capture stdout/stderr per task with retry on failure (`retries=1`, `retry_delay=10min`)
- MLflow artifacts preserve intermediate CSV outputs for post-hoc inspection
- `docker compose logs -f <service>` for container-level debugging
- `--skip_preprocessing` flag in `run_pipeline.py` allows re-running inference without reprocessing data, isolating pipeline stages

---

#### 6.3 Technical Debt and Improvement Ideas
- **No pytest suite** — unit tests for `PreprocessingPipeline`, `DriftDetector`, and `AggregationEngine` would prevent regression
- **No DVC** — data version control tool (DVC) would replace manual parquet regeneration
- **No rate limiting on login** — brute-force protection not implemented
- **No RBAC** — all authenticated users share the same view; no admin vs. read-only roles
- **Gemma3 dependency on local Ollama** — makes cloud deployment non-trivial without a dedicated GPU server
- **Single-node Airflow** — `LocalExecutor` does not scale; `CeleryExecutor` needed for parallel DAG tasks

---

### 7. Monitoring

#### 7.1 Simple Monitoring of Model Performance / Input Data
**What we did:** `DriftDetector` (`src/drift_detection.py`) produces four boolean flags per (ticker, date) after each pipeline run. These flags are displayed in the dashboard as red alert banners.

**Key file:** `src/drift_detection.py`

**Four drift signals:**

| Flag | Logic |
|---|---|
| `drift_flag` | Any label class (Buy/Hold/Sell/No Opinion %) shifts >15 percentage points vs. 7-day rolling baseline |
| `volume_spike_flag` | Daily tweet volume > rolling mean + 3×std over 7-day window |
| `weak_signal_flag` | No Opinion rate > 35% (model is uncertain on most tweets) |
| `divergence_flag` | Sentiment direction and price direction are opposite for 5+ consecutive days |

**Output:** `results/drift_flags_{model}.csv` — one row per (ticker, date) with all four flag columns.

---

#### 7.2 Demonstration of Data Drift / Performance Change
**What we did:**
- Dashboard (`dashboard/templates/index.html`) renders a **red drift alert banner** on the day-detail panel when any flag is active for the selected ticker/model/date
- `/api/drift` endpoint returns all flags for the current ticker and model selection
- Airflow DAG reads drift files and logs the drift ratio: *"Drift check: X/Y dates flagged (Z%)"*

**Dashboard API:**
```
GET /api/drift?ticker=AAPL&model=vader
→ { dates: [...], drift_flag: [...], volume_spike_flag: [...], ... }
```

---

#### 7.3 Basic Strategy for Handling Model Degradation
**What we did:** Champion/Challenger pattern in `dags/retrain_dag.py`:

1. Drift flags trigger automatic retraining (threshold: ≥30% of recent dates flagged)
2. Challenger model evaluated against human-labeled test set
3. Challenger F1 compared against current Production model F1 from MLflow registry
4. Only promoted if F1 improves — prevents regression from bad retrains
5. Old Production model archived (not deleted) for rollback

```python
# dags/retrain_dag.py
challenger_wins = challenger_f1 > champion_f1
# If True → transition to Production, archive champion
# If False → retain current champion, log result
```

---

### 8. Process and Teamwork

> This requirement is addressed in the written report and live presentation, not in code.

**What to cover in the report:**
- Development methodology used (e.g., iterative sprints, task division by module)
- Role assignments: who owned data pipeline, models, dashboard, MLOps, evaluation
- Collaboration challenges: environment differences (GPU/CPU), merge conflicts, Airflow DAG iteration

---

### 9. Responsible Use of Machine Learning

#### 9.1 Fairness / Explainability
**What we did:**
- **Confidence thresholding:** `src/models/finbert_model.py` uses a 0.6 confidence threshold — predictions below this are labeled "No Opinion" rather than forcing a low-confidence Buy/Hold/Sell label. This makes model uncertainty explicit.
- **Inter-model agreement:** `src/evaluation.py` (`inter_model_agreement()`) measures how consistently different models agree, identifying tweets that are inherently ambiguous.
- **Cohen's Kappa:** Measures inter-annotator reliability between 2 human annotators on gold labels, quantifying subjectivity in the ground truth itself.

---

#### 9.2 Privacy / Ethical Concerns
**What we considered:**
- All tweet data is publicly sourced; no private messages or account credentials are processed
- Supabase `users` table stores only bcrypt-hashed passwords — plaintext passwords never persisted
- The system is a **decision-support tool**, not an automated trading system — human judgment is required before any financial action
- System covers only 3 tickers with historical data — outputs should not be generalized to other stocks or time periods

---

#### 9.3 Communication of System Limitations
**What we did:**
- Footer disclaimer rendered on every dashboard page: *"For informational purposes only. Not financial advice."*
- "No Opinion" label class is shown prominently in the stacked bar chart — users can see when the model is uncertain
- Dashboard shows Pearson r and p-value, allowing users to assess whether correlation is statistically meaningful

**Key limitations communicated:**
- Correlation between sentiment and price is weak by design (financial markets are complex)
- Data covers Sept 2021–Sept 2022 only — model may not generalize to different market regimes
- English-only filter removes multilingual investor sentiment
- Fine-tuned FinBERT trained on VADER silver labels inherits VADER's classification errors

---

## Coverage Summary

| # | Requirement | Status | Key Files |
|---|---|---|---|
| 1.1 | Real-world ML use case | **Done** | `run_pipeline.py`, `src/models/` |
| 1.2 | User access control | **Done** | `dashboard/auth.py`, `dashboard/app.py` |
| 1.3 | Front-end UI | **Done** | `dashboard/templates/index.html`, `dashboard/app.py` |
| 2.1 | Data pipeline | **Done** | `src/preprocessing.py`, `src/aggregation.py` |
| 2.2 | Data versioning | **Done** | `mlruns/`, Git, `seed_data.py` |
| 2.3 | Reproducible experiments | **Done** | `run_pipeline.py`, `src/preprocessing.py` |
| 3.1 | Evaluation metrics | **Done** | `src/evaluation.py` |
| 3.2 | Robustness / bias analysis | **Done** | `src/models/finbert_model.py`, `src/evaluation.py` |
| 3.3 | Model limitations | **Done** | `PROJECT_OVERVIEW.md`, dashboard footer |
| 4.1 | ML system architecture | **Done** | `run_pipeline.py`, `dags/retrain_dag.py` |
| 4.2 | Separation of concerns | **Done** | `src/` module structure |
| 4.3 | ML-to-software integration | **Done** | `dashboard/app.py` APIs |
| 5.1 | Deployment setup | **Done** | `docker-compose.yml`, `Dockerfile.*` |
| 5.2 | Automation / CI for ML | **Done** | `dags/retrain_dag.py` |
| 5.3 | Model update strategy | **Done** | `dags/retrain_dag.py`, `run_pipeline.py` |
| 6.1 | Testing | **Partial** | `notebooks/model_evaluation.ipynb` (no pytest) |
| 6.2 | Debugging | **Done** | Logging, Airflow logs, MLflow artifacts |
| 6.3 | Technical debt | **Done** | Identified above |
| 7.1 | Monitoring | **Done** | `src/drift_detection.py` |
| 7.2 | Drift demonstration | **Done** | `dashboard/app.py` `/api/drift`, dashboard alerts |
| 7.3 | Degradation strategy | **Done** | `dags/retrain_dag.py` champion/challenger |
| 8 | Process & teamwork | **Report only** | Written report + presentation |
| 9.1 | Fairness / explainability | **Partial** | Confidence threshold, Cohen's Kappa, agreement |
| 9.2 | Privacy / ethics | **Done** | bcrypt, disclaimer, decision-support framing |
| 9.3 | Limitations communication | **Done** | Dashboard footer, "No Opinion" class |
