# Stock Trend Prediction from Twitter Posts
## Project Overview for Claude Code — Implementation Planning

> **Status:** Not yet implemented. This document is the single source of truth for the project.
> Claude Code should read this file first and use it to plan a full implementation roadmap before writing any code.

---

## 1. Project Summary

This is a **Software Engineering for Machine Learning (SE4ML) course project**.
The goal is to build a complete, end-to-end ML pipeline that:

1. Reads raw tweet data about three stock tickers (AAPL, TSLA, TSM)
2. Classifies each tweet into a sentiment label: **Buy / Hold / Sell / No Opinion**
3. Aggregates daily sentiment signals per ticker
4. Correlates sentiment signals with historical stock price movements
5. Selects the single best-performing model using predefined criteria
6. Serves results via a local web dashboard

The system is a **decision-support tool for retail investors**, not an automated trading system.
Every output must include a disclaimer: *"For informational purposes only. Not financial advice."*

---

## 2. Datasets (CSV — Already Available, No Download Needed)

### Dataset 1: `stock_prices.csv`
Historical OHLCV data for three tickers.

| Column | Type | Notes |
|---|---|---|
| Date | string | Format: `DD/MM/YYYY` (e.g., `30/9/2021`) |
| Open | float | Opening price |
| High | float | Daily high |
| Low | float | Daily low |
| Close | float | Closing price |
| Adj Close | float | Adjusted closing price |
| Volume | int | Trading volume |
| Stock Name | string | `AAPL`, `TSLA`, or `TSM` |
| Daily Return % | float | `((Close - prev_Close) / prev_Close) * 100` — may be empty for first row |
| Intraday Trend | int | `1` = bullish day, `0` = bearish/flat day |

**Example row:**
```
30/9/2021, 143.66, 144.38, 141.28, 141.5, 140.48, 89056700, AAPL, (empty), 0
1/10/2021, 141.9, 142.92, 139.11, 142.65, 141.62, 94639600, AAPL, 0.8127, 1
```

---

### Dataset 2: `tweets.csv`
Raw tweet data aligned to trading dates.

| Column | Type | Notes |
|---|---|---|
| Date | string | Format: `DD/MM/YYYY HH:MM` (e.g., `29/9/2022 23:41`) |
| Tweet | string | Raw tweet text — may contain URLs, HTML entities, emojis |
| Stock Name | string | `AAPL`, `TSLA`, or `TSM` |
| Company Name | string | Full company name (e.g., `Tesla, Inc.`) |
| Trading Date | string | Format: `DD/MM/YYYY` — the market day this tweet is assigned to |

**Example rows:**
```
29/9/2022 23:41, "Mainstream media has done an amazing job at brainwashing people...", TSLA, Tesla Inc., 29/9/2022
29/9/2022 23:24, "Tesla delivery estimates are at around 364k from the analysts. $tsla", TSLA, Tesla Inc., 29/9/2022
```

**Known data issues to handle:**
- HTML entities in tweet text (e.g., `&amp;`, `&#x201C;`, `&lt;`)
- URLs embedded in tweets
- Non-English tweets present
- Possible duplicate tweets
- Missing `Daily Return %` on first row per ticker

---

## 3. ML Task Definition

### Task Type
Multi-class text classification (per tweet, independently).

### Label Schema

| Label | Meaning | Example Tweet Signal |
|---|---|---|
| `Buy` | Bullish / positive outlook | "AAPL crushed earnings 🚀" |
| `Sell` | Bearish / negative outlook | "Dumping all my TSLA shares" |
| `Hold` | Neutral / wait-and-see | "Not sure about TSM right now" |
| `No Opinion` | Irrelevant or unclassifiable | "Good morning everyone!" |

**Important:** No ground-truth labels exist in the dataset. All labels are model-inferred.
A **200-tweet human-labeled subset** (sampled from the test split) will serve as a semi-ground-truth anchor for F1/Accuracy evaluation.

### Data Split — TIME-BASED ONLY
Split ordered by `Trading Date`. **Never use random splitting** (would leak future data into training).

| Split | Proportion | Purpose |
|---|---|---|
| Train | 60% | Model configuration, threshold tuning |
| Validation | 20% | Confidence threshold selection, inter-model calibration |
| Test | 20% | Final evaluation — held out until all models are configured |

---

## 4. Three Models to Compare

The project compares three models. **No model is pre-selected as the winner.**
Final selection happens after all experiments complete, using the decision criteria in Section 6.

### Model 1: VADER (Baseline)
- **Type:** Rule-based lexicon (no training, no GPU)
- **Library:** `vaderSentiment`
- **Label mapping:**
  - `compound >= 0.05` → `Buy`
  - `compound <= -0.05` → `Sell`
  - `-0.05 < compound < 0.05` → `Hold`
  - `token count < 3` → `No Opinion`
- **Speed:** ~10,000 tweets/sec on CPU (~5 seconds for full dataset)
- **Cost:** $0

### Model 2: FinBERT
- **Type:** Neural — BERT pretrained on financial text
- **Model:** `ProsusAI/finbert` (HuggingFace)
- **Label mapping:** FinBERT outputs `positive/negative/neutral` → map to `Buy/Sell/Hold`
- **No Opinion rule:** if `max softmax confidence < 0.6` → `No Opinion`
- **Batch size:** 32 (fits in 4GB VRAM; Google Colab T4 compatible)
- **Speed:** ~200–500 tweets/sec on GPU; ~2–4 min for full dataset
- **Cost:** $0 (local GPU or Colab free tier)

### Model 3: GPT-4o-mini (LLM, Few-Shot)
- **Type:** LLM via OpenAI API
- **Model string:** `gpt-4o-mini`
- **Temperature:** `0` (reproducibility)
- **Approach:** Few-shot prompting (8 examples: 2 per class)
- **Batch strategy:** Send 20 tweets per API call as a numbered list; parse numbered response
- **Retry logic:** Exponential backoff, max 3 retries on rate limit errors
- **Speed:** ~50–200 tweets/sec (API rate-limited); ~5–18 min for full dataset
- **Cost:** ~$5–15 for full 53K tweet inference
  - Input: $0.150 / 1M tokens
  - Output: $0.600 / 1M tokens
- **Requires:** `OPENAI_API_KEY` environment variable

**Few-shot prompt examples (must use exactly these):**

```
System: You are a financial sentiment classifier. Classify tweets about stocks into
exactly one of: Buy, Hold, Sell, No Opinion.
Buy = bullish/positive outlook. Sell = bearish/negative outlook.
Hold = neutral/wait-and-see. No Opinion = irrelevant or unclassifiable.
Respond with ONLY the label, nothing else.

Examples:
"$AAPL just crushed earnings, this stock is going to the moon! 🚀" → Buy
"Apple's new iPhone demand is insane, definitely adding more shares" → Buy
"Tesla recalls again, I'm dumping all my shares TSLA" → Sell
"Really bad quarter for $AAPL, revenue miss and lowered guidance" → Sell
"Not sure about TSLA right now, mixed signals from the market" → Hold
"$TSM is flat today, waiting for the next earnings before deciding" → Hold
"Good morning everyone, happy Monday!" → No Opinion
"Just had the best coffee of my life" → No Opinion
```

---

## 5. Evaluation Metrics

Since no ground-truth trading labels exist, evaluation uses multiple complementary metrics:

### 5.1 Metrics on Human-Labeled Subset (200 tweets)
- **Accuracy** (sklearn)
- **Precision, Recall, F1** — weighted and macro average
- **Cohen's Kappa** — inter-annotator agreement between the 2 human annotators
- Full **classification report** (sklearn `classification_report`)

> The 200 tweets are randomly sampled from the **test split** and labeled by 2 team members using Label Studio.

### 5.2 Sentiment-Price Correlation
- **Pearson r** between `sentiment_score` and `next-day Daily Return %`
- Computed at **lag = 0, 1, 2, 3 days** (using `.shift()`)
- For **each ticker separately** (AAPL, TSLA, TSM) AND combined
- Include: **p-value** (scipy) + **95% bootstrap confidence interval** (n=1000 resamples, numpy)
- `sentiment_score = (Buy_count - Sell_count) / total_classified`

### 5.3 Supporting Metrics (Per Model)
| Metric | Description |
|---|---|
| No Opinion Rate | % of tweets labeled No Opinion — proxy for model confidence |
| Inter-model Agreement Rate | % agreement between each model pair (VADER↔FinBERT, VADER↔GPT, FinBERT↔GPT) |
| Label Distribution | [%Buy, %Hold, %Sell, %NoOpinion] per ticker per day |
| Processing Time | Wall-clock seconds for full test split inference |
| API Cost (GPT only) | Actual USD spent (input tokens + output tokens) |

---

## 6. Model Selection Criteria

After all experiments complete, select **one** model using this decision table (in priority order):

| Priority | Criterion | Threshold to Pass |
|---|---|---|
| 1 | Weighted F1 on human-labeled subset | Higher is better; must be > 0.50 to be considered |
| 2 | Pearson r (best lag, combined tickers) | Higher absolute value is better |
| 3 | No Opinion Rate | Lower is better; < 20% preferred |
| 4 | Inter-model agreement with majority | Higher agreement = more reliable signal |
| 5 | Inference cost | VADER=$0, FinBERT=$0, GPT=$5–15 |
| 6 | Processing time | VADER<FinBERT<GPT |

The winning model is registered in the **MLflow Model Registry** as `stock_sentiment_classifier` with stage `Production`.

---

## 7. MLflow Experiment Tracking

**Every** model run must be logged to MLflow. No exceptions.

### Experiment Structure
```
Experiment: "stock_sentiment_prediction"
└── Parent Run: pipeline_run_{timestamp}
    ├── Child Run: vader
    ├── Child Run: finbert
    └── Child Run: gpt4o_mini
```

### What to Log Per Model Run

**Parameters (log once at start):**
- `model_type` (vader / finbert / gpt)
- `dataset_split` (test)
- `ticker_scope` (AAPL,TSLA,TSM)
- Model-specific: `batch_size`, `confidence_threshold`, `temperature`, `num_few_shot_examples`, etc.

**Metrics (log after inference):**
- `buy_rate`, `sell_rate`, `hold_rate`, `no_opinion_rate`
- `processing_time_seconds`
- `total_tweets_processed`
- `avg_confidence` (FinBERT only)
- `total_tokens_used`, `estimated_cost_usd`, `retry_count` (GPT only)

**After evaluation — add to each model's run:**
- `accuracy`, `f1_weighted`, `f1_macro`, `precision_weighted`, `recall_weighted`
- `pearson_r_AAPL_lag0` through `lag3`, same for TSLA and TSM
- `pearson_p_AAPL_lag0` through `lag3` (p-values)
- `pearson_r_combined_best_lag`

**Artifacts:**
- `results/{model}_results.csv` — raw inference output
- `results/classification_report_{model}.txt`
- `results/aggregated_{model}.csv`
- `results/drift_flags_{model}.csv`

### Model Registry (after selection)
```python
mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name="stock_sentiment_classifier"
)
# Transition to Production stage
client.transition_model_version_stage(
    name="stock_sentiment_classifier",
    version=1,
    stage="Production"
)
```

**MLflow storage:** Local `./mlruns/` directory (no server needed).
**View UI:** `mlflow ui --port 5000` → `http://localhost:5000`

---

## 8. Data Drift Detection

Monitor for distribution shifts using these flags:

| Flag | Trigger Condition | Column Name |
|---|---|---|
| `drift_flag` | Any label class shifts > 15 percentage points vs 7-day rolling baseline | `drift_flag` |
| `volume_spike_flag` | Tweet volume > mean + 3×std of 7-day window | `volume_spike_flag` |
| `weak_signal_flag` | No Opinion rate > 35% | `weak_signal_flag` |
| `divergence_flag` | Sentiment and price move in opposite directions for 5+ consecutive days | `divergence_flag` |

Flags are added as boolean columns to the aggregated DataFrame and saved to `results/drift_flags_{model}.csv`.
Active flags trigger alert banners in the dashboard.

---

## 9. Web Dashboard Requirements

**Stack:** Flask (backend) + Chart.js + Bootstrap 5 (frontend) — dark theme.

### Routes
| Route | Method | Description |
|---|---|---|
| `/` | GET | Main dashboard page |
| `/api/sentiment` | GET | Params: `ticker`, `model` → returns aggregated sentiment + price JSON |
| `/api/drift` | GET | Params: `ticker`, `model` → returns drift flags JSON |
| `/api/compare` | GET | Returns model comparison metrics JSON |

### Dashboard UI Components
1. **Ticker selector** — dropdown: AAPL / TSLA / TSM
2. **Model selector** — dropdown: VADER / FinBERT / GPT-4o-mini
3. **Date range slider** — filter the time window
4. **Chart 1 (dual-axis line chart):** sentiment score (left Y) + stock close price (right Y) vs date (X)
5. **Chart 2 (stacked bar chart):** daily Buy / Hold / Sell / No Opinion % breakdown
6. **Drift alert banner** — red banner shown when any drift flag is active for selected ticker
7. **Metrics panel** — shows Pearson r, p-value, and best lag for selected ticker + model
8. **Footer disclaimer** — *"For informational purposes only. Not financial advice."* — visible on every page

---

## 10. Project Folder Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── stock_prices.csv          # Dataset 1 — place here
│   │   └── tweets.csv                # Dataset 2 — place here
│   └── processed/
│       ├── tweets_train.parquet
│       ├── tweets_val.parquet
│       ├── tweets_test.parquet
│       └── prices_clean.parquet
│
├── src/
│   ├── preprocessing.py              # PreprocessingPipeline class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vader_model.py            # VADERModel class
│   │   ├── finbert_model.py          # FinBERTModel class
│   │   └── gpt_model.py             # GPTModel class
│   ├── aggregation.py               # AggregationEngine class
│   ├── evaluation.py                # EvaluationEngine class
│   └── drift_detection.py           # DriftDetector class
│
├── dashboard/
│   ├── app.py                        # Flask application
│   ├── templates/
│   │   └── index.html                # Main dashboard page
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── charts.js
│
├── notebooks/
│   └── exploration.ipynb             # EDA — optional
│
├── results/                          # Auto-created on first run
│   ├── vader_results.csv
│   ├── finbert_results.csv
│   ├── gpt_results.csv
│   ├── aggregated_vader.csv
│   ├── aggregated_finbert.csv
│   ├── aggregated_gpt.csv
│   ├── drift_flags_vader.csv
│   ├── drift_flags_finbert.csv
│   ├── drift_flags_gpt.csv
│   └── classification_report_{model}.txt
│
├── mlruns/                           # Auto-created by MLflow — do not edit manually
│
├── human_labels/
│   └── human_labels.csv             # Columns: tweet_id, annotator_1_label, annotator_2_label, final_label
│
├── run_pipeline.py                   # Master orchestration script
├── requirements.txt                  # All pinned dependencies
├── .env                              # API keys — never commit to git
├── .env.example                      # Template for .env
└── .gitignore                        # Includes: .env, mlruns/, __pycache__/, *.pyc
```

---

## 11. Module Specifications

### `preprocessing.py` — `PreprocessingPipeline`
**Responsibilities:**
- Load both CSVs; parse dates with `dayfirst=True`
- Tweet cleaning: decode HTML entities (`html.unescape`), remove URLs (regex), strip whitespace, preserve emojis
- Filter: AAPL/TSLA/TSM only; English only (`langdetect`); remove duplicates
- Align tweets to `Trading Date` column
- Price cleaning: forward-fill missing days; compute missing `Daily Return %`
- Time-based 60/20/20 split by `Trading Date`
- Save processed splits to `data/processed/` as parquet
- Return summary dict: `{split: {ticker: row_count}}`

---

### `vader_model.py` — `VADERModel`
**Responsibilities:**
- Classify all tweets in test split using VADER compound score
- Process in batches of 1,000
- Output DataFrame: `[tweet_id, trading_date, ticker, label, confidence, compound_score]`
- Save to `results/vader_results.csv`
- Log to MLflow nested run

---

### `finbert_model.py` — `FinBERTModel`
**Responsibilities:**
- Load `ProsusAI/finbert`; auto-detect GPU/CPU
- Classify in batches of 32 with tqdm progress bar
- Map `positive→Buy`, `negative→Sell`, `neutral→Hold`; low confidence→`No Opinion`
- Output DataFrame: `[tweet_id, trading_date, ticker, label, confidence, positive_score, negative_score, neutral_score]`
- Save to `results/finbert_results.csv`
- Log to MLflow nested run

---

### `gpt_model.py` — `GPTModel`
**Responsibilities:**
- Load `OPENAI_API_KEY` from `.env`
- Few-shot classify in batches of 20 tweets per API call
- Implement exponential backoff retry (max 3) for rate limit errors
- Track and log actual token usage and cost
- Output DataFrame: `[tweet_id, trading_date, ticker, label, raw_response]`
- Save to `results/gpt_results.csv`
- Log to MLflow nested run

---

### `aggregation.py` — `AggregationEngine`
**Responsibilities:**
- Accept any model results DataFrame + prices DataFrame
- Compute per ticker per trading_date:
  - `sentiment_score = (Buy_count - Sell_count) / total_classified`
  - `buy_pct`, `hold_pct`, `sell_pct`, `no_opinion_pct`
  - `tweet_volume`
  - `rolling_3day_sentiment`, `rolling_7day_sentiment`
- Merge with stock price data (join on ticker + trading_date)
- Save to `results/aggregated_{model_name}.csv`

---

### `evaluation.py` — `EvaluationEngine`
**Responsibilities:**

`evaluate_on_human_labels(results_df, human_labels_csv)`:
- Compute Accuracy, Precision, Recall, F1 (weighted + macro)
- Compute Cohen's Kappa between two annotators
- Print sklearn classification report
- Save report to `results/classification_report_{model}.txt`

`evaluate_sentiment_price_correlation(aggregated_df)`:
- Pearson r + p-value + 95% bootstrap CI (n=1000) for lag 0/1/2/3
- Per ticker and combined
- Print formatted results table

`compare_models(vader_df, finbert_df, gpt_df)`:
- Inter-model agreement matrix (3 pairs)
- Summary comparison table
- Return name of best model by decision criteria

---

### `drift_detection.py` — `DriftDetector`
**Responsibilities:**
- Compute 7-day rolling label distribution baseline
- Assign `drift_flag`, `volume_spike_flag`, `weak_signal_flag`, `divergence_flag`
- Save to `results/drift_flags_{model_name}.csv`

---

### `run_pipeline.py` — Master Orchestration
**CLI usage:**
```bash
python run_pipeline.py --models vader finbert gpt --ticker all
python run_pipeline.py --models vader --ticker AAPL   # run just one model
python run_pipeline.py --skip_gpt False               # confirm before spending API $
```

**Execution order:**
1. Load and preprocess data
2. Run selected models on test split (with MLflow tracking)
3. Aggregate results per model
4. Run drift detection per model
5. Run evaluation (correlation + model comparison)
6. Print final model recommendation with justification
7. Register winning model in MLflow Model Registry
8. Print: `mlflow ui --port 5000` to view results

---

### `dashboard/app.py` — Flask App
**Responsibilities:**
- Serve `index.html` on `/`
- Load aggregated CSVs and drift flag CSVs at startup
- Expose JSON API routes: `/api/sentiment`, `/api/drift`, `/api/compare`
- All API responses include ticker, model, date range filters

---

## 12. Dependencies (`requirements.txt`)

```
# Data
pandas==2.2.2
numpy==1.26.4
pyarrow==16.0.0

# NLP / ML
vaderSentiment==3.3.2
transformers==4.41.2
torch==2.3.1
openai==1.30.1
scikit-learn==1.5.0
scipy==1.13.1
langdetect==1.0.9

# Experiment Tracking
mlflow==2.13.2

# Web Dashboard
flask==3.0.3

# Utilities
tqdm==4.66.4
python-dotenv==1.0.1

# Annotation (optional, local tool)
# label-studio -- install separately via pip if needed
```

---

## 13. Reproducibility Requirements

All of the following must be set at the top of every script:
```python
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

Every MLflow run logs `random_seed: 42` as a parameter.

---

## 14. Professor / Grading Requirements to Satisfy

The following are the explicit grading criteria from the course assignment (Progress Report 3 rubric). Every item must be addressed in the final codebase:

### ✅ Project Overview
- Clear problem statement and use case
- ML task type documented
- Dataset described (size, tickers, date range)
- Three models listed with rationale
- Current system status trackable (via MLflow)

### ✅ MLOps & CI-CD Orchestration
- End-to-end pipeline: data → preprocessing → inference → evaluation → deployment
- Manual steps documented; automation roadmap clear
- Data drift detection implemented (4 flag types)
- Deployment mode justified (batch chosen over online — rationale in code comments)
- Model optimization considered (VADER: CPU; FinBERT: batch size tuning; GPT: temperature=0)

### ✅ Model Serving
- Clear input/output schema (ticker → sentiment chart + metrics)
- Batch prediction mode implemented and justified
- Working local demo (Flask dashboard)
- Example request/response demonstrable

### ✅ Process & Teams
- Role separation: preprocessing / model integration / evaluation / dashboard
- Interface contracts between modules (fixed DataFrame schemas)
- Experiment-driven iterative workflow (MLflow tracks each iteration)

### ✅ Responsible ML Engineering
- **Reproducibility:** fixed seeds, pinned deps, MLflow versioning
- **Explainability:** VADER labels traceable to lexicon rules; GPT few-shot prompt visible in UI
- **Fairness:** ticker imbalance acknowledged (AAPL=70% of data); volume normalization applied
- **Safety:** disclaimer on every dashboard page; divergence alert when signal is weak
- **Security:** no PII used; OpenAI calls exclude usernames; `.env` for API keys
- **Transparency:** system limitations documented in dashboard footer; MLflow logs all decisions

---

## 15. Known Limitations to Document

Include these in code comments and dashboard UI:

1. No ground-truth trading outcome labels — all evaluation is indirect
2. Dataset is historical and static (Sept 2021 – Sept 2022); does not generalize to future data without re-evaluation
3. GPT-4o-mini behavior may change with model updates (reproducibility risk for future runs)
4. FinBERT attention-based explainability not implemented (future work)
5. System does not account for market microstructure, insider information, or macroeconomic factors
6. Correlation ≠ causation — sentiment signal may lag, lead, or be spurious

---

## 16. Implementation Order Recommendation for Claude Code

Claude Code should implement modules in this order to enable incremental testing:

```
Phase 1 — Core Pipeline (no GPU, no API key needed)
  1. preprocessing.py         ← test with both CSVs immediately
  2. vader_model.py           ← validate full pipeline end-to-end
  3. aggregation.py           ← verify sentiment score computation
  4. evaluation.py            ← verify Pearson correlation logic
  5. drift_detection.py       ← verify flag logic

Phase 2 — Advanced Models
  6. finbert_model.py         ← requires GPU or Colab
  7. gpt_model.py             ← requires OPENAI_API_KEY and budget approval

Phase 3 — Tracking & Serving
  8. Add MLflow logging to all Phase 1 + Phase 2 modules
  9. run_pipeline.py          ← wire everything together
  10. dashboard/app.py        ← Flask backend
  11. dashboard/templates/index.html + static/ ← frontend

Phase 4 — Evaluation & Selection
  12. Run all 3 models on test split
  13. Annotate 200-tweet human-labeled subset (manual step)
  14. Run evaluation.py.compare_models() → select winner
  15. Register winning model in MLflow Model Registry
```

---

## 17. `.env.example` Template

```
# Copy this file to .env and fill in your values
# Never commit .env to git

OPENAI_API_KEY=your_openai_api_key_here
MLFLOW_TRACKING_URI=./mlruns
FLASK_PORT=8080
FLASK_DEBUG=True
```

---

## 18. `.gitignore` Requirements

```
.env
mlruns/
__pycache__/
*.pyc
*.pyo
data/processed/
results/
*.egg-info/
.DS_Store
```

---

*End of Project Overview Document*
*This file is the canonical reference for all implementation decisions.*
*When in doubt, refer back to this document before writing code.*
