# C4 Architecture Diagram — Stock Sentiment Analysis System

> **System:** Stock Sentiment Analysis System  
> **Purpose:** End-to-end ML pipeline that classifies stock-related tweets (AAPL, TSLA, TSM) into sentiment signals (Buy/Hold/Sell/No Opinion), correlates them with price movements, detects data drift, and triggers automated retraining.

---

## Level 1 — System Context Diagram

Shows who uses the system and what external systems it depends on.

```mermaid
C4Context
    title System Context Diagram — Stock Sentiment Analysis System

    Person(investor, "Retail Investor", "Views sentiment insights and live simulation via the web dashboard")
    Person(ml_engineer, "ML Engineer", "Runs pipeline, tunes hyperparameters, monitors experiments")
    Person(data_engineer, "Data Engineer", "Seeds data into Supabase, manages preprocessing")
    Person(sysadmin, "System Admin", "Deploys containers, manages Airflow DAGs and environment config")

    System(system, "Stock Sentiment Analysis System", "Classifies tweet sentiment per ticker, aggregates signals, detects drift, serves insights, and retrains models automatically")

    System_Ext(supabase, "Supabase", "Cloud PostgreSQL database. Stores users, aggregated results, drift flags, and live simulation state")
    System_Ext(huggingface, "Hugging Face Hub", "Hosts pre-trained FinBERT weights (ProsusAI/finbert)")
    System_Ext(ollama, "Ollama (Local LLM)", "OpenAI-compatible local inference server running gemma3:4b for free LLM classification")
    System_Ext(openai, "OpenAI API", "Optional cloud LLM (GPT-4o-mini). Used when Ollama is unavailable or higher quality is needed")
    System_Ext(csv_data, "Static Data Files", "tweets.csv (~53k tweets), stock_prices.csv (OHLCV Sept 2021–Sept 2022), human_labels/lebeled.csv (200 gold-labeled tweets)")

    Rel(investor, system, "Views sentiment dashboard, runs live simulation", "HTTPS")
    Rel(ml_engineer, system, "Runs pipeline CLI, views MLflow tracking", "CLI / Browser")
    Rel(data_engineer, system, "Seeds data, monitors preprocessing", "CLI / Browser")
    Rel(sysadmin, system, "Deploys via Docker Compose, configures .env", "SSH / Terminal")

    Rel(system, supabase, "Reads/writes users, results, drift flags, simulation state", "Supabase SDK / REST")
    Rel(system, huggingface, "Downloads pre-trained FinBERT model weights at startup", "HTTPS")
    Rel(system, ollama, "Sends tweet batches for LLM classification", "HTTP / OpenAI-compatible API")
    Rel(system, openai, "Optional: sends tweet batches for GPT-4o-mini classification", "HTTPS / OpenAI SDK")
    Rel(system, csv_data, "Reads raw tweets, prices, and gold labels as pipeline input", "File I/O")
```

### Design Decisions & Trade-offs (Context Level)

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Ollama as default LLM backend** | Zero API cost, runs locally, supports multiple open-source models | Slower inference (5–18 min vs. seconds), quality varies by model |
| **Supabase for cloud persistence** | Managed PostgreSQL with Auth and SDK — no infrastructure overhead | Requires internet; local fallback reads from CSV files |
| **Static CSV data source** | Historical dataset allows reproducible experiments and controlled evaluation | No live tweet ingestion; system is research/decision-support only |
| **Decision-support framing** | "Not financial advice" disclaimer prevents regulatory exposure | Limits productization potential |

---

## Level 2 — Container Diagram

Shows the major deployable units, their technologies, and how they communicate.

```mermaid
C4Container
    title Container Diagram — Stock Sentiment Analysis System

    Person(investor, "Retail Investor")
    Person(ml_engineer, "ML Engineer")
    Person(sysadmin, "System Admin")

    System_Boundary(sys, "Stock Sentiment Analysis System") {

        Container(dashboard, "Flask Dashboard", "Python 3.11, Flask 3.0, Gunicorn, Jinja2, Chart.js, Bootstrap 5", "Serves the web UI with ticker/date selectors, sentiment vs. price charts, drift alerts, model comparison, and live simulation. Handles user authentication.")

        Container(mlflow, "MLflow Tracking Server", "Python 3.11, MLflow 2.13+", "Stores experiment runs, parameters, metrics, and model artifacts. Hosts the model registry with champion/challenger promotion.")

        Container(airflow_web, "Airflow Webserver", "Apache Airflow 2.9.3, Python 3.11", "Web UI for monitoring and triggering DAGs. Exposes DAG run status and task logs.")

        Container(airflow_sched, "Airflow Scheduler", "Apache Airflow 2.9.3, Python 3.11", "Executes the retraining DAG on a daily schedule (02:00 UTC). Checks drift flags, retrains FinBERT, evaluates challenger, promotes if better.")

        Container(airflow_db, "Airflow Metadata DB", "PostgreSQL 15-alpine", "Stores Airflow DAG state, task instances, XCom values, and scheduler metadata.")

        Container(ml_pipeline, "ML Pipeline (CLI)", "Python 3.11, PyTorch 2.3, Transformers 4.41, scikit-learn, pandas, MLflow SDK", "Orchestrated via run_pipeline.py. Runs preprocessing, model inference (VADER/FinBERT/GPT), aggregation, drift detection, evaluation, and MLflow logging.")

        ContainerDb(local_fs, "Local File Store", "Parquet files, CSV files, ./mlruns/", "Stores processed parquet splits (train/val/test), inference result CSVs, aggregated result CSVs, drift flag CSVs, and MLflow artifact store.")
    }

    System_Ext(supabase, "Supabase", "Cloud PostgreSQL + Auth")
    System_Ext(huggingface, "Hugging Face Hub")
    System_Ext(ollama, "Ollama (Local LLM)")
    System_Ext(openai, "OpenAI API (optional)")

    Rel(investor, dashboard, "Views dashboard, runs simulation", "HTTPS / Browser")
    Rel(ml_engineer, ml_pipeline, "Runs pipeline CLI with flags", "Terminal / CLI")
    Rel(ml_engineer, mlflow, "Views experiments and model registry", "Browser / HTTP")
    Rel(sysadmin, airflow_web, "Monitors and triggers DAGs", "Browser / HTTP")

    Rel(dashboard, supabase, "Reads aggregated results, drift flags; reads/writes simulation state; authenticates users", "Supabase SDK")
    Rel(dashboard, local_fs, "Reads aggregated CSVs at startup (fallback if Supabase unavailable)", "File I/O")

    Rel(ml_pipeline, local_fs, "Reads raw CSVs and parquets; writes result CSVs and parquets", "File I/O")
    Rel(ml_pipeline, mlflow, "Logs runs, metrics, params, artifacts; registers models", "MLflow SDK / HTTP")
    Rel(ml_pipeline, huggingface, "Downloads FinBERT model weights", "HTTPS")
    Rel(ml_pipeline, ollama, "Sends tweet batches for LLM classification", "HTTP")
    Rel(ml_pipeline, openai, "Optional LLM classification (GPT-4o-mini)", "HTTPS")

    Rel(airflow_sched, local_fs, "Reads drift flag CSVs; writes retrained model to models/", "File I/O")
    Rel(airflow_sched, mlflow, "Reads champion model F1; registers and promotes challenger", "MLflow SDK")
    Rel(airflow_sched, huggingface, "Downloads FinBERT base weights for fine-tuning", "HTTPS")
    Rel(airflow_sched, airflow_db, "Reads/writes DAG state, XCom values", "PostgreSQL")
    Rel(airflow_web, airflow_db, "Reads DAG/task run state", "PostgreSQL")
```

### Design Decisions & Trade-offs (Container Level)

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **File system as primary inter-container bus** | Simple, no broker overhead, suitable for batch pipeline with low throughput | No real-time streaming; containers must share a Docker volume or local mount |
| **MLflow as model registry** | Provides experiment tracking + model versioning in one tool with minimal setup | SQLite backend is not HA; needs migration to PostgreSQL for production scale |
| **Airflow for retraining orchestration** | Industry-standard scheduler with XCom, retry, and short-circuit operator support | Heavy dependency; overkill for a single daily DAG in research context |
| **Separate Airflow and Dashboard services** | Clean separation of concerns: orchestration vs. serving | Requires shared volume and synced model paths between containers |
| **Gunicorn in Dashboard container** | Multi-worker WSGI server for production readiness | Requires careful worker count tuning (CPU-bound workers for Flask) |
| **Local fallback for Supabase** | Dashboard remains functional when cloud is unreachable | Data may be stale; no live simulation without Supabase |

---

## Level 3 — Component Diagram (ML Pipeline Only)

Shows the internal components of the `ML Pipeline` container and how they interact.

```mermaid
C4Component
    title Component Diagram — ML Pipeline Container

    Container_Boundary(ml_pipeline, "ML Pipeline Container") {

        Component(preproc, "PreprocessingPipeline", "Python class, pandas, langdetect", "Loads raw CSVs, cleans tweets (HTML entities, URLs, non-English), deduplicates, and performs time-based 60/20/20 splits per ticker. Outputs parquet files.")

        Component(vader, "VADERModel", "Python class, vaderSentiment", "Rule-based lexicon model. Computes compound score [-1,+1] per tweet, maps to Buy/Hold/Sell/No Opinion. No training required.")

        Component(finbert, "FinBERTModel", "Python class, Transformers, PyTorch", "Loads ProsusAI/finbert transformer. Runs batched inference (batch=32). Maps positive/negative/neutral logits to Buy/Sell/Hold. Applies confidence threshold (0.6) → No Opinion.")

        Component(finbert_ft, "FinBERTFineTuner", "Python class, Transformers Trainer, PyTorch", "Fine-tunes FinBERT on silver labels (VADER on train split) + oversampled gold labels (human annotations x5). Early stopping patience=2. Saves best checkpoint by eval F1.")

        Component(gpt, "GPTModel", "Python class, OpenAI SDK, Ollama", "Sends batches of 5 tweets to local Ollama (gemma3:4b) or OpenAI (GPT-4o-mini) with system prompt + 8-shot examples. Parses structured label responses. Checkpoint-resumes on failure.")

        Component(aggregator, "AggregationEngine", "Python class, pandas", "Groups results by (ticker, trading_date). Computes label distributions, sentiment_score=(buy-sell)/classified, rolling 3-day and 7-day averages, and merges OHLCV stock price data.")

        Component(drift, "DriftDetector", "Python class, NumPy, pandas", "Detects 4 drift signals: label class shift >15pp, volume spike (3-sigma), weak signal (no_opinion>35%), and sentiment-price divergence (5+ consecutive days).")

        Component(evaluator, "EvaluationEngine", "Python class, scikit-learn, SciPy", "Evaluates models on 200 human-labeled tweets (F1, accuracy, kappa) and computes Pearson r correlation between sentiment and price (lags 0-3, bootstrap CI). Selects best model by priority rules.")

        Component(orchestrator, "Pipeline Orchestrator", "run_pipeline.py, argparse, MLflow SDK", "CLI entry point. Coordinates all pipeline phases sequentially. Manages MLflow parent run with nested child runs per model. Accepts flags: --models, --finetune, --finetune_epochs, --finetune_batch_size.")

        Component(mlflow_client, "MLflow Integration", "MLflow SDK", "Logs parameters, metrics, and artifacts for each model run. Registers best model in the 'stock_sentiment_classifier' model registry. Promotes challenger to Production stage.")
    }

    ContainerDb(local_fs, "Local File Store", "Parquet, CSV, mlruns/")
    Container_Ext(mlflow_server, "MLflow Tracking Server")
    System_Ext(huggingface, "Hugging Face Hub")
    System_Ext(ollama, "Ollama")

    Rel(orchestrator, preproc, "Calls Phase 1: preprocessing")
    Rel(orchestrator, vader, "Calls Phase 2: VADER inference")
    Rel(orchestrator, finbert, "Calls Phase 2: FinBERT inference")
    Rel(orchestrator, finbert_ft, "Calls Phase 2 (optional): fine-tune then infer")
    Rel(orchestrator, gpt, "Calls Phase 2: GPT/Ollama inference")
    Rel(orchestrator, aggregator, "Calls Phase 3: aggregate per model")
    Rel(orchestrator, drift, "Calls Phase 4: drift detection per model")
    Rel(orchestrator, evaluator, "Calls Phase 5: evaluate + compare models")
    Rel(orchestrator, mlflow_client, "Wraps all phases in parent MLflow run")

    Rel(preproc, local_fs, "Reads tweets.csv, stock_prices.csv; writes parquets")
    Rel(vader, local_fs, "Reads tweets_test.parquet; writes vader_results.csv")
    Rel(finbert, local_fs, "Reads tweets_test.parquet; writes finbert_results.csv")
    Rel(finbert_ft, local_fs, "Reads train/val parquets; writes models/finbert_finetuned/best/")
    Rel(gpt, local_fs, "Reads tweets_test.parquet; writes gpt_results.csv; checkpoint CSV")
    Rel(aggregator, local_fs, "Reads result CSVs + prices_clean.parquet; writes aggregated_{model}.csv")
    Rel(drift, local_fs, "Reads aggregated CSVs; writes drift_flags_{model}.csv")
    Rel(evaluator, local_fs, "Reads result CSVs + human_labels/lebeled.csv; writes classification_report_{model}.txt")

    Rel(finbert, huggingface, "Downloads ProsusAI/finbert weights on first run")
    Rel(finbert_ft, huggingface, "Downloads base FinBERT weights for fine-tuning")
    Rel(gpt, ollama, "HTTP: sends tweet batch, receives label list")

    Rel(mlflow_client, mlflow_server, "Logs runs, metrics, artifacts, registers models", "MLflow SDK / HTTP")
```

### Design Decisions & Trade-offs (Component Level)

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **Three-model ensemble approach** | Each model has different strengths: VADER (fast/free), FinBERT (domain-specific), GPT (contextual reasoning) | EvaluationEngine picks one winner — ensemble voting not implemented |
| **Silver + Gold label fine-tuning** | Gold labels are scarce (200 tweets); VADER provides cheap silver labels for the full train split | Silver labels are noisy; VADER bias may propagate into fine-tuned FinBERT |
| **Time-based split (no random shuffle)** | Prevents temporal data leakage — a model trained on Sept data shouldn't see Aug data in test | Smaller effective training set; no cross-validation across time folds |
| **Confidence threshold → No Opinion** | Prevents FinBERT from forcing a label when uncertain | Threshold (0.6) is empirically chosen; may need calibration per dataset |
| **GPT checkpoint-resume** | LLM inference is slow (5–18 min); partial results are saved every batch | Checkpoint file must be manually cleared between fresh pipeline runs |
| **DriftDetector as separate component** | Modular: drift signals are computed post-aggregation and can trigger Airflow independently | Thresholds (15pp, 3-sigma, 35%) are fixed constants — not learned from data |
| **EvaluationEngine priority table** | Deterministic model selection avoids human bias in champion selection | Priority rules may not generalize; Pearson r significance is not gated |

---

## Level 4 — Code Diagram (ML Components Only)

Shows the key classes, methods, and data structures within the ML components.

### 4.1 Class Structure

```mermaid
classDiagram

    class PreprocessingPipeline {
        +str data_dir
        +str processed_dir
        +run() dict
        -_load_prices() DataFrame
        -_load_tweets() DataFrame
        -_clean_tweet(text: str) str
        -_is_english(text: str) bool
        -_split(df: DataFrame) tuple
    }

    class VADERModel {
        +str processed_dir
        +int batch_size
        +run(mlflow_parent_run_id: str) DataFrame
        -_classify_batch(texts: list) list~tuple~
        -_map_label(compound: float, token_count: int) tuple~str, float~
    }

    class FinBERTModel {
        +str model_name_or_path
        +int batch_size
        +float confidence_threshold
        +str model_key
        +str device
        +run(mlflow_parent_run_id: str) DataFrame
        -_classify_batch(texts: list) list~dict~
        -_load_model() void
    }

    class FinBERTFineTuner {
        +str processed_dir
        +str output_dir
        +run(epochs: int, batch_size: int) dict
        -_build_training_df() DataFrame
        -_compute_metrics(eval_pred) dict
    }

    class _TweetDataset {
        +dict encodings
        +list labels
        +__len__() int
        +__getitem__(idx: int) dict
    }

    class GPTModel {
        +int batch_size
        +float temperature
        +int max_retries
        +str ollama_model
        +str ollama_base_url
        +run(mlflow_parent_run_id: str) DataFrame
        -_call_api(tweets: list) tuple
        -_parse_response(text: str, n: int) list~str~
        -_build_user_message(tweets: list) str
    }

    class AggregationEngine {
        +str processed_dir
        +run(results_df: DataFrame, model_name: str) DataFrame
        -_compute_label_distribution(group) Series
        -_compute_sentiment_score(group) float
        -_merge_prices(agg_df: DataFrame) DataFrame
    }

    class DriftDetector {
        +run(aggregated_df: DataFrame, model_name: str) DataFrame
        -_drift_flag(df: DataFrame) Series
        -_volume_spike_flag(df: DataFrame) Series
        -_weak_signal_flag(df: DataFrame) Series
        -_divergence_flag(df: DataFrame) Series
    }

    class EvaluationEngine {
        +evaluate_on_human_labels(results_df, labels_csv, model_name) dict
        +evaluate_sentiment_price_correlation(aggregated_df, model_name, n_bootstrap) dict
        +compare_models(metrics: dict) str
        +inter_model_agreement(*result_dfs, model_names) DataFrame
        -_bootstrap_pearson(x, y, n) tuple~float, float~
    }

    FinBERTFineTuner ..> _TweetDataset : creates
    FinBERTFineTuner ..> FinBERTModel : instantiates after training
```

### 4.2 ML Pipeline Data Flow

```mermaid
flowchart TD
    subgraph Input["Input Layer"]
        A[(tweets.csv\n~53k tweets\nSept 2021–Sept 2022)]
        B[(stock_prices.csv\nOHLCV per ticker)]
        C[(human_labels/lebeled.csv\n200 gold-labeled tweets)]
    end

    subgraph Preproc["PreprocessingPipeline"]
        D[Load & clean tweets\nHTML, URLs, non-English, dedup]
        E[Time-based split\n60% train / 20% val / 20% test\nper ticker — no shuffle]
        F[(tweets_train.parquet\ntweets_val.parquet\ntweets_test.parquet\nprices_clean.parquet)]
    end

    subgraph Inference["Inference Layer — 3 Models Run in Sequence"]
        G[VADERModel\ncompound score → label\nbatch=1000, ~5s]
        H[FinBERTModel\nsoftmax argmax → label\nthreshold=0.6, batch=32]
        I[FinBERTFineTuner\nsilver labels VADER +\ngold labels x5\nepochs=3, early_stop patience=2]
        J[GPTModel\n8-shot prompt → label\nbatch=5, retry=3, checkpoint]
        K[FinBERTModel\nfine-tuned path\nbatch=32]
    end

    subgraph Results["Result CSVs"]
        L[(vader_results.csv)]
        M[(finbert_results.csv)]
        N[(finbert_finetuned_results.csv)]
        O[(gpt_results.csv)]
    end

    subgraph Agg["AggregationEngine — per model"]
        P[Group by ticker + trading_date\nCompute: buy/sell/hold/no_opinion pct\nsentiment_score = buy-sell/classified\nRolling 3-day & 7-day averages]
        Q[(aggregated_vader.csv\naggregated_finbert.csv\naggregated_gpt.csv)]
    end

    subgraph Drift["DriftDetector — per model"]
        R{drift_flag\n>15pp class shift}
        S{volume_spike_flag\nvolume > mean+3σ}
        T{weak_signal_flag\nno_opinion > 35%}
        U{divergence_flag\nsentiment↔price 5+ days}
        V[(drift_flags_{model}.csv)]
    end

    subgraph Eval["EvaluationEngine"]
        W[evaluate_on_human_labels\nF1 macro/weighted, accuracy\nprecision, recall, Cohen's Kappa]
        X[evaluate_sentiment_price_correlation\nPearson r, p-value\n95% bootstrap CI — n=1000\nlags 0-3 per ticker + combined]
        Y[compare_models\nPriority: F1>0.5 → Pearson r → no_opinion rate → agreement → cost]
        Z[inter_model_agreement\npairwise agreement rate]
    end

    subgraph MLflow["MLflow Tracking"]
        AA[parent_run: pipeline_run_timestamp]
        AB[child_run: vader]
        AC[child_run: finbert]
        AD[child_run: finbert_finetuned]
        AE[child_run: gpt4o_mini]
        AF[Model Registry: stock_sentiment_classifier\nNone → Production → Archived]
    end

    A --> D
    B --> D
    D --> E
    E --> F

    F --> G
    F --> H
    F --> I
    F --> J
    I -->|saves best checkpoint| K

    G --> L
    H --> M
    K --> N
    J --> O

    L --> P
    M --> P
    N --> P
    O --> P
    P --> Q

    Q --> R & S & T & U
    R & S & T & U --> V

    L & M & N & O --> W
    C --> W
    Q --> X
    W & X --> Y
    L & M & N & O --> Z
    Z --> Y

    Y -->|best_model| AF
    AA --> AB & AC & AD & AE
    AB -.->|logs params/metrics| G
    AC -.->|logs params/metrics| H
    AD -.->|logs params/metrics| K
    AE -.->|logs params/metrics| J
```

### 4.3 Airflow Retraining DAG

```mermaid
flowchart LR
    subgraph DAG["Airflow DAG: sentiment_retrain_pipeline\nSchedule: 0 2 ✱ ✱ ✱  daily 02:00 UTC"]
        T1["check_drift\nShortCircuitOperator\nReads drift_flags_{model}.csv\nShort-circuits if <30% of last 7 days flagged"]
        T2["retrain_finbert\nPythonOperator\nFinBERTFineTuner.run\nepochs=3, batch_size=16\nXCom: retrain_result"]
        T3["evaluate_challenger\nPythonOperator\nFinBERTModel inference on val split\nEvaluationEngine.evaluate_on_human_labels\nXCom: challenger_f1, challenger_metrics"]
        T4["compare_champion\nPythonOperator\nLoad champion F1 from MLflow Production\nXCom: challenger_wins, champion_f1"]
        T5["promote_if_better\nPythonOperator\nIf challenger_wins: register + archive champion + promote\nElse: log champion retained"]
    end

    T1 -->|drift detected| T2
    T1 -->|no drift| X([Skip downstream])
    T2 --> T3
    T3 --> T4
    T4 --> T5
```

### 4.4 Key Data Schemas

```mermaid
erDiagram
    TWEETS_PARQUET {
        int index
        datetime Date
        string Tweet
        string Stock_Name
        datetime Trading_Date
        string split
    }

    PRICES_CLEAN_PARQUET {
        datetime Date
        float Open
        float High
        float Low
        float Close
        float Adj_Close
        int Volume
        string Stock_Name
        float Daily_Return_pct
        int Intraday_Trend
    }

    RESULTS_CSV {
        int tweet_id
        datetime trading_date
        string ticker
        string label
        float confidence
        float compound_score
        float positive_score
        float negative_score
        float neutral_score
        string raw_response
    }

    AGGREGATED_CSV {
        string ticker
        datetime trading_date
        int tweet_volume
        float buy_pct
        float sell_pct
        float hold_pct
        float no_opinion_pct
        float sentiment_score
        float rolling_3day_sentiment
        float rolling_7day_sentiment
        float Close
        float Daily_Return_pct
        int Intraday_Trend
    }

    DRIFT_FLAGS_CSV {
        string ticker
        datetime trading_date
        bool drift_flag
        bool volume_spike_flag
        bool weak_signal_flag
        bool divergence_flag
    }

    HUMAN_LABELS_CSV {
        int tweet_id
        string Tweet
        string final_label
        string annotator_1_label
        string annotator_2_label
    }

    TWEETS_PARQUET ||--o{ RESULTS_CSV : "inference"
    RESULTS_CSV ||--|| AGGREGATED_CSV : "aggregation"
    AGGREGATED_CSV ||--|| DRIFT_FLAGS_CSV : "drift detection"
    PRICES_CLEAN_PARQUET ||--|| AGGREGATED_CSV : "price merge"
    RESULTS_CSV }o--|| HUMAN_LABELS_CSV : "evaluation join on tweet_id"
```

### 4.5 Model Selection Logic

```
EvaluationEngine.compare_models(metrics: dict) → str

Priority Rules (applied in order):
┌─────────────────────────────────────────────────────────────────────┐
│ 1. GATE: Weighted F1 > 0.50                                         │
│    → Models below threshold are eliminated from consideration        │
│                                                                     │
│ 2. Highest |Pearson r| (combined ticker, any lag 0-3)               │
│    → Measures alignment between sentiment signal and price movement  │
│                                                                     │
│ 3. Lowest no_opinion_rate                                           │
│    → Fewer abstentions = more actionable signal                     │
│                                                                     │
│ 4. Highest inter-model agreement rate (pairwise)                    │
│    → Agreement with other models = more reliable prediction         │
│                                                                     │
│ 5. Lowest cost (tie-breaker)                                        │
│    VADER ($0) < FinBERT ($0) < GPT (~$5-15)                        │
└─────────────────────────────────────────────────────────────────────┘

Drift Thresholds (DriftDetector):
  drift_flag:         |label_class_pct - 7day_rolling_mean| > 0.15 (15pp)
  volume_spike_flag:  tweet_volume > rolling_mean + 3 × rolling_std
  weak_signal_flag:   no_opinion_pct > 0.35
  divergence_flag:    sign(sentiment_score) ≠ sign(daily_return) for 5+ consecutive days
```

---

## Architecture Summary

### Key Design Decisions

| Layer | Decision | Justification |
|-------|----------|---------------|
| **Data** | Time-based split, no shuffle | Prevents temporal leakage — training on future data to predict past is invalid for financial signals |
| **ML** | Three independent models (VADER, FinBERT, GPT) | Complementary strengths; rule-based baseline vs. domain-specific transformer vs. general-purpose LLM |
| **ML** | Silver + Gold fine-tuning | Bootstraps fine-tuning from 200 gold labels by augmenting with VADER-labeled silver data |
| **ML** | Confidence threshold for "No Opinion" | Reduces false signals from low-confidence FinBERT predictions |
| **Serving** | Champion/Challenger via MLflow registry | Automated promotion only if challenger exceeds champion F1 — guards against regressions |
| **Orchestration** | Airflow short-circuit on low drift | Avoids unnecessary retraining cost when data distribution is stable |
| **Infrastructure** | Docker Compose + shared volumes | Self-contained local deployment; reproducible across environments |
| **Persistence** | Dual path: Supabase cloud + local CSV | Resilient; system runs offline; Supabase adds multi-user and live simulation support |

### ML Trade-offs Summary

| Trade-off | Chosen Approach | Alternative Not Taken |
|-----------|----------------|----------------------|
| Speed vs. quality | Three separate models evaluated independently | Ensemble voting (would improve accuracy but hide model transparency) |
| Data quantity vs. correctness | Silver labels from VADER for fine-tuning | Manual annotation of full dataset (cost-prohibitive for 53k tweets) |
| Drift thresholds | Fixed statistical constants | Learned thresholds from historical drift windows |
| Evaluation scope | Test split + 200 human labels | Full cross-validation (not possible with time-series data and small gold set) |
| LLM cost | Ollama local by default | OpenAI cloud API (higher quality, higher cost, latency) |
