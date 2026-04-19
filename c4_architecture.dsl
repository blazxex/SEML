workspace "Stock Sentiment Analysis System" "End-to-end ML pipeline that classifies stock-related tweets into sentiment signals, correlates them with price movements, detects data drift, and triggers automated retraining." {

    model {

        # ── People ──────────────────────────────────────────────────────────
        investor     = person "Retail Investor"    "Views sentiment insights and live portfolio simulation via the web dashboard."
        mlEngineer   = person "ML Engineer"        "Runs the pipeline CLI, tunes hyperparameters, and monitors MLflow experiments."
        dataEngineer = person "Data Engineer"      "Seeds processed data into Supabase and manages preprocessing quality."
        sysAdmin     = person "System Admin"       "Deploys Docker containers, configures environment variables, and monitors Airflow DAGs."

        # ── External Systems ─────────────────────────────────────────────────
        supabase    = softwareSystem "Supabase"          "Managed cloud PostgreSQL + Auth. Stores users, aggregated results, drift flags, and live simulation state." "External"
        huggingface = softwareSystem "Hugging Face Hub"  "Hosts pre-trained FinBERT weights (ProsusAI/finbert) downloaded at runtime." "External"
        ollama      = softwareSystem "Ollama"            "OpenAI-compatible local LLM inference server running gemma3:4b. Default free backend for tweet classification." "External"
        openaiApi   = softwareSystem "OpenAI API"        "Optional cloud LLM (GPT-4o-mini). Higher quality but incurs ~$5-15 cost per full pipeline run." "External"
        csvData     = softwareSystem "Static Data Files" "Raw input: tweets.csv (~53k tweets Sep 2021–Sep 2022), stock_prices.csv (OHLCV), human_labels/lebeled.csv (200 gold-labeled tweets)." "External"

        # ── Main System ──────────────────────────────────────────────────────
        system = softwareSystem "Stock Sentiment Analysis System" "Classifies tweet sentiment per ticker, aggregates signals, detects distribution drift, serves insights on a Bloomberg-style dashboard, and retrains models automatically when drift is detected." {

            # ── Containers ───────────────────────────────────────────────────
            dashboard = container "Flask Dashboard" "Serves the web UI: ticker/date selectors, sentiment-vs-price charts, drift alerts, model comparison table, and live portfolio simulation. Handles user authentication." "Python 3.11, Flask 3.0, Gunicorn, Jinja2, Chart.js, Bootstrap 5" "WebApp" {

                authHandler    = component "Auth Handler"         "Validates username/password against Supabase users table using bcrypt. Issues session tokens." "Flask-Login, bcrypt"
                chartController = component "Chart Controller"    "Serves JSON aggregated results and drift flags to Chart.js frontend. Reads from Supabase or falls back to local CSVs." "Flask Blueprint"
                simEngine      = component "Live Simulation Engine" "Runs day-by-day portfolio simulation based on selected model's Buy/Hold/Sell signals. Reads/writes sim_state and sim_results to Supabase." "Python, Supabase SDK"
                staticAssets   = component "Static Assets"        "Chart.js charts, Bootstrap 5 dark theme, Jinja2 templates." "HTML, CSS, JavaScript"
            }

            mlflowServer = container "MLflow Tracking Server" "Stores experiment runs, parameters, metrics, and model artifacts. Hosts the model registry with champion/challenger promotion workflow." "Python 3.11, MLflow 2.13+" "MLOps"

            airflowWeb   = container "Airflow Webserver" "Web UI for monitoring DAG runs, viewing task logs, and manually triggering the retraining pipeline." "Apache Airflow 2.9.3, Python 3.11" "WebApp"

            airflowSched = container "Airflow Scheduler" "Executes the daily retraining DAG (02:00 UTC). Checks drift flags, fine-tunes FinBERT, evaluates challenger, and promotes if it outperforms the champion." "Apache Airflow 2.9.3, Python 3.11" "Scheduler" {

                checkDrift      = component "check_drift Task"      "ShortCircuitOperator. Reads drift_flags CSV. Short-circuits if fewer than 30% of the last 7 days have any drift flag." "Airflow ShortCircuitOperator"
                retrainFinbert  = component "retrain_finbert Task"  "PythonOperator. Calls FinBERTFineTuner.run(epochs=3, batch_size=16). Pushes model path to XCom." "Airflow PythonOperator"
                evalChallenger  = component "evaluate_challenger Task" "PythonOperator. Runs FinBERTModel inference on val split. Calls EvaluationEngine.evaluate_on_human_labels(). Pushes challenger_f1 to XCom." "Airflow PythonOperator"
                compareChampion = component "compare_champion Task" "PythonOperator. Loads champion F1 from MLflow Production stage. Compares with challenger_f1 from XCom." "Airflow PythonOperator"
                promoteModel    = component "promote_if_better Task" "PythonOperator. If challenger wins: registers fine-tuned model, archives champion, promotes challenger to Production. Otherwise retains champion." "Airflow PythonOperator"
            }

            airflowDb = container "Airflow Metadata DB" "Stores Airflow DAG state, task instance history, XCom values, and scheduler heartbeats." "PostgreSQL 15-alpine" "Database"

            mlPipeline = container "ML Pipeline" "CLI-driven orchestrator that runs preprocessing, model inference (VADER/FinBERT/GPT), aggregation, drift detection, evaluation, and MLflow logging in sequence." "Python 3.11, PyTorch 2.3, Transformers 4.41, scikit-learn, pandas, MLflow SDK" "BatchProcess" {

                # ── ML Components ─────────────────────────────────────────
                pipelineOrchestrator = component "Pipeline Orchestrator"   "CLI entry point (run_pipeline.py). Coordinates all pipeline phases. Manages MLflow parent run with nested child runs per model. Flags: --models, --finetune, --finetune_epochs." "Python, argparse, MLflow SDK"

                preprocessingPipeline = component "PreprocessingPipeline" "Loads raw CSVs. Cleans tweets: strips HTML entities, URLs, removes non-English tweets, deduplicates. Performs time-based 60/20/20 split per ticker. Saves parquet files." "Python, pandas, langdetect"

                vaderModel    = component "VADERModel"         "Rule-based lexicon model. Computes compound score [-1,+1] per tweet via SentimentIntensityAnalyzer. Maps to Buy/Hold/Sell/No Opinion. No GPU required. ~5s for 2k tweets." "Python, vaderSentiment, NumPy"

                finbertModel  = component "FinBERTModel"       "Loads ProsusAI/finbert transformer. Batch inference (batch=32, max_len=512). Softmax argmax maps positive/negative/neutral to Buy/Sell/Hold. Confidence threshold 0.6 → No Opinion. Supports CUDA/MPS/CPU." "Python, Transformers, PyTorch"

                finbertFT     = component "FinBERTFineTuner"   "Fine-tunes FinBERT on silver labels (VADER applied to train split ~28k tweets) plus gold labels (200 human annotations oversampled 5x). Early stopping patience=2 on eval F1. Saves best checkpoint." "Python, Transformers Trainer, PyTorch, scikit-learn"

                gptModel      = component "GPTModel"           "Sends batches of 5 tweets to Ollama (default) or OpenAI with system prompt and 8-shot examples. Parses structured label response. Retries up to 3x with exponential backoff. Checkpoint-resumes on failure." "Python, OpenAI SDK, Ollama"

                aggregationEngine = component "AggregationEngine" "Groups inference results by (ticker, trading_date). Computes label distributions, sentiment_score=(buy-sell)/classified, 3-day and 7-day rolling averages. Merges OHLCV stock price data." "Python, pandas"

                driftDetector = component "DriftDetector"      "Detects 4 drift signals per (ticker, date): label class shift >15pp vs 7-day baseline, volume spike (3-sigma rule), weak signal (no_opinion>35%), and sentiment-price divergence (5+ consecutive days)." "Python, pandas, NumPy"

                evaluationEngine = component "EvaluationEngine" "Evaluates models on 200 human-labeled tweets (F1 macro/weighted, accuracy, precision, recall, Cohen's Kappa). Computes Pearson r between sentiment and price at lags 0-3 with 95% bootstrap CI (n=1000). Selects best model by priority rules." "Python, scikit-learn, SciPy"

                mlflowClient  = component "MLflow Integration"  "Logs parameters, metrics, and artifacts for each model in nested child runs. Registers best model in stock_sentiment_classifier registry. Promotes champion to Production stage." "Python, MLflow SDK"
            }

            localFs = container "Local File Store" "Stores all pipeline I/O: raw CSVs, processed parquets, inference result CSVs, aggregated CSVs, drift flag CSVs, MLflow artifact store (./mlruns/)." "Parquet, CSV, File System" "Database"
        }

        # ── Context-Level Relationships ──────────────────────────────────────
        investor     -> system "Views sentiment dashboard and runs live simulation" "HTTPS"
        mlEngineer   -> system "Runs pipeline CLI and monitors MLflow experiments" "CLI / Browser"
        dataEngineer -> system "Seeds data into Supabase and monitors preprocessing" "CLI / Browser"
        sysAdmin     -> system "Deploys via Docker Compose and manages Airflow" "Terminal"

        system -> supabase    "Reads/writes users, results, drift flags, simulation state" "Supabase SDK / REST"
        system -> huggingface "Downloads ProsusAI/finbert model weights at first run" "HTTPS"
        system -> ollama      "Sends tweet batches for LLM classification" "HTTP / OpenAI-compatible API"
        system -> openaiApi   "Optional: sends tweet batches for GPT-4o-mini classification" "HTTPS / OpenAI SDK"
        system -> csvData     "Reads raw tweets, prices, and gold labels as pipeline input" "File I/O"

        # ── Container-Level Relationships ────────────────────────────────────
        investor     -> dashboard    "Views dashboard, runs simulation" "HTTPS / Browser"
        mlEngineer   -> mlPipeline   "Runs pipeline CLI with flags" "Terminal"
        mlEngineer   -> mlflowServer "Views experiments and model registry" "Browser / HTTP :5000"
        sysAdmin     -> airflowWeb   "Monitors and triggers DAGs" "Browser / HTTP :8082"

        dashboard    -> supabase     "Authenticates users; reads aggregated results and drift flags; reads/writes simulation state" "Supabase SDK"
        dashboard    -> localFs      "Reads aggregated CSVs at startup (fallback when Supabase is unavailable)" "File I/O"
        dashboard    -> mlflowServer "Reads model metadata for comparison view" "HTTP :5000"

        mlPipeline   -> localFs      "Reads raw CSVs and parquets; writes result CSVs, aggregated CSVs, drift flag CSVs" "File I/O"
        mlPipeline   -> mlflowServer "Logs runs, metrics, artifacts; registers models" "MLflow SDK / HTTP :5000"
        mlPipeline   -> huggingface  "Downloads FinBERT base and fine-tuned weights" "HTTPS"
        mlPipeline   -> ollama       "Sends tweet batches (batch=5) for LLM classification" "HTTP"
        mlPipeline   -> openaiApi    "Optional GPT-4o-mini classification" "HTTPS"

        airflowSched -> localFs      "Reads drift flag CSVs; writes retrained model to models/finbert_finetuned/best/" "File I/O"
        airflowSched -> mlflowServer "Loads champion model F1; registers and promotes challenger" "MLflow SDK"
        airflowSched -> huggingface  "Downloads FinBERT base weights for fine-tuning" "HTTPS"
        airflowSched -> airflowDb    "Reads/writes DAG state and XCom values" "PostgreSQL"
        airflowWeb   -> airflowDb    "Reads DAG and task run state" "PostgreSQL"

        # ── Component-Level Relationships (ML Pipeline) ──────────────────────
        pipelineOrchestrator -> preprocessingPipeline "Phase 1: preprocess raw data"
        pipelineOrchestrator -> vaderModel            "Phase 2: VADER inference"
        pipelineOrchestrator -> finbertModel          "Phase 2: FinBERT inference"
        pipelineOrchestrator -> finbertFT             "Phase 2 (optional --finetune): fine-tune FinBERT"
        pipelineOrchestrator -> gptModel              "Phase 2: GPT/Ollama inference"
        pipelineOrchestrator -> aggregationEngine     "Phase 3: aggregate per model"
        pipelineOrchestrator -> driftDetector         "Phase 4: drift detection per model"
        pipelineOrchestrator -> evaluationEngine      "Phase 5: evaluate and compare models"
        pipelineOrchestrator -> mlflowClient          "Wraps all phases in MLflow parent run"

        finbertFT    -> finbertModel  "Instantiates for inference after fine-tuning"

        preprocessingPipeline -> localFs "Reads tweets.csv, stock_prices.csv; writes parquet splits"
        vaderModel            -> localFs "Reads tweets_test.parquet; writes vader_results.csv"
        finbertModel          -> localFs "Reads tweets_test.parquet; writes finbert_results.csv"
        finbertFT             -> localFs "Reads train/val parquets; writes models/finbert_finetuned/best/"
        gptModel              -> localFs "Reads tweets_test.parquet; writes gpt_results.csv; checkpoint CSV"
        aggregationEngine     -> localFs "Reads result CSVs + prices_clean.parquet; writes aggregated_{model}.csv"
        driftDetector         -> localFs "Reads aggregated CSVs; writes drift_flags_{model}.csv"
        evaluationEngine      -> localFs "Reads result CSVs + human_labels/lebeled.csv; writes classification reports"
        mlflowClient          -> mlflowServer "Logs runs, metrics, artifacts; registers models" "MLflow SDK"

        finbertModel -> huggingface "Downloads ProsusAI/finbert weights on first run" "HTTPS"
        finbertFT    -> huggingface "Downloads base FinBERT weights for fine-tuning" "HTTPS"
        gptModel     -> ollama      "HTTP: sends tweet batch (batch=5), receives label list" "HTTP"
        gptModel     -> openaiApi   "Optional: sends tweet batch to GPT-4o-mini" "HTTPS"

        # ── Component-Level Relationships (Dashboard) ────────────────────────
        authHandler     -> supabase "Validates credentials against users table" "Supabase SDK"
        chartController -> supabase "Reads aggregated_results and drift_flags tables" "Supabase SDK"
        chartController -> localFs  "Fallback: reads aggregated CSVs" "File I/O"
        simEngine       -> supabase "Reads/writes sim_state and sim_results tables" "Supabase SDK"

        # ── Component-Level Relationships (Airflow Scheduler) ────────────────
        checkDrift      -> localFs      "Reads drift_flags_{model}.csv for last 7 days"
        retrainFinbert  -> localFs      "Calls FinBERTFineTuner; writes models/finbert_finetuned/best/"
        retrainFinbert  -> huggingface  "Downloads FinBERT base weights" "HTTPS"
        evalChallenger  -> localFs      "Reads val parquet and human labels for evaluation"
        evalChallenger  -> mlflowServer "Logs challenger evaluation run" "MLflow SDK"
        compareChampion -> mlflowServer "Reads champion model F1 from Production stage" "MLflow SDK"
        promoteModel    -> mlflowServer "Registers challenger; archives champion; promotes to Production" "MLflow SDK"
        promoteModel    -> localFs      "Reads fine-tuned model from models/finbert_finetuned/best/"
    }

    # ── Views ────────────────────────────────────────────────────────────────
    views {

        # Level 1: System Context
        systemContext system "SystemContext" "C4 Level 1 — System Context: who uses the system and what external systems it depends on." {
            include *
            autoLayout lr
        }

        # Level 2: Container
        container system "Containers" "C4 Level 2 — Container: major deployable units, their technologies, and communication." {
            include *
            autoLayout lr
        }

        # Level 3: Component — ML Pipeline
        component mlPipeline "MLPipelineComponents" "C4 Level 3 — Component (ML Pipeline): internal structure of the ML batch processing container." {
            include *
            autoLayout tb
        }

        # Level 3: Component — Dashboard
        component dashboard "DashboardComponents" "C4 Level 3 — Component (Dashboard): internal structure of the Flask web container." {
            include *
            autoLayout tb
        }

        # Level 3: Component — Airflow Scheduler
        component airflowSched "AirflowComponents" "C4 Level 3 — Component (Airflow Scheduler): retraining DAG task structure." {
            include *
            autoLayout lr
        }

        # Styles
        styles {
            element "Person" {
                shape person
                background #08427B
                color #ffffff
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "WebApp" {
                shape WebBrowser
                background #1168BD
                color #ffffff
            }
            element "Database" {
                shape cylinder
                background #438DD5
                color #ffffff
            }
            element "Scheduler" {
                background #6B3A6B
                color #ffffff
            }
            element "BatchProcess" {
                background #2D6A4F
                color #ffffff
            }
            element "MLOps" {
                background #E76F51
                color #ffffff
            }
            element "softwareSystem" {
                background #1168BD
                color #ffffff
            }
            element "container" {
                background #438DD5
                color #ffffff
            }
            element "component" {
                background #85BBF0
                color #000000
            }
            relationship "Relationship" {
                dashed false
            }
        }
    }
}
