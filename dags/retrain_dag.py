"""
Airflow DAG: Sentiment model retrain & champion/challenger pipeline.

Schedule: daily at 02:00 UTC
Flow:
  check_drift
      │  (short-circuit: skip if no drift)
      ▼
  retrain_finbert          ← fine-tune on train split + human labels
      ▼
  evaluate_challenger      ← F1 on human labels test set
      ▼
  compare_champion         ← load current Production model metrics from MLflow
      ▼
  promote_if_better        ← transition to Production if challenger wins

Set AIRFLOW_HOME to point to this project or configure via airflow.cfg.
Run once to init DB:  airflow db init
Then:                 airflow dags trigger sentiment_retrain_pipeline
"""
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make project importable from DAG context
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

log = logging.getLogger(__name__)

DRIFT_WINDOW    = 7    # days of recent drift flags to inspect
DRIFT_THRESHOLD = 0.3  # fraction of recent dates with a flag to trigger retrain
HUMAN_LABELS    = str(PROJECT_ROOT / "human_labels" / "lebeled.csv")
RESULTS_DIR     = PROJECT_ROOT / "results"
REGISTERED_NAME = "stock_sentiment_classifier"


# ------------------------------------------------------------------
# Task functions
# ------------------------------------------------------------------

def _check_drift(**context) -> bool:
    """Return True (continue) if drift is detected across any model."""
    import pandas as pd

    flagged = 0
    total   = 0
    for model in ("vader", "finbert", "gpt"):
        path = RESULTS_DIR / f"drift_flags_{model}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["trading_date"])
        cutoff = df["trading_date"].max() - pd.Timedelta(days=DRIFT_WINDOW)
        recent = df[df["trading_date"] >= cutoff]
        flag_cols = ["drift_flag", "volume_spike_flag", "weak_signal_flag", "divergence_flag"]
        present = [c for c in flag_cols if c in recent.columns]
        if not present:
            continue
        flagged += int(recent[present].any(axis=1).sum())
        total   += len(recent)

    if total == 0:
        log.warning("No drift files found — skipping retrain.")
        return False

    ratio = flagged / total
    log.info("Drift check: %d/%d dates flagged (%.1f%%)", flagged, total, ratio * 100)
    triggered = ratio >= DRIFT_THRESHOLD
    if triggered:
        log.info("Drift threshold reached — triggering retrain.")
    return triggered


def _retrain_finbert(**context):
    """Fine-tune FinBERT on silver + gold labels; push metrics to XCom."""
    import mlflow
    os.chdir(PROJECT_ROOT)
    from src.models.finbert_finetune import FinBERTFineTuner

    mlflow.set_experiment("stock_sentiment_prediction")
    with mlflow.start_run(run_name=f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"):
        result = FinBERTFineTuner().run(epochs=3, batch_size=16)

    context["ti"].xcom_push(key="retrain_result", value=result)
    log.info("Retrain complete: %s", result)


def _evaluate_challenger(**context):
    """Run EvaluationEngine on the fine-tuned model; push metrics to XCom."""
    import pandas as pd
    import mlflow
    os.chdir(PROJECT_ROOT)

    from src.models.finbert_model import FinBERTModel
    from src.evaluation import EvaluationEngine

    retrain_result = context["ti"].xcom_pull(key="retrain_result", task_ids="retrain_finbert")
    model_path = retrain_result.get("model_path") if retrain_result else None

    # Run inference on test split using fine-tuned weights
    finbert = FinBERTModel(model_name_or_path=model_path)
    results_df = finbert.run()

    # Evaluate against human labels
    engine  = EvaluationEngine()
    metrics = engine.evaluate_on_human_labels(results_df, HUMAN_LABELS, "finbert_finetuned")

    challenger_f1 = metrics.get("macro_f1", metrics.get("f1", 0.0))
    context["ti"].xcom_push(key="challenger_f1", value=challenger_f1)
    context["ti"].xcom_push(key="challenger_metrics", value=metrics)
    log.info("Challenger F1: %.4f", challenger_f1)


def _compare_champion(**context):
    """
    Load champion metrics from MLflow Model Registry.
    Push champion_f1 and whether challenger wins to XCom.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    os.chdir(PROJECT_ROOT)
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))

    challenger_f1 = context["ti"].xcom_pull(key="challenger_f1", task_ids="evaluate_challenger") or 0.0

    client = MlflowClient()
    champion_f1 = 0.0
    try:
        versions = client.get_latest_versions(REGISTERED_NAME, stages=["Production"])
        if versions:
            run_id = versions[0].run_id
            run    = client.get_run(run_id)
            # Try common metric key names
            for key in ("macro_f1", "f1", "val_f1", "test_f1"):
                val = run.data.metrics.get(key)
                if val is not None:
                    champion_f1 = val
                    break
            log.info("Current champion F1 (%s): %.4f", versions[0].version, champion_f1)
        else:
            log.info("No Production model registered yet — challenger wins by default.")
    except Exception as e:
        log.warning("Could not load champion metrics: %s", e)

    challenger_wins = challenger_f1 > champion_f1
    context["ti"].xcom_push(key="challenger_wins", value=challenger_wins)
    context["ti"].xcom_push(key="champion_f1",     value=champion_f1)
    context["ti"].xcom_push(key="challenger_f1",   value=challenger_f1)

    log.info(
        "Champion F1=%.4f  Challenger F1=%.4f  → challenger_wins=%s",
        champion_f1, challenger_f1, challenger_wins,
    )


def _promote_if_better(**context):
    """Promote challenger to Production in MLflow if it beats the champion."""
    import mlflow
    from mlflow.tracking import MlflowClient
    os.chdir(PROJECT_ROOT)
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))

    challenger_wins = context["ti"].xcom_pull(key="challenger_wins", task_ids="compare_champion")
    champion_f1     = context["ti"].xcom_pull(key="champion_f1",     task_ids="compare_champion") or 0.0
    challenger_f1   = context["ti"].xcom_pull(key="challenger_f1",   task_ids="compare_champion") or 0.0

    if not challenger_wins:
        log.info(
            "Champion retained (F1 %.4f ≥ challenger F1 %.4f).",
            champion_f1, challenger_f1,
        )
        return

    client = MlflowClient()
    try:
        # Register the fine-tuned model from its saved path
        model_path = PROJECT_ROOT / "models" / "finbert_finetuned" / "best"
        model_uri  = f"file://{model_path}"

        # Find the latest version of the registered model
        versions = client.get_latest_versions(REGISTERED_NAME, stages=["None", "Staging"])
        if not versions:
            # Register fresh
            mlflow.register_model(model_uri, REGISTERED_NAME)
            versions = client.get_latest_versions(REGISTERED_NAME, stages=["None"])

        new_version = max(versions, key=lambda v: int(v.version))

        # Archive existing Production
        for v in client.get_latest_versions(REGISTERED_NAME, stages=["Production"]):
            client.transition_model_version_stage(
                name=REGISTERED_NAME, version=v.version, stage="Archived"
            )

        # Promote challenger
        client.transition_model_version_stage(
            name=REGISTERED_NAME,
            version=new_version.version,
            stage="Production",
        )
        log.info(
            "Promoted finbert_finetuned v%s to Production (F1 %.4f → %.4f).",
            new_version.version, champion_f1, challenger_f1,
        )
    except Exception as e:
        log.error("Promotion failed: %s", e)
        raise


# ------------------------------------------------------------------
# DAG definition
# ------------------------------------------------------------------

default_args = {
    "owner":            "seml",
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id          = "sentiment_retrain_pipeline",
    description     = "Retrain FinBERT on drift, evaluate, promote if champion",
    schedule        = "0 2 * * *",   # daily 02:00 UTC
    start_date      = datetime(2024, 1, 1),
    catchup         = False,
    default_args    = default_args,
    tags            = ["seml", "nlp", "retrain"],
) as dag:

    check_drift = ShortCircuitOperator(
        task_id         = "check_drift",
        python_callable = _check_drift,
        doc             = "Skip downstream tasks if no drift is detected.",
    )

    retrain_finbert = PythonOperator(
        task_id         = "retrain_finbert",
        python_callable = _retrain_finbert,
        execution_timeout = timedelta(hours=4),
    )

    evaluate_challenger = PythonOperator(
        task_id         = "evaluate_challenger",
        python_callable = _evaluate_challenger,
        execution_timeout = timedelta(hours=1),
    )

    compare_champion = PythonOperator(
        task_id         = "compare_champion",
        python_callable = _compare_champion,
    )

    promote_if_better = PythonOperator(
        task_id         = "promote_if_better",
        python_callable = _promote_if_better,
    )

    check_drift >> retrain_finbert >> evaluate_challenger >> compare_champion >> promote_if_better
