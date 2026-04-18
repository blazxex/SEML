"""
Master orchestration script.

Usage:
    python run_pipeline.py --models vader finbert gpt --ticker all
    python run_pipeline.py --models vader --ticker AAPL
    python run_pipeline.py --models vader finbert gpt --skip_gpt False
"""
import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import PreprocessingPipeline
from src.aggregation import AggregationEngine
from src.evaluation import EvaluationEngine
from src.drift_detection import DriftDetector


def parse_args():
    p = argparse.ArgumentParser(description="Stock sentiment pipeline")
    p.add_argument(
        "--models", nargs="+", choices=["vader", "finbert", "gpt"], default=["vader"],
        help="Models to run (default: vader)"
    )
    p.add_argument(
        "--ticker", default="all",
        help="Ticker filter — 'all' or one of AAPL/TSLA/TSM"
    )
    p.add_argument(
        "--skip_preprocessing", action="store_true",
        help="Skip preprocessing if parquets already exist"
    )
    p.add_argument(
        "--skip_gpt", type=lambda x: x.lower() != "false", default=True,
        help="Skip GPT model to avoid API cost. Pass False to enable."
    )
    p.add_argument(
        "--human_labels", default="human_labels/human_labels.csv",
        help="Path to human-labeled CSV for evaluation"
    )
    return p.parse_args()


def main():
    args = parse_args()

    if "gpt" in args.models and args.skip_gpt:
        log.warning(
            "GPT model requested but --skip_gpt is True (default). "
            "Pass --skip_gpt False to run GPT and incur API costs (~$5-15)."
        )
        args.models = [m for m in args.models if m != "gpt"]

    # ------------------------------------------------------------------
    # Phase 1: Preprocessing
    processed_dir = Path("data/processed")
    if not args.skip_preprocessing or not (processed_dir / "tweets_test.parquet").exists():
        log.info("=== PHASE 1: Preprocessing ===")
        pipeline = PreprocessingPipeline(data_dir="cleaned_data", processed_dir=str(processed_dir))
        summary = pipeline.run()
        log.info("Split summary: %s", summary)
    else:
        log.info("Skipping preprocessing (parquets exist).")

    # ------------------------------------------------------------------
    # MLflow experiment
    mlflow.set_experiment("stock_sentiment_prediction")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=f"pipeline_run_{timestamp}") as parent_run:
        parent_run_id = parent_run.info.run_id
        mlflow.log_params({
            "models": ",".join(args.models),
            "ticker_filter": args.ticker,
            "random_seed": SEED,
        })

        results_map = {}
        aggregated_map = {}

        aggregator = AggregationEngine(processed_dir=str(processed_dir))
        drifter = DriftDetector()

        # ------------------------------------------------------------------
        # Phase 2: Model inference
        for model_name in args.models:
            log.info("=== Running model: %s ===", model_name.upper())

            if model_name == "vader":
                from src.models.vader_model import VADERModel
                results = VADERModel(processed_dir=str(processed_dir)).run(parent_run_id)

            elif model_name == "finbert":
                from src.models.finbert_model import FinBERTModel
                results = FinBERTModel(processed_dir=str(processed_dir)).run(parent_run_id)

            elif model_name == "gpt":
                from src.models.gpt_model import GPTModel
                results = GPTModel(processed_dir=str(processed_dir)).run(parent_run_id)

            results_map[model_name] = results

            # Phase 3: Aggregate
            log.info("=== Aggregating: %s ===", model_name)
            agg = aggregator.run(results, model_name)
            aggregated_map[model_name] = agg
            mlflow.log_artifact(f"results/aggregated_{model_name}.csv")

            # Phase 4: Drift detection
            log.info("=== Drift detection: %s ===", model_name)
            drifter.run(agg, model_name)
            mlflow.log_artifact(f"results/drift_flags_{model_name}.csv")

        # ------------------------------------------------------------------
        # Phase 5: Evaluation
        evaluator = EvaluationEngine()
        all_metrics = {}

        for model_name, results in results_map.items():
            log.info("=== Evaluating correlation: %s ===", model_name)
            agg = aggregated_map[model_name]
            corr = evaluator.evaluate_sentiment_price_correlation(agg, model_name)
            metrics = {}
            for key, val in corr.items():
                if isinstance(val, dict):
                    metrics[f"pearson_r_{key}"] = val["pearson_r"]
                    metrics[f"pearson_p_{key}"] = val["p_value"]

            # Human labels (only if file has content)
            hl_path = Path(args.human_labels)
            import pandas as _pd
            if hl_path.exists() and _pd.read_csv(hl_path).dropna().shape[0] > 0:
                hl_metrics = evaluator.evaluate_on_human_labels(results, str(hl_path), model_name)
                metrics.update(hl_metrics)
                mlflow.log_artifact(f"results/classification_report_{model_name}.txt")

            all_metrics[model_name] = metrics

        # Inter-model agreement
        if len(results_map) > 1:
            names = list(results_map.keys())
            dfs = [results_map[n] for n in names]
            agreement = evaluator.inter_model_agreement(*dfs, model_names=names)
            log.info("Inter-model agreement:\n%s", agreement.to_string(index=False))
            agreement_path = Path("results/inter_model_agreement.csv")
            agreement.to_csv(agreement_path, index=False)
            mlflow.log_artifact(str(agreement_path))

        # ------------------------------------------------------------------
        # Phase 6: Model selection
        if all_metrics:
            # Merge corr metrics into all_metrics for decision
            best_model = evaluator.compare_models(all_metrics)
            log.info("=== WINNER: %s ===", best_model.upper())
            mlflow.log_param("selected_model", best_model)

            # Register in MLflow Model Registry
            # Note: a logged model artifact is needed; VADER has no sklearn model to register,
            # so we register the run itself as a reference artifact.
            try:
                best_run_id = None
                client = mlflow.tracking.MlflowClient()
                runs = client.search_runs(
                    experiment_ids=[parent_run.info.experiment_id],
                    filter_string=f"tags.mlflow.runName = '{best_model if best_model != 'gpt' else 'gpt4o_mini'}'",
                )
                if runs:
                    best_run_id = runs[0].info.run_id
                    mlflow.register_model(
                        model_uri=f"runs:/{best_run_id}/model",
                        name="stock_sentiment_classifier",
                    )
                    client.set_registered_model_tag(
                        "stock_sentiment_classifier", "stage", "Production"
                    )
                    log.info("Model registered as stock_sentiment_classifier (Production).")
            except Exception as e:
                log.warning("Model registry step skipped: %s", e)

        best_label = best_model.upper() if all_metrics else "N/A"
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print(f"Selected model: {best_label}")
        print("View results: mlflow ui --port 5000")
        print("Run dashboard: python dashboard/app.py")
        print("=" * 60)


if __name__ == "__main__":
    main()
