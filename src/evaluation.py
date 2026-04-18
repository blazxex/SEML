import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

log = logging.getLogger(__name__)
RESULTS_DIR = Path("results")
SEED = 42


class EvaluationEngine:

    # ------------------------------------------------------------------
    def evaluate_on_human_labels(
        self, results_df: pd.DataFrame, human_labels_csv: str, model_name: str
    ) -> dict:
        RESULTS_DIR.mkdir(exist_ok=True)
        labels_df = pd.read_csv(human_labels_csv)
        labels_df = labels_df.dropna(subset=["final_label"])

        keep_cols = ["tweet_id", "final_label"]
        for col in ["annotator_1_label", "annotator_2_label"]:
            if col in labels_df.columns:
                keep_cols.append(col)
        merged = results_df.merge(labels_df[keep_cols], on="tweet_id", how="inner")
        if len(merged) == 0:
            log.warning("No matching tweet_ids between results and human labels.")
            return {}

        y_true = merged["final_label"]
        y_pred = merged["label"]

        acc = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        kappa = None
        if "annotator_1_label" in merged.columns and "annotator_2_label" in merged.columns:
            ann1 = merged["annotator_1_label"].dropna()
            ann2 = merged["annotator_2_label"].dropna()
            if len(ann1) == len(ann2) and len(ann1) > 0:
                kappa = cohen_kappa_score(ann1, ann2)

        report = classification_report(y_true, y_pred, zero_division=0)
        report_path = RESULTS_DIR / f"classification_report_{model_name}.txt"
        report_path.write_text(report)
        log.info("Classification report saved to %s", report_path)

        metrics = {
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
            "precision_weighted": round(prec_w, 4),
            "recall_weighted": round(rec_w, 4),
        }
        if kappa is not None:
            metrics["cohens_kappa"] = round(kappa, 4)

        log.info("%s human-label metrics: %s", model_name, metrics)
        return metrics

    # ------------------------------------------------------------------
    def evaluate_sentiment_price_correlation(
        self, aggregated_df: pd.DataFrame, model_name: str, n_bootstrap: int = 1000
    ) -> dict:
        df = aggregated_df.copy()
        df["trading_date"] = pd.to_datetime(df["trading_date"])
        df = df.sort_values(["ticker", "trading_date"])
        df["next_day_return"] = df.groupby("ticker")["Daily Return %"].shift(-1)

        results = {}
        tickers = list(df["ticker"].unique()) + ["combined"]

        for lag in range(4):
            for ticker in tickers:
                if ticker == "combined":
                    sub = df.dropna(subset=["sentiment_score", "Daily Return %"])
                else:
                    sub = df[df["ticker"] == ticker].dropna(
                        subset=["sentiment_score", "Daily Return %"]
                    )

                if lag > 0:
                    shifted = sub.copy()
                    if ticker == "combined":
                        # shift within each ticker to avoid cross-ticker leakage
                        shifted["sentiment_score"] = (
                            shifted.groupby("ticker")["sentiment_score"].shift(lag)
                        )
                    else:
                        shifted["sentiment_score"] = shifted["sentiment_score"].shift(lag)
                    sub = shifted.dropna(subset=["sentiment_score"])

                x = sub["sentiment_score"].values
                y = sub["Daily Return %"].values

                if len(x) < 5:
                    continue

                r, p = stats.pearsonr(x, y)

                # Bootstrap CI
                rng = np.random.default_rng(SEED)
                boot_rs = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(x), size=len(x))
                    bx, by = x[idx], y[idx]
                    if np.std(bx) > 0 and np.std(by) > 0:
                        br, _ = stats.pearsonr(bx, by)
                        boot_rs.append(br)

                ci_lo = float(np.percentile(boot_rs, 2.5)) if boot_rs else float("nan")
                ci_hi = float(np.percentile(boot_rs, 97.5)) if boot_rs else float("nan")

                key = f"{ticker}_lag{lag}"
                results[key] = {
                    "pearson_r": round(r, 4),
                    "p_value": round(p, 6),
                    "ci_95_lo": round(ci_lo, 4),
                    "ci_95_hi": round(ci_hi, 4),
                    "n": len(x),
                }

        log.info("%s correlation results: %s", model_name, results)
        return results

    # ------------------------------------------------------------------
    def compare_models(
        self,
        metrics: dict,  # {model_name: {metric_key: value, ...}}
    ) -> str:
        """Apply decision table and return name of best model."""
        candidates = list(metrics.keys())

        # Priority 1: weighted F1 > 0.50
        valid = [m for m in candidates if metrics[m].get("f1_weighted", 0) > 0.50]
        if not valid:
            valid = candidates  # fall back to all if none pass threshold

        # Priority 2: best absolute Pearson r (combined, any lag)
        def best_r(m):
            vals = [abs(v["pearson_r"]) for k, v in metrics[m].items() if "combined_lag" in k and isinstance(v, dict)]
            return max(vals) if vals else 0.0

        # Priority 3: lowest no_opinion_rate
        # Priority 4: not implemented here (inter-model agreement done separately)
        # Priority 5: cost — prefer VADER > FinBERT > GPT
        cost_order = {"vader": 0, "finbert": 1, "gpt": 2}

        def score(m):
            return (
                metrics[m].get("f1_weighted", 0),
                best_r(m),
                -metrics[m].get("no_opinion_rate", 1.0),
                -cost_order.get(m, 99),
            )

        best = max(valid, key=score)
        log.info("Selected model: %s", best)
        return best

    # ------------------------------------------------------------------
    def inter_model_agreement(self, *result_dfs: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
        merged = result_dfs[0][["tweet_id", "label"]].rename(columns={"label": model_names[0]})
        for df, name in zip(result_dfs[1:], model_names[1:]):
            merged = merged.merge(df[["tweet_id", "label"]].rename(columns={"label": name}), on="tweet_id", how="inner")

        pairs = []
        names = model_names
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                agree = (merged[a] == merged[b]).mean()
                pairs.append({"pair": f"{a}↔{b}", "agreement_rate": round(agree, 4), "n": len(merged)})

        return pd.DataFrame(pairs)
