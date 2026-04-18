import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
RESULTS_DIR = Path("results")


class DriftDetector:

    def run(self, aggregated_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        RESULTS_DIR.mkdir(exist_ok=True)
        df = aggregated_df.copy().sort_values(["ticker", "trading_date"])

        df["drift_flag"] = False
        df["volume_spike_flag"] = False
        df["weak_signal_flag"] = False
        df["divergence_flag"] = False

        for ticker, grp in df.groupby("ticker"):
            idx = grp.index

            # drift_flag: any label class shifts >15pp vs 7-day rolling baseline
            for col in ["buy_pct", "sell_pct", "hold_pct", "no_opinion_pct"]:
                rolling_mean = grp[col].rolling(7, min_periods=1).mean()
                shift = (grp[col] - rolling_mean).abs()
                df.loc[idx[shift > 0.15], "drift_flag"] = True

            # volume_spike_flag: volume > mean + 3*std of 7-day window
            roll_mean = grp["tweet_volume"].rolling(7, min_periods=1).mean()
            roll_std = grp["tweet_volume"].rolling(7, min_periods=1).std().fillna(0)
            spike = grp["tweet_volume"] > (roll_mean + 3 * roll_std)
            df.loc[idx[spike], "volume_spike_flag"] = True

            # weak_signal_flag: no_opinion_pct > 35%
            weak = grp["no_opinion_pct"] > 0.35
            df.loc[idx[weak], "weak_signal_flag"] = True

            # divergence_flag: sentiment and price move opposite for 5+ consecutive days
            sent_dir = np.sign(grp["sentiment_score"].diff().fillna(0))
            price_dir = np.sign(grp["Daily Return %"].diff().fillna(0))
            diverge = (sent_dir != 0) & (price_dir != 0) & (sent_dir != price_dir)

            consecutive = 0
            diverg_flags = [False] * len(grp)
            for i, d in enumerate(diverge):
                consecutive = consecutive + 1 if d else 0
                if consecutive >= 5:
                    diverg_flags[i] = True
            df.loc[idx, "divergence_flag"] = diverg_flags

        out_path = RESULTS_DIR / f"drift_flags_{model_name}.csv"
        df.to_csv(out_path, index=False)
        log.info(
            "Drift flags saved to %s — drift=%d volume=%d weak=%d diverge=%d",
            out_path,
            df["drift_flag"].sum(),
            df["volume_spike_flag"].sum(),
            df["weak_signal_flag"].sum(),
            df["divergence_flag"].sum(),
        )
        return df
