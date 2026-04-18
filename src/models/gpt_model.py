import os
import random
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import mlflow
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

load_dotenv()
log = logging.getLogger(__name__)
RESULTS_DIR = Path("results")

# Default Ollama settings — override via .env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

SYSTEM_PROMPT = (
    "You are a financial sentiment classifier. Classify tweets about stocks into "
    "exactly one of: Buy, Hold, Sell, No Opinion.\n"
    "Buy = bullish/positive outlook. Sell = bearish/negative outlook.\n"
    "Hold = neutral/wait-and-see. No Opinion = irrelevant or unclassifiable.\n"
    "Respond with ONLY the label, nothing else. /no_think"
)

FEW_SHOT_EXAMPLES = (
    '"$AAPL just crushed earnings, this stock is going to the moon! 🚀" → Buy\n'
    '"Apple\'s new iPhone demand is insane, definitely adding more shares" → Buy\n'
    '"Tesla recalls again, I\'m dumping all my shares TSLA" → Sell\n'
    '"Really bad quarter for $AAPL, revenue miss and lowered guidance" → Sell\n'
    '"Not sure about TSLA right now, mixed signals from the market" → Hold\n'
    '"$TSM is flat today, waiting for the next earnings before deciding" → Hold\n'
    '"Good morning everyone, happy Monday!" → No Opinion\n'
    '"Just had the best coffee of my life" → No Opinion'
)

VALID_LABELS = {"Buy", "Hold", "Sell", "No Opinion"}


class GPTModel:
    def __init__(
        self,
        processed_dir: str = "data/processed",
        batch_size: int = 5,  # smaller batches = shorter generation time per call
        temperature: float = 0.0,
        max_retries: int = 3,
        ollama_model: str = OLLAMA_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
    ):
        self.processed_dir = Path(processed_dir)
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_retries = max_retries
        self.model_string = ollama_model

        log.info("Using Ollama model '%s' at %s", ollama_model, ollama_base_url)
        self.client = OpenAI(
            base_url=ollama_base_url,
            api_key="ollama",
            timeout=120,  # 2 min max per batch — avoids indefinite hangs
        )

    def _build_user_message(self, tweets: list[str]) -> str:
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tweets))
        return (
            f"Examples:\n{FEW_SHOT_EXAMPLES}\n\n"
            f"Now classify each tweet below. Respond with a numbered list of labels only "
            f"(e.g. '1. Buy').\n\n{numbered}"
        )

    def _parse_response(self, text: str, n: int) -> list[str]:
        import re
        # Strip <think>...</think> blocks (qwen3/deepseek thinking tokens)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        labels = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            matched = False
            # Match "No Opinion" before "No" to avoid partial match
            for label in ["No Opinion", "Buy", "Sell", "Hold"]:
                if label.lower() in line.lower():
                    labels.append(label)
                    matched = True
                    break
            if not matched:
                labels.append("No Opinion")

        labels = (labels + ["No Opinion"] * n)[:n]
        return labels

    def _call_api(self, tweets: list[str]) -> tuple[list[str], str, int, int]:
        user_msg = self._build_user_message(tweets)
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_string,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = response.choices[0].message.content or ""
                if not raw.strip():
                    log.warning("Empty response from model — retrying")
                    raise ValueError("Empty response")
                log.debug("Raw response: %s", raw[:200])
                # Ollama may not return token counts — default to 0
                input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                return self._parse_response(raw, len(tweets)), raw, input_tokens, output_tokens
            except RateLimitError:
                wait = 2 ** (attempt + 1)
                log.warning("Rate limit hit. Waiting %ds…", wait)
                time.sleep(wait)
            except Exception as e:
                wait = 2 ** (attempt + 1)
                log.warning("API error on attempt %d/%d: %s — retrying in %ds",
                            attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)

        log.error("Max retries exceeded for batch. Defaulting to No Opinion.")
        return ["No Opinion"] * len(tweets), "", 0, 0

    def run(self, mlflow_parent_run_id: str | None = None) -> pd.DataFrame:
        RESULTS_DIR.mkdir(exist_ok=True)
        test_df = pd.read_parquet(self.processed_dir / "tweets_test.parquet")
        test_df.index.name = "tweet_id"
        test_df = test_df.reset_index()

        out_path = RESULTS_DIR / "gpt_results.csv"
        checkpoint_path = RESULTS_DIR / "gpt_results_checkpoint.csv"

        # Resume from checkpoint if exists
        start_from = 0
        if checkpoint_path.exists():
            done = pd.read_csv(checkpoint_path)
            start_from = len(done)
            log.info("Resuming from checkpoint: %d/%d tweets already done", start_from, len(test_df))
        else:
            # Write header only
            pd.DataFrame(columns=["tweet_id", "trading_date", "ticker", "label", "raw_response"]).to_csv(
                checkpoint_path, index=False
            )

        log.info("Ollama (%s): classifying %d tweets …", self.model_string, len(test_df) - start_from)

        with mlflow.start_run(
            run_name="gpt4o_mini",
            nested=True,
            parent_run_id=mlflow_parent_run_id,
        ):
            mlflow.log_params({
                "model_type": "gpt",
                "model_string": self.model_string,
                "backend": "ollama",
                "dataset_split": "test",
                "ticker_scope": "AAPL,TSLA,TSM",
                "batch_size": self.batch_size,
                "temperature": self.temperature,
                "num_few_shot_examples": 8,
                "random_seed": SEED,
            })

            t0 = time.time()
            total_input_tokens, total_output_tokens, retry_count = 0, 0, 0
            remaining_df = test_df.iloc[start_from:].reset_index(drop=True)
            texts = remaining_df["Tweet"].tolist()
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            for i, start in enumerate(range(0, len(texts), self.batch_size)):
                batch_texts = texts[start: start + self.batch_size]
                batch_rows = remaining_df.iloc[start: start + self.batch_size]

                labels, raw, inp_tok, out_tok = self._call_api(batch_texts)
                total_input_tokens += inp_tok
                total_output_tokens += out_tok
                if inp_tok == 0:
                    retry_count += 1

                # Save batch to checkpoint immediately
                batch_result = batch_rows[["tweet_id", "Trading Date", "Stock Name"]].copy()
                batch_result.rename(
                    columns={"Stock Name": "ticker", "Trading Date": "trading_date"}, inplace=True
                )
                batch_result["label"] = labels
                batch_result["raw_response"] = raw
                batch_result.to_csv(checkpoint_path, mode="a", header=False, index=False)

                if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                    total_done = start_from + start + len(batch_texts)
                    log.info("Progress: %d/%d tweets done (%d%%)",
                             total_done, len(test_df),
                             int(total_done / len(test_df) * 100))

            elapsed = time.time() - t0

            # Assemble final result from checkpoint
            result_df = pd.read_csv(checkpoint_path)
            result_df.to_csv(out_path, index=False)
            checkpoint_path.unlink()  # remove checkpoint once complete
            mlflow.log_artifact(str(out_path))

            vc = result_df["label"].value_counts(normalize=True)
            mlflow.log_metrics({
                "processing_time_seconds": round(elapsed, 2),
                "total_tweets_processed": len(result_df),
                "buy_rate": round(vc.get("Buy", 0), 4),
                "sell_rate": round(vc.get("Sell", 0), 4),
                "hold_rate": round(vc.get("Hold", 0), 4),
                "no_opinion_rate": round(vc.get("No Opinion", 0), 4),
                "total_tokens_used": total_input_tokens + total_output_tokens,
                "estimated_cost_usd": 0.0,  # local model, $0 cost
                "retry_count": retry_count,
            })

            log.info(
                "Ollama done in %.1fs (~%.1f tweets/sec). Distribution: %s",
                elapsed, len(result_df) / elapsed if elapsed > 0 else 0, vc.to_dict()
            )
            return result_df
