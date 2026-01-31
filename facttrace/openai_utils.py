"""Shared helpers for OpenAI calls."""

import json
import time
from typing import Any, Dict, Tuple

from .config import MAX_RETRIES, MODEL, PRICING, TEMPERATURE, client


def _tight_json_extract(text: str) -> str:
    """Best-effort extraction of a single JSON object from model output."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= 0:
        raise RuntimeError(f"No JSON object found in model output: {text!r}")
    return text[start:end]


def call_openai_json(
    system: str,
    user: str,
    *,
    max_retries: int = MAX_RETRIES,
    temperature: float = TEMPERATURE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Call the Responses API and expect a JSON object back.
    Returns: (parsed_json, meta)
    meta includes elapsed, usage, cost, raw_text.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            start = time.perf_counter()

            response = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )

            elapsed = time.perf_counter() - start
            usage = response.usage

            raw_text = (response.output_text or "").strip()
            if not raw_text:
                raise RuntimeError("Empty model output")

            json_text = _tight_json_extract(raw_text)
            parsed = json.loads(json_text)

            in_price, out_price = PRICING[MODEL]
            cost = (
                usage.input_tokens * in_price
                + usage.output_tokens * out_price
            ) / 1_000_000

            meta = {
                "elapsed": elapsed,
                "usage": usage,
                "cost": cost,
                "raw_text": raw_text,
            }
            return parsed, meta

        except Exception as e:
            last_error = e
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"OpenAI call failed: {last_error}")


__all__ = ["call_openai_json", "_tight_json_extract"]
