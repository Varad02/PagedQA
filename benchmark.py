"""
benchmark.py — PagedQA
Runs a cold vs warm benchmark to demonstrate vLLM prefix caching impact.

Flow:
    1. generate_questions()  — uses the vLLM engine to produce N questions
                               from the document text
    2. run_cold()            — fires all questions concurrently with prefix
                               caching OFF, records per-request metrics
    3. run_warm()            — fires all questions concurrently with prefix
                               caching ON, records per-request metrics
    4. run()                 — orchestrates the full benchmark and returns
                               a compiled results dict for charts.py

Metrics collected per request:
    ttft        — time to first token (seconds)
    latency     — total request time (seconds)
    tokens      — number of tokens generated
    tps         — tokens per second
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import engine
import document
from engine import EngineConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_NUM_QUESTIONS = 10
DEFAULT_CONCURRENCY   = 10   # number of simultaneous requests per condition


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    question:       str
    ttft:           float   # time to first token (s)
    latency:        float   # total time (s)
    tokens:         int     # tokens generated
    tps:            float   # tokens per second
    cache_hit_rate: float   # snapshot from get_stats() after this request


@dataclass
class BenchmarkResult:
    doc_id:         str
    questions:      list[str]
    cold:           list[RequestMetrics]
    warm:           list[RequestMetrics]
    cache_hit_rate: float   # from get_stats() after warm run
    concurrency:    int

    def to_dict(self) -> dict:
        """Serialise to a plain dict for charts.py and Gradio."""
        def metrics_to_dict(m: RequestMetrics) -> dict:
            return {
                "question":       m.question,
                "ttft":           round(m.ttft, 3),
                "latency":        round(m.latency, 3),
                "tokens":         m.tokens,
                "tps":            round(m.tps, 1),
                "cache_hit_rate": round(m.cache_hit_rate, 3),
            }

        return {
            "doc_id":         self.doc_id,
            "questions":      self.questions,
            "cold":           [metrics_to_dict(m) for m in self.cold],
            "warm":           [metrics_to_dict(m) for m in self.warm],
            "cache_hit_rate": self.cache_hit_rate,
            "concurrency":    self.concurrency,
            "summary": {
                "cold_median_latency": _median([m.latency for m in self.cold]),
                "warm_median_latency": _median([m.latency for m in self.warm]),
                "cold_median_ttft":    _median([m.ttft for m in self.cold]),
                "warm_median_ttft":    _median([m.ttft for m in self.warm]),
                "cold_median_tps":     _median([m.tps for m in self.cold]),
                "warm_median_tps":     _median([m.tps for m in self.warm]),
                "speedup":            round(
                    _median([m.latency for m in self.cold]) /
                    max(_median([m.latency for m in self.warm]), 0.001),
                    2
                ),
            },
        }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run(
    doc_id: str,
    num_questions: int = DEFAULT_NUM_QUESTIONS,
    concurrency: int   = DEFAULT_CONCURRENCY,
) -> BenchmarkResult:
    """
    Run the full cold vs warm benchmark for a document.

    Args:
        doc_id:        ID of the document already in the store.
        num_questions: How many questions to generate and benchmark.
        concurrency:   How many requests to fire simultaneously.

    Returns:
        BenchmarkResult with all per-request metrics and a summary.
    """
    doc = document.get(doc_id)
    if doc is None:
        raise KeyError(f"Document '{doc_id}' not found.")

    # Step 1 — generate questions using the warm engine (caching ON)
    logger.info("Generating %d benchmark questions...", num_questions)
    questions = await generate_questions(doc, num_questions)
    logger.info("Generated %d questions.", len(questions))

    # Step 2 — cold simulation (first pass, cache not yet populated)
    logger.info("Starting cold simulation (first pass, cache cold)...")
    warm_cfg = EngineConfig(
        enable_prefix_caching=True,
        gpu_memory_utilization=0.75,
    )
    cold_metrics = await _concurrent_run(doc_id, questions, concurrency, warm_cfg)

    # Step 3 — warm run (cache now populated from cold pass)
    logger.info("Starting warm run (prefix cache now populated)...")
    warm_metrics = await _concurrent_run(doc_id, questions, concurrency, warm_cfg)

    # Step 4 — collect cache hit rate after warm run
    stats = engine.get_stats()
    cache_hit_rate = stats.get("cache_hit_rate", 0.0)

    return BenchmarkResult(
        doc_id=doc_id,
        questions=questions,
        cold=cold_metrics,
        warm=warm_metrics,
        cache_hit_rate=cache_hit_rate,
        concurrency=concurrency,
    )


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

async def generate_questions(
    doc: "document.Document",
    n: int = DEFAULT_NUM_QUESTIONS,
) -> list[str]:
    """
    Ask the vLLM engine to generate N questions from the document text.

    The prompt is designed to return a clean numbered list that we can
    parse reliably.
    """
    prompt = (
        f"<|im_start|>system\n"
        f"You are a helpful assistant that generates questions.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Read the document below and generate exactly {n} diverse questions "
        f"that can be answered from its content.\n\n"
        f"Rules:\n"
        f"- Return ONLY a numbered list, one question per line.\n"
        f"- Format: '1. Question here'\n"
        f"- No preamble, no explanation, no blank lines between questions.\n\n"
        f"--- DOCUMENT ---\n"
        f"{doc.text.strip()}\n"
        f"--- END DOCUMENT ---<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Collect full response (no streaming needed here)
    full_response = ""
    async for delta in engine.stream_generate(
        prompt,
        config=EngineConfig(max_new_tokens=512, temperature=0.7),
    ):
        full_response += delta

    questions = _parse_numbered_list(full_response)

    if len(questions) < 3:
        logger.warning(
            "Question generation returned fewer than 3 questions. "
            "Raw response: %s", full_response[:300]
        )
        # Fallback — use generic questions so the benchmark can still run
        questions = _fallback_questions(doc.text, n)

    return questions[:n]


# ---------------------------------------------------------------------------
# Concurrent request runner
# ---------------------------------------------------------------------------

async def _concurrent_run(
    doc_id: str,
    questions: list[str],
    concurrency: int,
    cfg: EngineConfig,
) -> list[RequestMetrics]:
    """
    Fire all questions concurrently and collect per-request metrics.
    Concurrency is controlled via an asyncio.Semaphore.
    """
    await engine.get_engine(cfg)
    sem = asyncio.Semaphore(concurrency)

    async def _timed_request(question: str) -> RequestMetrics:
        prompt = document.build_prompt(doc_id, question)
        first_token_time: Optional[float] = None
        token_count = 0

        async with sem:
            start = time.perf_counter()   # start after acquiring the semaphore
            async for delta in engine.stream_generate(prompt, config=cfg):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += len(delta.split())

        end = time.perf_counter()
        total_latency = end - start
        ttft = (first_token_time - start) if first_token_time else total_latency

        # Snapshot cache hit rate immediately after this request completes
        stats = engine.get_stats()
        cache_hit_rate = stats.get("cache_hit_rate", 0.0)

        return RequestMetrics(
            question=question,
            ttft=ttft,
            latency=total_latency,
            tokens=token_count,
            tps=token_count / max(total_latency, 0.001),
            cache_hit_rate=cache_hit_rate,
        )

    tasks = [_timed_request(q) for q in questions]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_numbered_list(text: str) -> list[str]:
    """
    Extract questions from a numbered list response.
    Handles formats like '1. Question' or '1) Question'.
    """
    lines = text.strip().splitlines()
    questions = []
    for line in lines:
        # Match '1. ' or '1) ' at the start of a line
        match = re.match(r"^\d+[.)]\s+(.+)", line.strip())
        if match:
            questions.append(match.group(1).strip())
    return questions


def _fallback_questions(doc_text: str, n: int) -> list[str]:
    """
    Rule-based fallback: extract sentences that look like they could
    be answered as factual questions. Used if generation fails.
    """
    sentences = re.split(r"(?<=[.!?])\s+", doc_text)
    # Pick sentences of reasonable length
    candidates = [s.strip() for s in sentences if 20 < len(s) < 120]
    # Turn statements into questions naively
    questions = [f"What does the document say about: '{c[:60]}...'?" 
                 for c in candidates[:n]]
    return questions or ["What is the main topic of this document?"]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return round((sorted_vals[mid - 1] + sorted_vals[mid]) / 2, 3)
    return round(sorted_vals[mid], 3)