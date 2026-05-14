"""
engine.py — PagedQA
vLLM AsyncLLMEngine wrapper.

Responsibilities:
  - Lazy singleton initialization of the engine with prefix caching enabled
  - stream_generate(): async generator that yields token deltas
  - get_stats(): pulls block memory stats for the live dashboard

Design notes:
  - The engine is initialized once on the first real request (lazy singleton).
    This keeps the Gradio UI responsive at startup while the model loads.
  - stream_generate() tracks previous output length and yields only the new
    characters each iteration — Gradio's streaming expects deltas, not
    cumulative text.
  - get_stats() reaches into vLLM's scheduler block manager. This is an
    internal API but stable across recent vLLM versions (0.4.x+).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    quantization: str = "awq"   # add this field to EngineConfig
    gpu_memory_utilization: float = 0.85

    # Prefix caching — this is the whole point of the project
    enable_prefix_caching: bool = True

    # Fraction of GPU memory reserved for the KV cache.
    # 0.90 leaves ~10% headroom for activations and other overhead.
    gpu_memory_utilization: float = 0.90

    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.1        # low temp for factual Q&A
    top_p: float = 0.95

    # Tensor parallelism — set to number of GPUs available.
    # 1 for a single-GPU RunPod instance.
    tensor_parallel_size: int = 1


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine: Optional[AsyncLLMEngine] = None
_config: Optional[EngineConfig] = None
_init_lock = asyncio.Lock()


async def get_engine(config: Optional[EngineConfig] = None) -> AsyncLLMEngine:
    """
    Return the running engine, initializing it on first call.

    Args:
        config: EngineConfig to use on first init. Ignored on subsequent calls.
                If None, uses default EngineConfig().

    Returns:
        The initialized AsyncLLMEngine singleton.
    """
    global _engine, _config

    if _engine is not None:
        return _engine

    async with _init_lock:
        # Double-checked locking — another coroutine may have initialized
        # while we were waiting for the lock.
        if _engine is not None:
            return _engine

        cfg = config or EngineConfig()
        _config = cfg

        logger.info(
            "Initializing vLLM engine | model=%s | prefix_caching=%s | "
            "gpu_mem_util=%.2f",
            cfg.model,
            cfg.enable_prefix_caching,
            cfg.gpu_memory_utilization,
        )

        engine_args = AsyncEngineArgs(
            model=cfg.model,
            enable_prefix_caching=cfg.enable_prefix_caching,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=cfg.tensor_parallel_size,
            # Disable the usage stats ping — cleaner logs in demos
            disable_log_stats=False,
        )

        _engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM engine ready.")

    return _engine


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------

async def stream_generate(
    prompt: str,
    config: Optional[EngineConfig] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream token deltas for a given prompt.

    Yields new text only (not cumulative output) so callers can simply
    concatenate what they receive.

    Args:
        prompt: The full prompt string (system + document + question).
        config: Optional EngineConfig. Uses module default if None.

    Yields:
        str — new characters generated since the last yield.

    Example:
        async for delta in stream_generate(prompt):
            print(delta, end="", flush=True)
    """
    cfg = config or _config or EngineConfig()
    engine = await get_engine(cfg)

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
    )

    request_id = random_uuid()
    previous_length = 0

    async for output in engine.generate(prompt, sampling_params, request_id):
        # output.outputs is a list; we always take the first (single) sequence
        if not output.outputs:
            continue

        current_text = output.outputs[0].text
        delta = current_text[previous_length:]
        previous_length = len(current_text)

        if delta:
            yield delta

        # Stop early if the sequence is finished
        if output.outputs[0].finish_reason is not None:
            break


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_stats() -> dict:
    """
    Return current block memory stats from the vLLM scheduler.

    Reaches into the engine's internal block manager — internal API,
    but stable across vLLM 0.4.x+.

    Returns a dict with:
        total_blocks    — total physical KV cache blocks on GPU
        used_blocks     — blocks currently allocated to active requests
        free_blocks     — blocks available for new requests
        utilization_pct — used / total * 100
        cache_hit_rate  — prefix cache hit rate since engine start (0.0–1.0)
        prefix_caching  — whether prefix caching is enabled
        model           — model name
        ready           — False if engine not yet initialized

    If the engine is not initialized yet, returns a zeroed-out dict with
    ready=False so the dashboard can show a "waiting" state gracefully.
    """
    if _engine is None:
        return {
            "ready": False,
            "total_blocks": 0,
            "used_blocks": 0,
            "free_blocks": 0,
            "utilization_pct": 0.0,
            "cache_hit_rate": 0.0,
            "prefix_caching": False,
            "model": "",
        }

    try:
        # Navigate into the scheduler's block manager
        scheduler = _engine.engine.scheduler
        block_manager = scheduler.block_manager

        total = block_manager.get_num_total_gpu_blocks()
        free = block_manager.get_num_free_gpu_blocks()
        used = total - free

        # Cache hit rate — available when prefix caching is on
        hit_rate = 0.0
        try:
            stats = _engine.engine.stat_logger.stats
            hit_rate = getattr(stats, "gpu_prefix_cache_hit_rate", 0.0) or 0.0
        except Exception:
            pass

        return {
            "ready": True,
            "total_blocks": total,
            "used_blocks": used,
            "free_blocks": free,
            "utilization_pct": round((used / total * 100) if total > 0 else 0.0, 1),
            "cache_hit_rate": round(float(hit_rate), 3),
            "prefix_caching": _config.enable_prefix_caching if _config else False,
            "model": _config.model if _config else "",
        }

    except Exception as e:
        logger.warning("Could not read block stats: %s", e)
        return {
            "ready": True,
            "total_blocks": 0,
            "used_blocks": 0,
            "free_blocks": 0,
            "utilization_pct": 0.0,
            "cache_hit_rate": 0.0,
            "prefix_caching": False,
            "model": _config.model if _config else "",
        }


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

async def shutdown() -> None:
    """
    Gracefully shut down the engine.
    Call this when the server is stopping to free GPU memory cleanly.
    """
    global _engine
    if _engine is not None:
        logger.info("Shutting down vLLM engine...")
        await _engine.abort_all()
        _engine = None
        logger.info("vLLM engine shut down.")