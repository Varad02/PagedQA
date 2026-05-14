# PagedQA

Document Q&A powered by [vLLM](https://github.com/vllm-project/vllm) with **prefix caching**. Upload a PDF or text file, ask questions against it, and benchmark the latency speedup from KV cache reuse.

## How it works

Every question sent to the same document shares an identical prompt prefix — the system message + document text. vLLM's prefix caching stores those KV blocks on the GPU after the first request, so subsequent questions skip recomputing them entirely. The benchmark tab measures the difference between cold (no cache) and warm (cached) conditions.

## Features

- **Upload** — ingest PDF or `.txt`/`.md` files; get a `doc_id` for further use
- **Ask** — stream answers against an uploaded document with live latency display
- **Benchmark** — run concurrent cold vs warm requests and visualize the speedup with Plotly charts
- **Memory** — live KV block visualization polling vLLM's internal block manager every 2 seconds

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU. The engine loads `meta-llama/Llama-3.1-8B-Instruct` by default — set a Hugging Face token if the model is gated.

## Run

```bash
python app.py
```

Opens on port `7860` and generates a public `gradio.live` share link.

## Configuration

Edit `EngineConfig` in [engine.py](engine.py) to change defaults:

| Field | Default | Description |
|---|---|---|
| `model` | `meta-llama/Llama-3.1-8B-Instruct` | vLLM model to load |
| `enable_prefix_caching` | `True` | Toggle KV cache prefix reuse |
| `gpu_memory_utilization` | `0.90` | Fraction of GPU VRAM for KV cache |
| `max_new_tokens` | `512` | Max tokens per answer |
| `tensor_parallel_size` | `1` | Number of GPUs |

## Project structure

```
app.py        — Gradio UI (4 tabs: Upload, Ask, Benchmark, Memory)
engine.py     — vLLM AsyncLLMEngine singleton + streaming + stats
document.py   — File ingestion, text extraction, prompt builder, in-memory store
benchmark.py  — Cold vs warm benchmark runner with concurrent request support
charts.py     — Plotly figures: latency, throughput, cache hit rate
```

## Dependencies

- `vllm >= 0.4.0`
- `gradio >= 4.0.0`
- `pdfplumber >= 0.10.0`
- `plotly >= 5.0.0`
