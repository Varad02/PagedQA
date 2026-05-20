"""
app.py — PagedQA
Gradio UI with four tabs:
    1. Upload   — ingest a PDF or .txt file, get a doc_id
    2. Ask      — stream answers against an uploaded document
    3. Benchmark — cold vs warm benchmark with Plotly charts
    4. Memory   — live KV block memory visualization

Run:
    python app.py
"""

from __future__ import annotations

import sys

# Flush stdout immediately so Colab sees output
sys.stdout.reconfigure(line_buffering=True)

import asyncio
import logging
import time

import gradio as gr

import document
import engine
import benchmark
import charts
from engine import EngineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tab 1 — Upload
# ---------------------------------------------------------------------------

def handle_upload(file) -> tuple[str, str, str]:
    """
    Ingest the uploaded file and return:
        (doc_id, info string, text preview)
    """
    if file is None:
        return "", "No file uploaded.", ""

    try:
        doc = document.ingest(file.name, file.name.split("/")[-1])
    except ValueError as e:
        return "", f"❌ {e}", ""

    info = (
        f"✅ Uploaded successfully\n"
        f"doc_id     : {doc.doc_id}\n"
        f"Pages      : {doc.page_count or 'N/A (plain text)'}\n"
        f"Characters : {doc.char_count:,}\n"
        f"Truncated  : {'Yes' if doc.truncated else 'No'}"
    )
    preview = doc.text[:1000] + ("\n\n[... truncated for preview]" if doc.char_count > 1000 else "")
    return doc.doc_id, info, preview


def build_upload_tab() -> None:
    gr.Markdown("### Upload a document\nSupports PDF and plain text (.txt, .md).")

    with gr.Row():
        file_input = gr.File(label="Upload PDF or .txt", file_types=[".pdf", ".txt", ".md"])

    with gr.Row():
        doc_id_out = gr.Textbox(label="doc_id (copy this for the Ask and Benchmark tabs)", interactive=False)

    with gr.Row():
        info_out    = gr.Textbox(label="Upload info", lines=6,  interactive=False)
        preview_out = gr.Textbox(label="Text preview", lines=6, interactive=False)

    file_input.change(
        fn=handle_upload,
        inputs=[file_input],
        outputs=[doc_id_out, info_out, preview_out],
    )


# ---------------------------------------------------------------------------
# Tab 2 — Ask
# ---------------------------------------------------------------------------

async def handle_ask(doc_id: str, question: str):
    """
    Async generator — yields (answer_so_far, latency_string) tuples
    so Gradio can stream the answer token by token.
    """
    if not doc_id.strip():
        yield "Please enter a doc_id from the Upload tab.", ""
        return
    if not question.strip():
        yield "Please enter a question.", ""
        return

    try:
        prompt = document.build_prompt(doc_id.strip(), question.strip())
    except KeyError as e:
        yield str(e), ""
        return

    answer = ""
    start  = time.perf_counter()

    async for delta in engine.stream_generate(prompt):
        answer += delta
        elapsed = time.perf_counter() - start
        yield answer, f"{elapsed:.2f}s"


def build_ask_tab() -> None:
    gr.Markdown("### Ask a question\nPaste the doc_id from the Upload tab.")

    with gr.Row():
        doc_id_in   = gr.Textbox(label="doc_id", placeholder="Paste doc_id here...")
        question_in = gr.Textbox(label="Question", placeholder="What is this document about?", lines=2)

    ask_btn = gr.Button("Ask ▶", variant="primary")

    with gr.Row():
        answer_out  = gr.Textbox(label="Answer", lines=10, interactive=False)
        latency_out = gr.Textbox(label="Latency", interactive=False, scale=0, min_width=100)

    ask_btn.click(
        fn=handle_ask,
        inputs=[doc_id_in, question_in],
        outputs=[answer_out, latency_out],
    )


# ---------------------------------------------------------------------------
# Tab 3 — Benchmark
# ---------------------------------------------------------------------------

async def handle_benchmark(doc_id: str, num_questions: int, concurrency: int):
    """
    Run the full cold vs warm benchmark and return three Plotly figures
    plus a summary text.
    """
    if not doc_id.strip():
        empty = charts._base_layout()
        yield "Please enter a doc_id.", None, None, None
        return

    yield "⏳ Running benchmark — this takes a few minutes...", None, None, None

    try:
        result = await benchmark.run(
            doc_id=doc_id.strip(),
            num_questions=int(num_questions),
            concurrency=int(concurrency),
        )
    except Exception as e:
        yield f"❌ Benchmark failed: {e}", None, None, None
        return

    result_dict = result.to_dict()
    summary     = result_dict["summary"]

    summary_text = (
        f"✅ Benchmark complete\n\n"
        f"Cold  median latency : {summary['cold_median_latency']:.3f}s\n"
        f"Warm  median latency : {summary['warm_median_latency']:.3f}s\n"
        f"Speedup              : {summary['speedup']}×\n\n"
        f"Cold  median TTFT    : {summary['cold_median_ttft']:.3f}s\n"
        f"Warm  median TTFT    : {summary['warm_median_ttft']:.3f}s\n\n"
        f"Cold  median tps     : {summary['cold_median_tps']:.1f}\n"
        f"Warm  median tps     : {summary['warm_median_tps']:.1f}\n\n"
        f"Cache hit rate       : {result_dict['cache_hit_rate'] * 100:.1f}%\n"
        f"Concurrency          : {result_dict['concurrency']}"
    )

    fig_latency, fig_throughput, fig_cache = charts.all_charts(result_dict)
    yield summary_text, fig_latency, fig_throughput, fig_cache


def build_benchmark_tab() -> None:
    gr.Markdown(
        "### Benchmark prefix caching\n"
        "Runs cold (no cache) and warm (cached) conditions concurrently "
        "and plots the difference."
    )

    with gr.Row():
        doc_id_in      = gr.Textbox(label="doc_id", placeholder="Paste doc_id here...")
        num_q_slider   = gr.Slider(minimum=5, maximum=20, value=10, step=1,  label="Number of questions")
        conc_slider    = gr.Slider(minimum=1, maximum=20, value=10, step=1,  label="Concurrency")

    run_btn     = gr.Button("Run benchmark ▶", variant="primary")
    summary_out = gr.Textbox(label="Summary", lines=14, interactive=False)

    with gr.Row():
        latency_plot    = gr.Plot(label="Latency: cold vs warm")
        throughput_plot = gr.Plot(label="Throughput: tokens / second")

    cache_plot = gr.Plot(label="Cache hit rate over requests")

    run_btn.click(
        fn=handle_benchmark,
        inputs=[doc_id_in, num_q_slider, conc_slider],
        outputs=[summary_out, latency_plot, throughput_plot, cache_plot],
    )


# ---------------------------------------------------------------------------
# Tab 4 — Memory
# ---------------------------------------------------------------------------

def _render_block_grid(stats: dict) -> str:
    """
    Render a simple HTML block grid showing GPU KV cache state.
    Each square represents one physical block:
        gray  — free
        teal  — occupied
        amber — shared / prefix-cached
    """
    if not stats.get("ready"):
        return "<p style='color: var(--body-text-color); padding: 1rem;'>⏳ Engine not initialized yet. Send a query first.</p>"

    total = stats["total_blocks"]
    used  = stats["used_blocks"]
    free  = stats["free_blocks"]

    if total == 0:
        return "<p style='padding:1rem'>No block data available.</p>"

    # Estimate cached blocks from cache_hit_rate
    hit_rate      = stats.get("cache_hit_rate", 0.0)
    cached_blocks = int(used * hit_rate)
    plain_used    = used - cached_blocks

    cells = []
    for i in range(total):
        if i < cached_blocks:
            color   = "#1D9E75"   # teal — cached prefix blocks
            title   = "Cached (shared prefix)"
        elif i < used:
            color   = "#E8593C"   # coral — occupied
            title   = "Occupied"
        else:
            color   = "#D3D1C7"   # gray — free
            title   = "Free"

        cells.append(
            f'<div title="{title}" style="'
            f'width:12px;height:12px;border-radius:2px;'
            f'background:{color};display:inline-block;margin:1px;">'
            f'</div>'
        )

    grid_html = "".join(cells)

    legend = (
        '<div style="margin-top:12px;font-size:13px;display:flex;gap:16px;">'
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#1D9E75;margin-right:4px;vertical-align:middle;"></span>Cached prefix ({cached_blocks})</span>'
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#E8593C;margin-right:4px;vertical-align:middle;"></span>Occupied ({plain_used})</span>'
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:#D3D1C7;margin-right:4px;vertical-align:middle;"></span>Free ({free})</span>'
        '</div>'
    )

    stats_line = (
        f'<div style="margin-bottom:8px;font-size:13px;">'
        f'<b>GPU KV cache blocks</b> — '
        f'Total: {total} &nbsp;|&nbsp; '
        f'Used: {used} ({stats["utilization_pct"]}%) &nbsp;|&nbsp; '
        f'Cache hit rate: {stats["cache_hit_rate"]*100:.1f}%'
        f'</div>'
    )

    return f'<div style="padding:1rem;">{stats_line}{grid_html}{legend}</div>'


def handle_memory_poll() -> str:
    stats = engine.get_stats()
    return _render_block_grid(stats)


def build_memory_tab() -> None:
    gr.Markdown(
        "### Live KV block memory\n"
        "Polls vLLM's block manager every 2 seconds. "
        "Send queries or run the benchmark to see blocks fill up."
    )

    memory_html = gr.HTML(value=handle_memory_poll())

    # Poll every 2 seconds using gr.Timer
    timer = gr.Timer(value=2)
    timer.tick(fn=handle_memory_poll, outputs=[memory_html])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="PagedQA", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# PagedQA\n"
            "Document Q&A powered by [vLLM](https://github.com/vllm-project/vllm) "
            "with prefix caching. Upload a document, ask questions, and benchmark "
            "the speedup from KV cache reuse."
        )

        with gr.Tab("📄 Upload"):
            build_upload_tab()

        with gr.Tab("💬 Ask"):
            build_ask_tab()

        with gr.Tab("📊 Benchmark"):
            build_benchmark_tab()

        with gr.Tab("🧠 Memory"):
            build_memory_tab()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",   # accessible from outside (needed on RunPod / Colab)
        server_port=7860,
        share=True,              # generates a public gradio.live link
    )