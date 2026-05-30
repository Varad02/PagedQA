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
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── container ── */
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }

/* ── doc-id output — green monospace ── */
#docid-out textarea {
    font-family: "JetBrains Mono", "Fira Code", monospace !important;
    background: #f0fdf4 !important;
    border-color: #86efac !important;
    color: #15803d !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
}

/* ── answer textbox — better readability ── */
#answer-out textarea {
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}

/* ── primary buttons ── */
button.primary {
    background: linear-gradient(135deg, #1D9E75 0%, #0f7a5a 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(29,158,117,0.30) !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(29,158,117,0.45) !important;
}

/* ── tab nav ── */
.tab-nav button { font-weight: 500; font-size: 0.9rem; }
"""


# ---------------------------------------------------------------------------
# Shared HTML helpers
# ---------------------------------------------------------------------------

def _tab_header(title: str, subtitle: str) -> str:
    return (
        f'<div style="margin-bottom:1.25rem;">'
        f'<h3 style="font-size:1.05rem;font-weight:700;color:#0f172a;margin:0 0 4px;">{title}</h3>'
        f'<p style="font-size:0.85rem;color:#64748b;margin:0;">{subtitle}</p>'
        f'</div>'
    )


def _error_html(msg: str) -> str:
    return (
        f'<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;'
        f'padding:1rem 1.25rem;color:#991b1b;font-size:0.9rem;">'
        f'❌ {msg}</div>'
    )


def _info_html(msg: str) -> str:
    return (
        f'<div style="background:#eff6ff;border:1px solid #93c5fd;border-radius:10px;'
        f'padding:1rem 1.25rem;color:#1e40af;font-size:0.9rem;">'
        f'{msg}</div>'
    )


def _summary_html(summary: dict, cache_hit_rate: float, concurrency: int) -> str:
    metrics = [
        ("Cold Median Latency", f"{summary['cold_median_latency']:.3f}s", "#E8593C"),
        ("Warm Median Latency", f"{summary['warm_median_latency']:.3f}s", "#1D9E75"),
        ("Speedup",             f"{summary['speedup']}×",                 "#7C3AED"),
        ("Cold Median TTFT",    f"{summary['cold_median_ttft']:.3f}s",    "#E8593C"),
        ("Warm Median TTFT",    f"{summary['warm_median_ttft']:.3f}s",    "#1D9E75"),
        ("Cache Hit Rate",      f"{cache_hit_rate * 100:.1f}%",           "#1D9E75"),
        ("Cold Median TPS",     f"{summary['cold_median_tps']:.1f}",      "#E8593C"),
        ("Warm Median TPS",     f"{summary['warm_median_tps']:.1f}",      "#1D9E75"),
    ]

    cards = "".join(
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
        f'padding:1rem 1.25rem;border-top:3px solid {color};">'
        f'<div style="font-size:0.72rem;color:#64748b;text-transform:uppercase;'
        f'letter-spacing:0.6px;font-weight:600;margin-bottom:5px;">{label}</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:#0f172a;">{value}</div>'
        f'</div>'
        for label, value, color in metrics
    )

    return (
        '<div style="padding:0.25rem 0;">'
        '<div style="display:inline-flex;align-items:center;gap:8px;padding:4px 14px;'
        'background:#d1fae5;border-radius:999px;margin-bottom:1rem;">'
        '<span style="color:#065f46;font-size:0.82rem;font-weight:600;">✅ Benchmark complete</span>'
        '</div>'
        '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:10px;">'
        f'{cards}'
        '</div>'
        f'<div style="margin-top:0.75rem;font-size:0.8rem;color:#94a3b8;">Concurrency: {concurrency}</div>'
        '</div>'
    )


# ---------------------------------------------------------------------------
# Tab 1 — Upload
# ---------------------------------------------------------------------------

def handle_upload(file) -> tuple[str, str, str]:
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
    gr.HTML(_tab_header("Upload a document", "Supports PDF and plain text (.txt, .md)."))

    with gr.Row():
        file_input = gr.File(label="Upload PDF or .txt", file_types=[".pdf", ".txt", ".md"])

    with gr.Row():
        doc_id_out = gr.Textbox(
            label="doc_id  —  copy this for the Ask and Benchmark tabs",
            interactive=False,
            elem_id="docid-out",
        )

    with gr.Row():
        info_out    = gr.Textbox(label="Upload info",   lines=6, interactive=False)
        preview_out = gr.Textbox(label="Text preview",  lines=6, interactive=False)

    file_input.change(
        fn=handle_upload,
        inputs=[file_input],
        outputs=[doc_id_out, info_out, preview_out],
    )


# ---------------------------------------------------------------------------
# Tab 2 — Ask
# ---------------------------------------------------------------------------

async def handle_ask(doc_id: str, question: str):
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


def build_ask_tab():
    gr.HTML(_tab_header("Ask a question", "Paste the doc_id from the Upload tab, then ask anything about your document."))

    with gr.Row():
        doc_id_in   = gr.Textbox(label="doc_id", placeholder="Paste doc_id here…")
        question_in = gr.Textbox(label="Question", placeholder="What is this document about?", lines=2)

    ask_btn = gr.Button("Ask  ▶", variant="primary")

    with gr.Row():
        answer_out  = gr.Textbox(label="Answer", lines=10, interactive=False, elem_id="answer-out")
        latency_out = gr.Textbox(label="Latency", interactive=False, scale=0, min_width=110)

    return ask_btn, [doc_id_in, question_in], [answer_out, latency_out]


# ---------------------------------------------------------------------------
# Tab 3 — Benchmark
# ---------------------------------------------------------------------------

async def handle_benchmark(doc_id: str, num_questions: int, concurrency: int):
    if not doc_id.strip():
        yield _error_html("Please enter a doc_id."), None, None, None
        return

    yield _info_html("⏳ Running benchmark — this takes a few minutes…"), None, None, None

    try:
        result = await benchmark.run(
            doc_id=doc_id.strip(),
            num_questions=int(num_questions),
            concurrency=int(concurrency),
        )
    except Exception as e:
        yield _error_html(f"Benchmark failed: {e}"), None, None, None
        return

    result_dict = result.to_dict()
    summary_html = _summary_html(
        result_dict["summary"],
        result_dict["cache_hit_rate"],
        result_dict["concurrency"],
    )

    fig_latency, fig_throughput, fig_cache = charts.all_charts(result_dict)
    yield summary_html, fig_latency, fig_throughput, fig_cache


def build_benchmark_tab():
    gr.HTML(_tab_header(
        "Benchmark prefix caching",
        "Runs cold (no cache) and warm (cached) conditions concurrently and plots the difference.",
    ))

    with gr.Row():
        doc_id_in    = gr.Textbox(label="doc_id", placeholder="Paste doc_id here…")
        num_q_slider = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Questions")
        conc_slider  = gr.Slider(minimum=1, maximum=20, value=10, step=1, label="Concurrency")

    run_btn     = gr.Button("Run benchmark  ▶", variant="primary")
    summary_out = gr.HTML()

    with gr.Row():
        latency_plot    = gr.Plot(label="Latency: cold vs warm")
        throughput_plot = gr.Plot(label="Throughput: tokens / second")

    cache_plot = gr.Plot(label="Cache hit rate over requests")

    return run_btn, [doc_id_in, num_q_slider, conc_slider], [summary_out, latency_plot, throughput_plot, cache_plot]


# ---------------------------------------------------------------------------
# Tab 4 — Memory
# ---------------------------------------------------------------------------

def _render_block_grid(stats: dict) -> str:
    if not stats.get("ready"):
        return (
            '<div style="display:flex;align-items:center;gap:14px;padding:1.5rem;'
            'background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;">'
            '<div style="font-size:1.6rem;">⏳</div>'
            '<div>'
            '<div style="font-weight:600;color:#1e293b;font-size:0.95rem;">Engine not initialized</div>'
            '<div style="color:#64748b;font-size:0.84rem;margin-top:3px;">Send a query first to initialize the vLLM engine.</div>'
            '</div></div>'
        )

    total = stats["total_blocks"]
    used  = stats["used_blocks"]
    free  = stats["free_blocks"]

    if total == 0:
        return "<p style='padding:1rem'>No block data available.</p>"

    hit_rate      = stats.get("cache_hit_rate", 0.0)
    cached_blocks = int(used * hit_rate)
    plain_used    = used - cached_blocks

    cells = []
    for i in range(total):
        if i < cached_blocks:
            color = "#1D9E75"
            title = "Cached (shared prefix)"
        elif i < used:
            color = "#E8593C"
            title = "Occupied"
        else:
            color = "#e2e8f0"
            title = "Free"

        cells.append(
            f'<div title="{title}" style="'
            f'width:12px;height:12px;border-radius:3px;'
            f'background:{color};display:inline-block;margin:1.5px;'
            f'transition:opacity 0.2s;"></div>'
        )

    def _stat_pill(label: str, value: str, color: str) -> str:
        return (
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'background:#fff;border:1px solid #e2e8f0;border-top:3px solid {color};'
            f'border-radius:10px;padding:0.6rem 1rem;min-width:110px;">'
            f'<span style="font-size:1.2rem;font-weight:700;color:#0f172a;">{value}</span>'
            f'<span style="font-size:0.72rem;color:#64748b;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-top:2px;">{label}</span>'
            f'</div>'
        )

    stats_row = (
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:1.25rem;">'
        + _stat_pill("Total blocks",   str(total),                          "#94a3b8")
        + _stat_pill("Cached prefix",  str(cached_blocks),                  "#1D9E75")
        + _stat_pill("Occupied",       str(plain_used),                     "#E8593C")
        + _stat_pill("Free",           str(free),                           "#94a3b8")
        + _stat_pill("Utilization",    f"{stats['utilization_pct']}%",      "#7C3AED")
        + _stat_pill("Cache hit rate", f"{stats['cache_hit_rate']*100:.1f}%", "#1D9E75")
        + '</div>'
    )

    legend = (
        '<div style="margin-top:1rem;display:flex;gap:16px;flex-wrap:wrap;font-size:0.82rem;color:#475569;">'
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;'
        f'background:#1D9E75;margin-right:5px;vertical-align:middle;"></span>Cached prefix</span>'
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;'
        f'background:#E8593C;margin-right:5px;vertical-align:middle;"></span>Occupied</span>'
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:3px;'
        f'background:#e2e8f0;margin-right:5px;vertical-align:middle;"></span>Free</span>'
        '</div>'
    )

    grid_html = "".join(cells)

    return (
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:1.5rem;">'
        + stats_row
        + f'<div style="line-height:0;">{grid_html}</div>'
        + legend
        + '</div>'
    )


def handle_memory_poll() -> str:
    stats = engine.get_stats()
    return _render_block_grid(stats)


def build_memory_tab() -> gr.HTML:
    gr.HTML(_tab_header(
        "Live KV block memory",
        "Updated after every query and benchmark run.",
    ))

    memory_html = gr.HTML(value=handle_memory_poll())
    return memory_html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="emerald",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="PagedQA", theme=theme, css=CUSTOM_CSS) as demo:

        # ── Hero banner ──────────────────────────────────────────────────────
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#0f172a 0%,#0a2e1e 100%);
                    border-radius:14px;padding:2rem 2.5rem;margin-bottom:0.5rem;">
            <div style="display:inline-block;background:rgba(29,158,117,0.2);
                        border:1px solid rgba(29,158,117,0.5);color:#5eead4;
                        padding:3px 12px;border-radius:999px;font-size:0.73rem;
                        font-weight:700;letter-spacing:0.8px;text-transform:uppercase;
                        margin-bottom:0.8rem;">
                ⚡ vLLM · Prefix Caching
            </div>
            <h1 style="font-size:2.2rem;font-weight:800;color:#fff;
                       margin:0 0 0.5rem;letter-spacing:-0.5px;">
                PagedQA
            </h1>
            <p style="color:rgba(255,255,255,0.68);font-size:0.92rem;
                      line-height:1.6;margin:0;max-width:580px;">
                Document Q&amp;A powered by
                <a href="https://github.com/vllm-project/vllm"
                   style="color:#4ade80;text-decoration:none;font-weight:500;">vLLM</a>
                with prefix caching. Upload a document, ask questions, and benchmark
                the speedup from KV cache reuse.
            </p>
        </div>
        """)

        with gr.Tab("📄 Upload"):
            build_upload_tab()

        with gr.Tab("💬 Ask"):
            ask_btn, ask_inputs, ask_outputs = build_ask_tab()

        with gr.Tab("📊 Benchmark"):
            run_btn, bench_inputs, bench_outputs = build_benchmark_tab()

        with gr.Tab("🧠 Memory"):
            memory_html = build_memory_tab()

        # Refresh memory panel after each query and after benchmark completes
        ask_btn.click(fn=handle_ask, inputs=ask_inputs, outputs=ask_outputs).then(
            fn=handle_memory_poll, outputs=[memory_html]
        )
        run_btn.click(fn=handle_benchmark, inputs=bench_inputs, outputs=bench_outputs).then(
            fn=handle_memory_poll, outputs=[memory_html]
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
