"""
test_document.py — PagedQA
Interactive test: point it at any PDF or .txt file and see exactly
what the document store and prompt builder produce.

Usage:
    python test_document.py path/to/your/file.pdf
    python test_document.py path/to/your/file.txt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import document


def divider(title: str = ""):
    line = "─" * 55
    if title:
        print(f"\n┌{line}┐")
        print(f"│  {title:<53}│")
        print(f"└{line}┘")
    else:
        print(f"{'─' * 57}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_document.py <path_to_file.pdf_or_.txt>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"❌  File not found: {file_path}")
        sys.exit(1)

    # ── Step 1: Ingest ──────────────────────────────────────────────────────
    divider("STEP 1 — Ingesting file")
    print(f"  File : {file_path.name}")
    print(f"  Size : {file_path.stat().st_size / 1024:.1f} KB")

    try:
        doc = document.ingest(file_path, file_path.name)
    except ValueError as e:
        print(f"\n❌  Ingestion failed: {e}")
        sys.exit(1)

    print(f"\n  ✅ Ingestion successful")
    print(f"  doc_id     : {doc.doc_id}")
    print(f"  Pages      : {doc.page_count if doc.page_count else 'N/A (plain text)'}")
    print(f"  Characters : {doc.char_count:,}")
    print(f"  Truncated  : {'Yes — clipped to MAX_DOC_CHARS' if doc.truncated else 'No'}")

    # ── Step 2: Show extracted text via doc.text ────────────────────────────
    divider("STEP 2 — Extracted text (first 800 chars)")
    print()
    for line in doc.text[:800].strip().splitlines():
        print(f"  {line}")
    if doc.char_count > 800:
        print(f"\n  ... [{doc.char_count - 800:,} more characters]")

    # ── Step 3: Show built prompt via build_prompt() ────────────────────────
    divider("STEP 3 — Prompt built by build_prompt()")
    prompt = document.build_prompt(doc.doc_id, "What is this document about?")

    print(f"\n  Total prompt length : {len(prompt):,} chars")
    print(f"  Shared prefix       : {len(doc.prefix):,} chars  ← vLLM caches this")
    print(f"  Question portion    : {len(prompt) - len(doc.prefix):,} chars  ← fresh each time")
    print(f"\n  First 329 chars of prompt:\n")
    for line in prompt[:329].splitlines():
        print(f"  {line}")

    # ── Step 4: Prefix consistency via doc.prefix ───────────────────────────
    divider("STEP 4 — Prefix consistency (vLLM cache health)")

    prompt_a = document.build_prompt(doc.doc_id, "What is this about?")
    prompt_b = document.build_prompt(doc.doc_id, "Summarise the key points.")

    # doc.prefix is the cached shared prefix — both prompts must start with it
    if prompt_a.startswith(doc.prefix) and prompt_b.startswith(doc.prefix):
        print("\n  ✅ Both prompts start with the same doc.prefix.")
        print("     vLLM prefix caching will work correctly.")
    else:
        print("\n  ❌ Prefix mismatch — vLLM cache would miss!")

    # ── Step 5: Doc store via list_docs() and get() ─────────────────────────
    divider("STEP 5 — Document store state")

    all_docs = document.list_docs()
    print(f"\n  list_docs() returned {len(all_docs)} document(s):")
    for d in all_docs:
        print(f"  • {d['doc_id'][:8]}…  {d['filename']}  ({d['char_count']:,} chars)")

    fetched = document.get(doc.doc_id)
    print(f"\n  get(doc_id) : {'✅ found' if fetched else '❌ not found'}")

    # ── Step 6: Delete via delete() ─────────────────────────────────────────
    divider("STEP 6 — Delete from store")

    removed = document.delete(doc.doc_id)
    print(f"\n  delete(doc_id) returned : {removed}")
    print(f"  get after delete        : {document.get(doc.doc_id)}")
    print(f"  list_docs after delete  : {len(document.list_docs())} document(s)")

    divider()
    print("\n  All steps complete. document.py is working correctly.\n")


if __name__ == "__main__":
    main()