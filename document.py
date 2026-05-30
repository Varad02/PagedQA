"""
document.py — PagedQA
Handles file ingestion, text extraction, and the in-memory document store.

Supports:
  - PDF files (via pdfplumber)
  - Plain text files (.txt, .md)

The prompt prefix is built once per document and cached here so that every
call to build_prompt() produces byte-for-byte identical output for the same
doc_id. This is what makes vLLM's prefix caching actually work — any
variation in whitespace or formatting would cause a cache miss.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum characters we'll stuff into the shared prefix.
# Llama 3.1 8B has a 128k token context window, but we leave headroom
# for the question and the generated answer.
# ~12 000 chars ≈ ~3 000 tokens for a typical English document.
MAX_DOC_CHARS = 12_000

SYSTEM_MESSAGE = (
    "You are a precise, helpful assistant. "
    "Answer the user's question using ONLY the document provided below. "
    "If the answer is not in the document, say so clearly."
    " Always use all available information from the document to answer as fully as possible."
    " Never make up an answer or add information that is not in the document. "
    "If the document does not contain enough information to answer the question, say that you don't know, rather than guessing. "
    "If the question is ambiguous, ask for clarification instead of guessing."
    " If the document is too long to fit in the context, use the most relevant parts, but do not ignore the rest entirely. "
    "If the document contains multiple sections, tables, or lists, be sure to consider all of them when formulating your answer."
    " Always provide a complete and accurate answer based on the document, even if the question is simple. "
    "If the document contains contradictory information, identify the contradictions and explain them in your answer. "
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    filename: str
    text: str                        # raw extracted text (truncated if needed)
    page_count: int                  # 0 for plain-text files
    char_count: int
    uploaded_at: float = field(default_factory=time.time)
    truncated: bool = False          # True if text was clipped to MAX_DOC_CHARS

    # The shared prefix is built once and stored here.
    # Every query for this doc uses this exact string as the leading prompt.
    _prefix: Optional[str] = field(default=None, repr=False)

    @property
    def prefix(self) -> str:
        if self._prefix is None:
            self._prefix = _build_prefix(self.text)
        return self._prefix


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

# Simple dict: doc_id (str) -> Document
# For a production system you'd replace this with Redis or a DB.
_store: dict[str, Document] = {}


def get(doc_id: str) -> Optional[Document]:
    """Return a Document by id, or None if not found."""
    return _store.get(doc_id)


def list_docs() -> list[dict]:
    """Return a summary list of all stored documents (for the UI)."""
    return [
        {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "char_count": doc.char_count,
            "page_count": doc.page_count,
            "truncated": doc.truncated,
            "uploaded_at": doc.uploaded_at,
        }
        for doc in _store.values()
    ]


def delete(doc_id: str) -> bool:
    """Remove a document from the store. Returns True if it existed."""
    return _store.pop(doc_id, None) is not None


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest(file_path: str | Path, original_filename: str) -> Document:
    """
    Extract text from a PDF or plain-text file, store it, and return
    the Document object.

    Args:
        file_path:          Path to the uploaded file on disk.
        original_filename:  Original name as uploaded by the user
                            (used for display only).

    Returns:
        A Document with a fresh doc_id.

    Raises:
        ValueError: if the file type is not supported or text extraction fails.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text, page_count = _extract_pdf(path)
    elif suffix in {".txt", ".md"}:
        text, page_count = _extract_text(path), 0
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            "Please upload a PDF, .txt, or .md file."
        )

    if not text.strip():
        raise ValueError(
            "Could not extract any text from this file. "
            "If it is a scanned PDF, OCR is not yet supported."
        )

    truncated = False
    if len(text) > MAX_DOC_CHARS:
        text = text[:MAX_DOC_CHARS]
        truncated = True

    doc = Document(
        doc_id=str(uuid.uuid4()),
        filename=original_filename,
        text=text,
        page_count=page_count,
        char_count=len(text),
        truncated=truncated,
    )

    _store[doc.doc_id] = doc
    return doc


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(doc_id: str, question: str) -> str:
    """
    Build the full prompt for a Q&A request.

    Structure:
        <|system|>
        {SYSTEM_MESSAGE}

        --- DOCUMENT ---
        {document text}
        --- END DOCUMENT ---
        <|user|>
        {question}
        <|assistant|>

    The block between <|system|> and <|user|> is identical for every
    question on the same document, so vLLM's prefix caching reuses those
    KV blocks after the first request.

    Args:
        doc_id:    ID of the document to query against.
        question:  The user's question string.

    Returns:
        The full prompt string ready to send to the vLLM engine.

    Raises:
        KeyError: if doc_id is not found in the store.
    """
    doc = _store.get(doc_id)
    if doc is None:
        raise KeyError(f"Document '{doc_id}' not found. Please upload it first.")

    return (
        doc.prefix
        + f"<|im_start|>user\n{question.strip()}<|im_end|>\n"
        + f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_prefix(document_text: str) -> str:
    """
    Build the invariant (shared) part of the prompt.
    This is called once per document and cached on the Document object.
    """
    return (
        f"<|im_start|>system\n"
        f"{SYSTEM_MESSAGE}\n\n"
        f"--- DOCUMENT ---\n"
        f"{document_text.strip()}\n"
        f"--- END DOCUMENT ---\n"
        f"<|im_end|>\n"
    )


def _extract_pdf(path: Path) -> tuple[str, int]:
    """
    Extract plain text from a PDF using pdfplumber.

    Returns:
        (text, page_count)
    """
    pages_text: list[str] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)

    return "\n\n".join(pages_text), page_count


def _extract_text(path: Path) -> str:
    """Read a plain-text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")