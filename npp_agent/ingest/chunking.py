"""3-tier RAG chunking helpers.

Tier A — step (already in procedures_data.ALL_PROCEDURES)
Tier B — body sliding chunks (200~500 token windows from raw PDF text)
Tier C — CAUTION / NOTE blocks (extracted from raw PDF text)

Used by ProcedureMCP._load() to populate the ChromaDB collection in addition
to Tier-A step chunks.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable

# Approx 4 chars per token for English text (sliding window of ~300 tokens
# = ~1200 chars, with 200-char overlap)
WINDOW_CHARS = 1200
OVERLAP_CHARS = 200


# Mapping of raw text file → (doc_type, source_label)
DOC_FILES = {
    "GOP-Westinghouse-Sec19.txt":  ("GOP", "Westinghouse §19.0"),
    "EOP-E-0_Ginna.txt":           ("EOP", "Ginna E-0"),
    "AOP-10_PointBeach.txt":       ("AOP", "Point Beach AOP-10"),
}


def _normalize(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    return t


def slide_body(text: str, win: int = WINDOW_CHARS,
               overlap: int = OVERLAP_CHARS) -> Iterable[str]:
    """Yield overlapping windows of `text`."""
    text = _normalize(text)
    n = len(text)
    if n <= win:
        if text:
            yield text
        return
    pos = 0
    while pos < n:
        end = min(n, pos + win)
        yield text[pos:end]
        if end == n:
            break
        pos += win - overlap


# CAUTION / NOTE block extraction
CAUTION_RE = re.compile(
    r"(?:CAUTION|NOTE|WARNING)\s*:?\s*(.{20,400}?)(?=(?:CAUTION|NOTE|WARNING|STEP|Step|\Z))",
    re.IGNORECASE | re.DOTALL,
)


def extract_cautions(text: str) -> list[str]:
    text = _normalize(text)
    out = []
    for m in CAUTION_RE.finditer(text):
        chunk = m.group(0).strip()
        if 20 <= len(chunk) <= 600:
            out.append(chunk)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for c in out:
        key = c[:80]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def collect_aux_chunks(procedures_dir: str | Path = "data/procedures") -> list[dict]:
    """Build B-tier (body) and C-tier (CAUTION) chunks from raw PDF text files.

    Returns list of dicts ready for ChromaDB.add():
      {id, document, metadata: {tier, doc_type, source}}
    """
    pdir = Path(procedures_dir)
    chunks: list[dict] = []
    for fname, (doc_type, source) in DOC_FILES.items():
        f = pdir / fname
        if not f.exists():
            continue
        body = f.read_text(errors="replace")
        # B-tier: sliding body windows
        for i, window in enumerate(slide_body(body)):
            chunks.append({
                "id": f"{doc_type}-B-{i:03d}",
                "document": window,
                "metadata": {"tier": "B", "doc_type": doc_type,
                             "source": source, "chunk_no": i,
                             "title": f"{source} body chunk {i}",
                             "step_no": -1,
                             "expected_action": ""},
            })
        # C-tier: CAUTION / NOTE blocks
        for j, c in enumerate(extract_cautions(body)):
            chunks.append({
                "id": f"{doc_type}-C-{j:03d}",
                "document": c,
                "metadata": {"tier": "C", "doc_type": doc_type,
                             "source": source, "chunk_no": j,
                             "title": f"{source} caution {j}",
                             "step_no": -1,
                             "expected_action": ""},
            })
    return chunks
