"""PyMuPDF로 한글 PDF에서 페이지별 텍스트와 절 메타데이터를 추출."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

# 한글 FSAR 절 번호 패턴: "1.1", "1.2.3", "제1.1절" 등
SECTION_RE = re.compile(r"^\s*(?:제\s*)?(\d+(?:\.\d+){0,3})\s*(?:절|항)?\s+(.+)$")


@dataclass
class PageRecord:
    page: int          # 1-based
    section: str       # 가장 최근 등장한 절 번호 (예: "1.2.3")
    section_title: str
    text: str
    source_file: str

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_section(line: str, prev_section: str, prev_title: str) -> tuple[str, str]:
    m = SECTION_RE.match(line.strip())
    if not m:
        return prev_section, prev_title
    num, title = m.group(1), m.group(2).strip()
    # 너무 긴 줄은 절 제목이 아닐 가능성이 큼
    if len(title) > 80:
        return prev_section, prev_title
    return num, title


def load_pdf(pdf_path: Path) -> Iterator[PageRecord]:
    """페이지마다 한 PageRecord를 yield."""
    doc = fitz.open(pdf_path)
    section, title = "1", "장 시작"
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        for line in text.splitlines():
            section, title = _detect_section(line, section, title)
        cleaned = _clean_text(text)
        if cleaned.strip():
            yield PageRecord(
                page=i,
                section=section,
                section_title=title,
                text=cleaned,
                source_file=pdf_path.name,
            )
    doc.close()


def _clean_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_jsonl(records: list[PageRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
