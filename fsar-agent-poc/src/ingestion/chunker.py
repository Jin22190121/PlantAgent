"""한국어 친화 청킹 — settings.yaml의 통제변수를 사용."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_settings
from src.ingestion.pdf_loader import PageRecord


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    section: str
    section_title: str
    source_file: str

    @property
    def metadata(self) -> dict:
        return {
            "page": self.page,
            "section": self.section,
            "section_title": self.section_title,
            "source_file": self.source_file,
        }


def build_splitter() -> RecursiveCharacterTextSplitter:
    cfg = load_settings()["ingestion"]
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=cfg["separators"],
        keep_separator=False,
    )


def chunk_pages(pages: Iterable[PageRecord]) -> list[Chunk]:
    splitter = build_splitter()
    chunks: list[Chunk] = []
    for rec in pages:
        for j, piece in enumerate(splitter.split_text(rec.text)):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    id=f"{rec.source_file}:p{rec.page}:c{j}",
                    text=piece.strip(),
                    page=rec.page,
                    section=rec.section,
                    section_title=rec.section_title,
                    source_file=rec.source_file,
                )
            )
    return chunks
