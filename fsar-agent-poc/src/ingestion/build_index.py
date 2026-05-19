"""엔드투엔드 인덱싱 CLI: data/raw/*.pdf → 청크 → ChromaDB.

사용법:
    python -m src.ingestion.build_index
    python -m src.ingestion.build_index --reset
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_settings, resolve_path
from src.ingestion.chunker import chunk_pages
from src.ingestion.pdf_loader import load_pdf, save_jsonl
from src.retrieval.retriever import add_chunks, count, reset_collection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 ChromaDB collection을 비우고 재인덱싱",
    )
    args = parser.parse_args()

    cfg = load_settings()
    raw_dir = resolve_path(cfg["paths"]["raw_pdf_dir"])
    processed_dir = resolve_path(cfg["paths"]["processed_dir"])

    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[!] {raw_dir} 에 PDF가 없습니다. data/raw/ 에 FSAR PDF를 배치하세요.")
        return 1

    if args.reset:
        print("[i] ChromaDB collection 초기화...")
        reset_collection()

    total_chunks = 0
    for pdf in pdfs:
        print(f"[+] 처리 중: {pdf.name}")
        pages = list(load_pdf(pdf))
        print(f"    페이지 수: {len(pages)}")

        jsonl_path = processed_dir / f"{pdf.stem}.pages.jsonl"
        save_jsonl(pages, jsonl_path)
        print(f"    페이지 JSONL → {jsonl_path}")

        chunks = chunk_pages(pages)
        print(f"    청크 수: {len(chunks)}")

        if chunks:
            add_chunks(
                ids=[c.id for c in chunks],
                texts=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
            total_chunks += len(chunks)

    print(f"\n[OK] 인덱싱 완료. ChromaDB 총 청크 수: {count()} (이번 추가: {total_chunks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
