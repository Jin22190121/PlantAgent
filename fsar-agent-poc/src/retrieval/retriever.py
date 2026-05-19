"""ChromaDB 기반 공통 검색기 — 두 시스템 동일 top-k."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb

from src.config import load_settings, resolve_path
from src.embeddings.embedder import embed


@dataclass
class RetrievedChunk:
    id: str
    text: str
    page: int
    section: str
    section_title: str
    source_file: str
    score: float

    def format_for_prompt(self) -> str:
        return f"[출처: p.{self.page} §{self.section} {self.section_title}]\n{self.text}"


@lru_cache(maxsize=1)
def _get_collection():
    cfg = load_settings()
    persist_dir = str(resolve_path(cfg["paths"]["vectorstore_dir"]))
    client = chromadb.PersistentClient(path=persist_dir)
    name = cfg["retrieval"]["collection_name"]
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": cfg["retrieval"]["distance"]},
    )


def add_chunks(ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
    coll = _get_collection()
    embeddings = embed(texts)
    coll.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)


def count() -> int:
    return _get_collection().count()


def reset_collection() -> None:
    cfg = load_settings()
    persist_dir = str(resolve_path(cfg["paths"]["vectorstore_dir"]))
    client = chromadb.PersistentClient(path=persist_dir)
    name = cfg["retrieval"]["collection_name"]
    try:
        client.delete_collection(name)
    except Exception:
        pass
    _get_collection.cache_clear()
    _get_collection()


def search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    cfg = load_settings()["retrieval"]
    k = top_k or cfg["top_k"]
    coll = _get_collection()
    q_emb = embed([query])[0]
    res = coll.query(query_embeddings=[q_emb], n_results=k)

    out: list[RetrievedChunk] = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        # cosine distance → similarity score
        score = 1.0 - float(dist)
        out.append(
            RetrievedChunk(
                id=cid,
                text=doc,
                page=int(meta.get("page", 0)),
                section=str(meta.get("section", "")),
                section_title=str(meta.get("section_title", "")),
                source_file=str(meta.get("source_file", "")),
                score=score,
            )
        )
    return out


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n---\n\n".join(c.format_for_prompt() for c in chunks)
