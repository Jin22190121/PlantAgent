"""Tiny TF-IDF lexical retriever for the Korean+English step corpus.

Used to complement ChromaDB vector search via Reciprocal Rank Fusion (RRF).
Offline, no model download. Korean is tokenized at the word level after
basic punctuation stripping; mixed-script tokens (e.g. 'MSIV', 'PZR')
match exactly.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[가-힣]+|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class TfIdfIndex:
    def __init__(self):
        self.docs: list[str] = []
        self.ids: list[str] = []
        self.metas: list[dict] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.lengths: list[float] = []

    def add(self, docs: Iterable[str], ids: Iterable[str],
            metadatas: Iterable[dict]) -> None:
        for d, i, m in zip(docs, ids, metadatas):
            tokens = tokenize(d)
            tfc = Counter(tokens)
            self.docs.append(d)
            self.ids.append(i)
            self.metas.append(m)
            self.tf.append(tfc)
            for term in tfc:
                self.df[term] += 1
        self._recompute_idf()

    def _recompute_idf(self):
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n / (df + 1))) for t, df in self.df.items()}
        self.lengths = []
        for tfc in self.tf:
            sq = 0.0
            for t, c in tfc.items():
                w = c * self.idf.get(t, 0.0)
                sq += w * w
            self.lengths.append(math.sqrt(sq) or 1.0)

    def search(self, query: str, n: int = 5,
               filter_fn=None) -> list[tuple[str, dict, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_tf = Counter(q_tokens)
        # cosine sim
        scores = []
        q_norm = math.sqrt(sum((c * self.idf.get(t, 0.0)) ** 2
                                for t, c in q_tf.items())) or 1.0
        for i, tfc in enumerate(self.tf):
            if filter_fn and not filter_fn(self.metas[i]):
                continue
            dot = 0.0
            for t, c in q_tf.items():
                if t in tfc:
                    dot += (c * self.idf.get(t, 0.0)) * (tfc[t] * self.idf.get(t, 0.0))
            if dot <= 0:
                continue
            s = dot / (q_norm * self.lengths[i])
            scores.append((s, i))
        scores.sort(reverse=True)
        out = []
        for s, i in scores[:n]:
            out.append((self.ids[i], self.metas[i], s))
        return out


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Each input is an ordered list of doc IDs."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
