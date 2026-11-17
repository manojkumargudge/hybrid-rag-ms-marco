"""Tiny MS MARCO-format raw data and a deterministic stub encoder for tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

TOPICS = [
    "diabetes insulin blood sugar glucose",
    "photosynthesis chlorophyll sunlight plants",
    "volcano lava magma eruption",
    "bitcoin blockchain cryptocurrency mining",
    "asthma inhaler airways wheezing",
    "glacier ice melting climate",
    "piano keys chords music",
    "python programming language code",
]
FILLER = "general information about many unrelated everyday things"


def write_raw_fixture(raw_dir: Path) -> None:
    """60 passages, 8 dev-small queries, 1 relevant + 12-candidate pool each."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    passages = []
    for pid in range(60):
        topic = TOPICS[pid % len(TOPICS)]
        if pid < len(TOPICS):  # pids 0-7: the relevant passage for query pid
            text = f"{topic} explained in detail: {topic}"
        else:
            text = f"{FILLER} {topic.split()[0]} note {pid}"
        passages.append((pid, text))
    # A mojibake passage to exercise text normalization.
    passages[59] = (59, "cafÃ© menu " + FILLER)

    (raw_dir / "collection.tsv").write_text(
        "".join(f"{pid}\t{text}\n" for pid, text in passages), encoding="utf-8"
    )
    (raw_dir / "queries.dev.small.tsv").write_text(
        "".join(
            f"{1000 + i}\twhat is {topic.split()[0]} {topic.split()[1]}\n"
            for i, topic in enumerate(TOPICS)
        ),
        encoding="utf-8",
    )
    (raw_dir / "qrels.dev.small.tsv").write_text(
        "".join(f"{1000 + i}\t0\t{i}\t1\n" for i in range(len(TOPICS))),
        encoding="utf-8",
    )
    lines = []
    texts = dict(passages)
    for i, topic in enumerate(TOPICS):
        query = f"what is {topic.split()[0]} {topic.split()[1]}"
        # Pool: relevant passage last (file order is not rank order) + others.
        pool = [pid for pid in range(8, 60) if pid % 5 == i % 5][:11] + [i]
        lines += [f"{1000 + i}\t{pid}\t{query}\t{texts[pid]}\n" for pid in pool]
    (raw_dir / "top1000.dev").write_text("".join(lines), encoding="utf-8")


class HashEncoder:
    """Deterministic bag-of-words hashing encoder (SentenceTransformer-like)."""

    dim = 64

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **_):
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vectors[row, bucket] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors /= np.where(norms == 0, 1.0, norms)
        return vectors
