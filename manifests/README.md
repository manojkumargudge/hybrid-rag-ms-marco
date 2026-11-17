# Artifact manifests

Committed copies of `artifacts/<version>/manifest.json`. The artifacts themselves
(passages, indices, raw MS MARCO files) are never committed; these manifests record
how each version was built: source URLs and checksums, seeds, sampling rules, counts,
model revisions, and the sha256 of every output file. The app checks those hashes
before it loads anything.

| Version | Status | Hard negatives |
|---|---|---|
| `v2.json` | Superseded, kept for history | Top-50 per query from the top1000.dev pool, **re-ranked by our BM25**. That biases the corpus against BM25, so it is not used for eval or serving. |
| `v3.json` | Served and evaluated | 50 per query sampled at random (seed 42) from the official top1000.dev pool; see `params.hard_negative_rule`. |

Rebuild: `python -m scripts.download_msmarco`, then `python -m scripts.prepare_data`,
`python -m scripts.build_indices --only bm25`, `python -m scripts.build_indices --only dense`,
and `python -m scripts.verify_artifacts`.
