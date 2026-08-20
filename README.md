# Hybrid RAG over MS MARCO

A Retrieval-Augmented Generation (RAG) system built with a production-style architecture: dense semantic retrieval, BM25 lexical retrieval, Reciprocal Rank Fusion, cross-encoder reranking, and grounded LLM generation — served over a REST API.

Instead of sending a user query directly to an LLM, the system retrieves and ranks the most relevant passages from a 350,000-passage MS MARCO subset first, then generates an answer grounded strictly in that retrieved context.

---

## Overview

Traditional LLM applications can generate plausible-sounding answers without access to a domain-specific knowledge source. This project implements a retrieval-first architecture instead:

1. Receive a natural-language query.
2. Retrieve candidates using dense semantic search.
3. Retrieve candidates independently using BM25 lexical search.
4. Fuse both ranked lists using Reciprocal Rank Fusion.
5. Rerank the fused candidates with a cross-encoder.
6. Select the top 5 passages.
7. Pass only those passages to the LLM as context.
8. Generate a grounded answer.
9. Return the answer together with its supporting source passages.

Separating **retrieval**, **relevance ranking**, and **generation** into distinct stages makes the system easier to evaluate and debug than a single end-to-end LLM call.

---

## Why hybrid retrieval

Dense retrieval and lexical retrieval fail in different ways:

- **Dense retrieval** (embeddings + FAISS) is strong at semantic matching — it can find a relevant passage even when the query and passage don't share exact words.
- **BM25** (lexical/keyword search) is strong when exact terminology matters — dense retrieval can miss it if the wording differs from the training distribution.

Running both and fusing the results with **Reciprocal Rank Fusion (RRF)** gets the benefit of each without needing to normalize or compare their raw scores directly. A **cross-encoder reranker** then re-scores the fused candidates more precisely before the top passages go to the LLM.

---

## Architecture

```
                    USER
                      |
                      v
             +----------------+
             |   FastAPI API  |
             +-------+--------+
                     |
                     v
              User Query
                     |
          +----------+----------+
          |                     |
          v                     v
 +----------------+     +----------------+
 | Dense Retrieval|     | BM25 Retrieval |
 | Sentence       |     | Lexical Search |
 | Transformers   |     |                |
 | + FAISS        |     | (Rank-BM25)    |
 +-------+--------+     +-------+--------+
         |                      |
         +----------+-----------+
                    |
                    v
          +---------------------+
          | Reciprocal Rank     |
          | Fusion (RRF, k=60)  |
          +----------+----------+
                     |
                     v
             Hybrid Candidates (10)
                     |
                     v
          +---------------------+
          | Cross-Encoder       |
          | Reranking           |
          | ms-marco-MiniLM-L-6 |
          +----------+----------+
                     |
                     v
               Top 5 Passages
                     |
                     v
          +---------------------+
          | Groq-hosted LLM     |
          | Grounded Generation |
          +----------+----------+
                     |
                     v
          Answer + Source Passages
```

---

## Tech stack

| Layer | Tools |
|---|---|
| API | FastAPI, Pydantic, Pydantic Settings |
| Dense retrieval | Sentence Transformers (`all-MiniLM-L6-v2`), FAISS |
| Lexical retrieval | Rank-BM25 |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Generation | Groq API (`openai/gpt-oss-120b`) |
| Data | NumPy, Pandas, PyArrow |
| Testing | Pytest |
| Deployment | Docker, Docker Compose |

---

## Dataset

- **Source:** MS MARCO passage subset
- **Scale:** 350,000 passages
- **Embeddings:** 384-dimensional vectors via `all-MiniLM-L6-v2`, normalized and indexed in FAISS
- **Embedding artifact size:** ~512.7 MB

---

## Retrieval configuration

| Parameter | Value |
|---|---|
| Dense/BM25 retrieval K | 20 |
| Hybrid candidates after RRF | 10 |
| RRF k | 60 |
| Final reranked passages | 5 |

---

## Retrieval quality — evaluation results

Retrieval quality was measured, not assumed. A local evaluation set was built with 20 queries and 63 manually reviewed relevance judgments (`eval/qrels.json`).

| Metric | Score |
|---|---|
| Recall@5 | 1.0000 |
| MRR | 0.9167 |
| nDCG@5 | 0.9389 |

> **Scope note:** these results are from a 20-query, manually judged local evaluation set — not an official MS MARCO benchmark run. Always describe them as *"100% Recall@5, 0.9167 MRR, and 0.9389 nDCG@5 on a 20-query manually judged evaluation set with 63 relevance judgments"* — never as an unqualified MS MARCO benchmark score.

Evaluation pipeline:
```
tests/generate_eval_candidates.py   # generate candidate passages per query
tests/build_candidate_review.py     # build review sheet for manual judging
tests/build_qrels.py                # compile relevance judgments
tests/evaluate_retrieval.py         # compute Recall@5, MRR, nDCG@5
```

---

## API

### Health check
```
GET /health
```
```json
{
  "status": "healthy",
  "service": "hybrid-rag"
}
```

### Query
```
POST /api/v1/query
```
Request:
```json
{
  "query": "what are the symptoms of diabetes?"
}
```
Response:
```json
{
  "query": "what are the symptoms of diabetes?",
  "answer": "...",
  "sources": [
    {
      "passage_id": 24998,
      "passage_text": "...",
      "score": 9.091580
    }
  ]
}
```

Empty or missing queries are rejected with `HTTP 422` (Pydantic validation — `query` must be at least one character).

---

## Project structure

```
hybrid-rag/
├── app/
│   ├── api/routes.py
│   ├── generation/groq_generator.py
│   ├── reranking/cross_encoder_reranker.py
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   └── hybrid_retriever.py
│   ├── schemas/query.py
│   ├── config.py
│   ├── main.py
│   └── rag_service.py
├── data/passages_subset.parquet
├── embeddings/all_embeddings.npy
├── indices/
│   ├── dense_index.faiss
│   └── bm25/{bm25_index.pkl, tokenized_corpus.pkl}
├── eval/{qrels.json, queries.json, candidate_review.json, ...}
├── tests/
│   ├── test_api.py
│   ├── test_bm25_retriever_manual.py
│   ├── test_dense_retriever_manual.py
│   ├── test_hybrid_retriever_manual.py
│   ├── test_reranker_manual.py
│   ├── test_generation_manual.py
│   ├── test_rag_service_manual.py
│   └── evaluate_retrieval.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/manojkumargudge/hybrid-rag-ms-marco.git
cd hybrid-rag-ms-marco

# 2. Create a virtual environment
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
# Edit .env and set:
#   GROQ_API_KEY=your_groq_api_key_here
#   GROQ_MODEL=openai/gpt-oss-120b

# 5. Run the API
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

> Adjust the run command if your actual entry point differs from `app.main:app`.

### Run with Docker

```bash
docker compose up --build
```
Exposes the service on port `8000`. Note: the image build is relatively heavy (~1.4 GB of retrieval artifacts + ML/NLP dependencies).

---

## Testing

```bash
pytest
```
Latest local run: **7 passed**, 1 warning (Starlette/httpx deprecation notice — does not affect test outcome).

Test coverage includes: health endpoint, query endpoint, empty-query validation (422), response structure, source validation, and duplicate-source checking.

---

## Configuration

All configuration is centralized in `app/config.py` and loaded via Pydantic Settings from `.env`:

```
GROQ_API_KEY
GROQ_MODEL
PASSAGES_PATH
DENSE_INDEX_PATH
BM25_INDEX_PATH
BM25_CORPUS_PATH
RETRIEVAL_K
HYBRID_TOP_K
RERANK_TOP_K
RRF_K
```

Secrets are excluded from version control via `.gitignore`; `.env.example` documents the required variables without real values.

---

## Known limitations

Being upfront about this matters more than it hurts:

- Evaluation set is small (20 queries, 63 judgments) and self-labeled — not an independently audited benchmark.
- No CI/CD pipeline configured yet.
- No load testing or latency benchmarking under concurrent requests.
- Docker image is untrimmed (~1.4 GB) — no multi-stage build optimization yet.

## Possible next steps

- Expand the evaluation set and/or run against an official MS MARCO benchmark split for a comparable score.
- Add CI (GitHub Actions) to run `pytest` and lint on every push.
- Add latency/throughput benchmarks for the full pipeline (retrieval + rerank + generation).
- Multi-stage Docker build to shrink image size.

---

