# Hybrid RAG over MS MARCO

A production-oriented Retrieval-Augmented Generation (RAG) system that combines dense semantic retrieval, lexical BM25 retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and grounded LLM generation.

The system retrieves relevant passages from an MS MARCO passage subset, combines semantic and lexical search results, reranks the strongest candidates using a cross-encoder, and generates an answer using only the retrieved context.

---

## Architecture

```text
                         User Query
                             |
                             v
                  +----------------------+
                  |    FastAPI REST API  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   Hybrid Retrieval   |
                  +----------+-----------+
                             |
                  +----------+----------+
                  |                     |
                  v                     v
          +---------------+     +---------------+
          | Dense Search  |     |   BM25 Search |
          |    FAISS      |     |   Lexical     |
          +-------+-------+     +-------+-------+
                  |                     |
                  +----------+----------+
                             |
                             v
                  +----------------------+
                  |    RRF Fusion         |
                  | Reciprocal Rank      |
                  | Fusion (k=60)        |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Cross-Encoder        |
                  | Reranking             |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Groq-hosted LLM       |
                  | Grounded Generation   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Answer + Sources      |
                  +----------------------+
