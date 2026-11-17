"""Download the pinned Hugging Face models into the local HF cache.

Used at Docker build time so the image runs with HF_HUB_OFFLINE=1.
Model names and commit SHAs come from app.config (single source of truth).
"""

from huggingface_hub import snapshot_download

from app.config import settings


def main() -> None:
    for repo_id, revision in (
        (settings.embedding_model, settings.embedding_model_revision),
        (settings.reranker_model, settings.reranker_model_revision),
    ):
        path = snapshot_download(repo_id=repo_id, revision=revision)
        print(f"{repo_id}@{revision} -> {path}")


if __name__ == "__main__":
    main()
