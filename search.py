from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    content: str
    source: str
    score: float


class SemanticSearch:
    """Semantic search over document chunks."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.embeddings = None
        self.metadata = []

    def index(self, documents):
        """Create an embedding index from documents."""

        texts = [document.content for document in documents]

        self.embeddings = self.embedder.embed(texts)

        self.metadata = [
            {
                "content": document.content,
                "source": document.source,
            }
            for document in documents
        ]

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Return the most similar document chunks."""

        if self.embeddings is None:
            raise RuntimeError("Search index has not been built.")

        query_embedding = self.embedder.embed([query])[0]

        scores = self.embeddings @ query_embedding

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            metadata = self.metadata[index]

            results.append(
                SearchResult(
                    content=metadata["content"],
                    source=metadata["source"],
                    score=float(scores[index]),
                )
            )

        return results

    def save(self, directory: Path):
        """Save the search index to disk."""

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        embeddings_file = directory / "embeddings.npy"

        metadata_file = directory / "metadata.npy"

        np.save(
            embeddings_file,
            self.embeddings,
        )

        np.save(
            metadata_file,
            np.array(
                self.metadata,
                dtype=object,
            ),
        )

    def load(self, directory: Path):
        """Load a previously saved search index."""

        embeddings_file = directory / "embeddings.npy"

        metadata_file = directory / "metadata.npy"

        if not embeddings_file.exists():
            raise FileNotFoundError(f"Missing embeddings file: {embeddings_file}")

        if not metadata_file.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_file}")

        self.embeddings = np.load(embeddings_file)

        self.metadata = np.load(
            metadata_file,
            allow_pickle=True,
        ).tolist()
