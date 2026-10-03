import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))


from documents import load_documents
from embeddings import EmbeddingModel
from search import SemanticSearch

INDEX_DIR = PROJECT_DIR / "data" / "index"
QUESTIONS_FILE = PROJECT_DIR / "evaluation" / "questions.json"


def load_questions() -> list[dict]:
    """Load retrieval evaluation questions."""

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def build_search() -> SemanticSearch:
    """Load the existing search index."""

    embedder = EmbeddingModel()

    search = SemanticSearch(embedder)

    if (INDEX_DIR / "embeddings.npy").exists() and (
        INDEX_DIR / "metadata.npy"
    ).exists():
        search.load(INDEX_DIR)

    else:
        documents = load_documents()
        search.index(documents)
        search.save(INDEX_DIR)

    return search


def reciprocal_rank(
    results,
    expected_source: str,
) -> float:
    """Return the reciprocal rank of the expected source."""

    for position, result in enumerate(results, start=1):
        if result.source == expected_source:
            return 1 / position

    return 0.0


def evaluate(
    search: SemanticSearch,
    questions: list[dict],
    top_k: int,
) -> tuple[float, float]:
    """Calculate retrieval accuracy and MRR."""

    correct = 0
    reciprocal_ranks = []

    for item in questions:
        question = item["question"]
        expected_source = item["expected_source"]

        results = search.search(
            query=question,
            top_k=top_k,
        )

        sources = [result.source for result in results]

        if expected_source in sources:
            correct += 1

        reciprocal_ranks.append(
            reciprocal_rank(
                results,
                expected_source,
            )
        )

    accuracy = correct / len(questions)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return accuracy, mrr


def main():
    questions = load_questions()
    search = build_search()

    print("Retrieval Evaluation")
    print("=" * 60)

    for top_k in [1, 2, 3, 5]:
        accuracy, mrr = evaluate(
            search=search,
            questions=questions,
            top_k=top_k,
        )

        print(
            f"top_k={top_k}: "
            f"accuracy={accuracy:.1%}, "
            f"MRR={mrr:.3f}"
        )


if __name__ == "__main__":
    main()
