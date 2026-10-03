import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Configuration
PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_DIR.parents[1]

sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(PROJECT_ROOT / ".env")


from documents import load_documents
from embeddings import EmbeddingModel
from evaluation.judge import RAGJudge
from evidence import EvidenceGate
from llm import RAGLLM
from rag import RAGPipeline
from search import SemanticSearch

INDEX_DIR = PROJECT_DIR / "data" / "index"
QUESTIONS_FILE = PROJECT_DIR / "evaluation" / "questions.json"
RESULTS_DIR = PROJECT_DIR / "evaluation" / "results"
RESULTS_FILE = RESULTS_DIR / "latest.json"
EVIDENCE_THRESHOLD = 0.50
TOP_K = 3


# Data Loading
def load_questions() -> list[dict]:
    """Load evaluation questions from JSON."""

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# Search
def build_search() -> SemanticSearch:
    """Load an existing search index or build one."""

    embedder = EmbeddingModel()

    search = SemanticSearch(embedder)

    embeddings_file = INDEX_DIR / "embeddings.npy"
    metadata_file = INDEX_DIR / "metadata.npy"

    if embeddings_file.exists() and metadata_file.exists():
        search.load(INDEX_DIR)

    else:
        documents = load_documents()

        search.index(documents)
        search.save(INDEX_DIR)

    return search


# RAG
def build_rag() -> RAGPipeline:
    """Build the RAG pipeline."""

    search = build_search()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    llm = RAGLLM(
        api_key=api_key,
        model=model,
    )

    evidence_gate = EvidenceGate(threshold=EVIDENCE_THRESHOLD)

    return RAGPipeline(
        search=search,
        llm=llm,
        evidence_gate=evidence_gate,
    )


# Context
def build_context(results) -> str:
    """Build context from retrieved results."""

    parts = []

    parts.extend(
        f"Source: {result.source}\nContent: {result.content}" for result in results
    )
    return "\n\n".join(parts)


# Main Evaluation
def main():
    questions = load_questions()

    rag = build_rag()

    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    judge = RAGJudge(
        api_key=api_key,
        model=model,
    )

    # Overall metrics
    total = 0

    # Answerable questions
    answerable_total = 0
    answerable_passed_gate = 0
    answerable_correct = 0
    answerable_grounded = 0
    answerable_relevant = 0
    answerable_score_total = 0

    # Unanswerable questions
    unanswerable_total = 0
    correctly_refused = 0

    # Detailed results
    results = []

    print("RAG Evaluation")
    print("=" * 60)

    print(f"Evidence threshold: {EVIDENCE_THRESHOLD:.2f}")

    print(f"Top-k: {TOP_K}")

    # Evaluate each question
    for number, item in enumerate(
        questions,
        start=1,
    ):
        question = item["question"]

        expected_answerable = item.get(
            "answerable",
            True,
        )

        total += 1

        print(f"\n[{number}] {question}")

        # Run RAG
        response = rag.answer(
            question=question,
            top_k=TOP_K,
        )

        print(f"Evidence score: {response.evidence_score:.3f}")

        print(f"Evidence sufficient: {response.evidence_sufficient}")

        # Handle unanswerable questions
        if not expected_answerable:
            unanswerable_total += 1

            refused = not response.evidence_sufficient

            if refused:
                correctly_refused += 1

            print("Expected refusal: True")

            print(f"Correctly refused: {refused}")

            results.append(
                {
                    "question": question,
                    "expected_answerable": False,
                    "answer": response.answer,
                    "sources": [source.source for source in response.sources],
                    "evidence_score": (response.evidence_score),
                    "evidence_sufficient": (response.evidence_sufficient),
                    "correctly_refused": refused,
                }
            )

            continue

        # Handle answerable questions
        answerable_total += 1

        if response.evidence_sufficient:
            answerable_passed_gate += 1

        # If the gate incorrectly rejects an answerable question,
        # there is no answer to send to the judge.
        if not response.evidence_sufficient:
            print("\nFAIL: Answerable question was rejected by the evidence gate.")

            results.append(
                {
                    "question": question,
                    "expected_answerable": True,
                    "answer": response.answer,
                    "sources": [source.source for source in response.sources],
                    "evidence_score": (response.evidence_score),
                    "evidence_sufficient": False,
                    "evaluation": None,
                }
            )

            continue

        # Build context
        context = build_context(response.sources)

        # Judge answer
        evaluation = judge.evaluate(
            question=question,
            context=context,
            answer=response.answer,
        )

        if evaluation is None:
            print("Judge evaluation failed.")

            results.append(
                {
                    "question": question,
                    "expected_answerable": True,
                    "answer": response.answer,
                    "sources": [source.source for source in response.sources],
                    "evidence_score": (response.evidence_score),
                    "evidence_sufficient": True,
                    "evaluation": None,
                }
            )

            continue

        # Update answer metrics
        if evaluation.correct:
            answerable_correct += 1

        if evaluation.grounded:
            answerable_grounded += 1

        if evaluation.relevant:
            answerable_relevant += 1

        answerable_score_total += evaluation.score

        # Display answer evaluation
        print(f"\nAnswer:\n{response.answer}")

        print("\nSources:")

        for source in response.sources:
            print(f"- {source.source} (score: {source.score:.3f})")

        print("\nEvaluation:")
        print(f"Grounded: {evaluation.grounded}")
        print(f"Relevant: {evaluation.relevant}")
        print(f"Correct: {evaluation.correct}")
        print(f"Score: {evaluation.score}/5")
        print(f"Reasoning: {evaluation.reasoning}")

        # Store result
        results.append(
            {
                "question": question,
                "expected_answerable": True,
                "answer": response.answer,
                "sources": [source.source for source in response.sources],
                "evidence_score": (response.evidence_score),
                "evidence_sufficient": (response.evidence_sufficient),
                "evaluation": {
                    "grounded": (evaluation.grounded),
                    "relevant": (evaluation.relevant),
                    "correct": (evaluation.correct),
                    "score": (evaluation.score),
                    "reasoning": (evaluation.reasoning),
                },
            }
        )

    # Summary
    print("\n")
    print("=" * 60)
    print("RAG Evaluation Summary")
    print("=" * 60)

    print(f"Total questions: {total}")

    # Answerable summary
    print("\nAnswerable questions:")

    print(f"  Total: {answerable_total}")

    if answerable_total > 0:
        print(
            f"  Passed evidence gate: "
            f"{answerable_passed_gate}/"
            f"{answerable_total} "
            f"({answerable_passed_gate / answerable_total:.1%})"
        )

        print(
            f"  Correct: "
            f"{answerable_correct}/"
            f"{answerable_total} "
            f"({answerable_correct / answerable_total:.1%})"
        )

        print(
            f"  Grounded: "
            f"{answerable_grounded}/"
            f"{answerable_total} "
            f"({answerable_grounded / answerable_total:.1%})"
        )

        print(
            f"  Relevant: "
            f"{answerable_relevant}/"
            f"{answerable_total} "
            f"({answerable_relevant / answerable_total:.1%})"
        )

        print(f"  Average score: {answerable_score_total / answerable_total:.2f}/5")

    # Unanswerable summary
    print("\nUnanswerable questions:")

    print(f"  Total: {unanswerable_total}")

    if unanswerable_total > 0:
        print(
            f"  Correctly refused: "
            f"{correctly_refused}/"
            f"{unanswerable_total} "
            f"({correctly_refused / unanswerable_total:.1%})"
        )

    # Save Results
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nDetailed results saved to:\n{RESULTS_FILE}")


if __name__ == "__main__":
    main()
