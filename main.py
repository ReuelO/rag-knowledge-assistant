import os
from pathlib import Path

from documents import load_documents
from dotenv import load_dotenv
from embeddings import EmbeddingModel
from evidence import EvidenceGate
from llm import RAGLLM
from rag import RAGPipeline
from search import SemanticSearch

# Configuration
PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parents[1]

load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free",
)

if not api_key:
    raise ValueError("OPENROUTER_API_KEY is not set.")


# Build Search
embedder = EmbeddingModel()

search = SemanticSearch(embedder)

index_dir = PROJECT_DIR / "data" / "index"

embeddings_file = index_dir / "embeddings.npy"
metadata_file = index_dir / "metadata.npy"

if embeddings_file.exists() and metadata_file.exists():
    print("Loading existing search index...\n")
    search.load(index_dir)

else:
    print("Building search index...\n")

    documents = load_documents()

    search.index(documents)
    search.save(index_dir)


# Build LLM
llm = RAGLLM(
    api_key=api_key,
    model=model,
)


# Evidence Gate
EVIDENCE_THRESHOLD = 0.50

evidence_gate = EvidenceGate(threshold=EVIDENCE_THRESHOLD)


# Build RAG Pipeline
rag = RAGPipeline(
    search=search,
    llm=llm,
    evidence_gate=evidence_gate,
)


# Application
print("RAG Knowledge Assistant")
print(f"Evidence threshold: {EVIDENCE_THRESHOLD:.2f}")
print("Type /exit to quit.\n")


while True:
    question = input("You: ").strip()

    if not question:
        continue

    if question.lower() == "/exit":
        print("Goodbye!")
        break

    response = rag.answer(
        question=question,
        top_k=3,
    )

    print(f"\nEvidence score: {response.evidence_score:.3f}")

    # Insufficient evidence
    if not response.evidence_sufficient:
        print("\nInsufficient evidence.")
        print(
            "I don't have enough information "
            "in my knowledge base to answer "
            "that question reliably."
        )
        print()

        continue

    # Answer
    print(f"\nAssistant:\n{response.answer}")

    # Sources
    print("\nSources:")

    seen_sources = set()

    for source in response.sources:
        if source.source in seen_sources:
            continue

        print(f"- {source.source} (score: {source.score:.3f})")

        seen_sources.add(source.source)

    print()
