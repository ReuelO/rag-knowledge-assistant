from dataclasses import dataclass


@dataclass
class RAGResponse:
    answer: str
    sources: list
    evidence_sufficient: bool
    evidence_score: float


class RAGPipeline:
    """Retrieval-augmented generation pipeline."""

    def __init__(
        self,
        search,
        llm,
        evidence_gate,
    ):
        self.search = search
        self.llm = llm
        self.evidence_gate = evidence_gate

    def answer(
        self,
        question: str,
        top_k: int = 3,
    ) -> RAGResponse:
        """Answer only when sufficient evidence exists."""

        results = self.search.search(
            query=question,
            top_k=top_k,
        )

        decision = self.evidence_gate.evaluate(results)

        if not decision.sufficient:
            return RAGResponse(
                answer=(
                    "I don't have enough information "
                    "in my knowledge base to answer "
                    "that question reliably."
                ),
                sources=results,
                evidence_sufficient=False,
                evidence_score=decision.score,
            )

        context = "\n\n".join(
            (f"Source: {result.source}\n{result.content}") for result in results
        )

        answer = self.llm.answer(
            question=question,
            context=context,
        )

        return RAGResponse(
            answer=answer,
            sources=results,
            evidence_sufficient=True,
            evidence_score=decision.score,
        )
