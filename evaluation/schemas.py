from pydantic import BaseModel, Field


class AnswerEvaluation(BaseModel):
    """Evaluation of a generated RAG answer."""

    grounded: bool = Field(
        description=("Whether the answer is supported by the provided context.")
    )

    relevant: bool = Field(
        description=("Whether the answer directly addresses the user's question.")
    )

    correct: bool = Field(
        description=(
            "Whether the answer is factually correct according to the provided context."
        )
    )

    score: int = Field(
        ge=1,
        le=5,
        description=("Overall answer quality from 1 to 5."),
    )

    reasoning: str = Field(description=("Brief explanation of the evaluation."))
