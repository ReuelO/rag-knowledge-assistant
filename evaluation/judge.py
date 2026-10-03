import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_DIR))


from openrouter import OpenRouter
from schemas import AnswerEvaluation


class RAGJudge:
    """Evaluate RAG answers using an LLM."""

    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        self.model = model
        self.client = OpenRouter(
            api_key=api_key,
        )

    def evaluate(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> AnswerEvaluation | None:
        """Evaluate an answer against its context."""

        system_prompt = """
You are an evaluator for a retrieval-augmented
generation system.

Evaluate the generated answer using ONLY the
provided context.

Do not use your own outside knowledge.

Determine:

1. Whether the answer is supported by the context.
2. Whether the answer addresses the question.
3. Whether the answer is factually correct according
   to the context.
4. An overall quality score from 1 to 5.

Return only the requested structured result.
"""

        user_prompt = f"""
Question:

{question}

Retrieved context:

{context}

Generated answer:

{answer}
"""

        try:
            response = self.client.chat.send(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer_evaluation",
                        "strict": True,
                        "schema": (AnswerEvaluation.model_json_schema()),
                    },
                },
            )

            raw_output = response.choices[0].message.content

            return AnswerEvaluation.model_validate_json(raw_output)

        except Exception as error:  # noqa: BLE001
            print("\nJudge request failed.")
            print(f"Error type: {type(error).__name__}")
            print(f"Error: {error}")

            return None
