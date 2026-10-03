from openrouter import OpenRouter


class RAGLLM:
    """Generate answers using an LLM."""

    def __init__(self, api_key: str, model: str):
        self.model = model
        self.client = OpenRouter(api_key=api_key)

    def generate(
        self,
        question: str,
        context: str,
    ) -> str | None:
        """Generate an answer using retrieved context."""

        system_prompt = (
            "You are a helpful knowledge assistant. "
            "Answer the user's question using the provided context. "
            "Do not invent information that is not supported by the context. "
            "If the context does not contain enough information to answer "
            "the question, say that the available knowledge does not "
            "contain enough information."
        )

        user_prompt = f"""
Context:

{context}

Question:

{question}

Answer the question using the context above.
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
            )

            return response.choices[0].message.content

        except Exception as error:  # noqa: BLE001
            print("\nLLM request failed.")
            print(f"Error type: {type(error).__name__}")
            print(f"Error: {error}")
            return None
