class PromptBuilder:

    @staticmethod
    def build_prompt(
        query: str,
        retrieved_chunks: list[str]
    ) -> str:

        context = "\n\n".join(
            retrieved_chunks
        )

        prompt = f"""
You are a helpful AI assistant.

Answer the question using only the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

        return prompt