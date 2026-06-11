class PromptBuilder:

    @staticmethod
    def build_prompt(
        query: str,
        retrieved_chunks: list
    ) -> str:

        context = "\n\n".join(
            chunk["text"]
            for chunk in retrieved_chunks
        )

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{query}

Answer:
"""

        return prompt