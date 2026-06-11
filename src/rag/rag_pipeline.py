from src.retrieval.retriever import Retriever
from src.llm.prompt_builder import PromptBuilder
from src.llm.llm_generator import LLMGenerator


class RAGPipeline:

    def __init__(
        self,
        retriever: Retriever
    ):
        self.retriever = retriever
        self.llm = LLMGenerator()

    def ask(
        self,
        query: str,
        k: int = 5
    ):

        retrieved_chunks = self.retriever.retrieve(
            query,
            k
        )

        prompt = PromptBuilder.build_prompt(
            query,
            retrieved_chunks
        )

        answer = self.llm.generate_answer(
            prompt
        )

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }