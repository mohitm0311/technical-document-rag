from src.llm.llm_generator import LLMGenerator

llm = LLMGenerator()

response = llm.generate_answer(
    "Explain k nearest neighbors in 3 lines."
)

print(response)