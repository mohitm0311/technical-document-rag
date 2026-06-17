"""
Run this from the root of your technical-document-rag repo:
    python benchmark.py

It reuses your own project's modules, so the numbers it prints are
real measurements from your actual pipeline -- not estimates.
Requires: your normal venv + `ollama serve` running in another terminal
with `qwen3:4b` already pulled (same setup as your README).
"""
import sys, time, json
sys.path.insert(0, ".")

from src.ingestion.pdf_loader import PDFLoader
from src.chunking.sentence_chunker import SentenceChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.faiss_indexer import FAISSIndexer
from src.retrieval.retriever import Retriever
from src.llm.prompt_builder import PromptBuilder   
from src.llm.llm_generator import LLMGenerator     # adjust names if yours differ

PDF_PATH = "data/pdfs/Introduction_to_Machine_Learning.pdf"  # or your own PDF

results = {}

t0 = time.perf_counter()
text = PDFLoader(PDF_PATH).extract_text()
results["pdf_extract_seconds"] = round(time.perf_counter() - t0, 4)
results["extracted_chars"] = len(text)

t0 = time.perf_counter()
chunks = SentenceChunker(max_chunk_size=1000, overlap_sentences=1).chunk_text(text)
results["chunking_seconds"] = round(time.perf_counter() - t0, 4)
results["num_chunks"] = len(chunks)

t0 = time.perf_counter()
embedder = EmbeddingGenerator()
embeddings = embedder.generate_embeddings(chunks)
results["embedding_seconds_total"] = round(time.perf_counter() - t0, 4)
results["embedding_dimension"] = int(embeddings.shape[1])

t0 = time.perf_counter()
indexer = FAISSIndexer()
index = indexer.build_index(embeddings)
results["faiss_build_seconds"] = round(time.perf_counter() - t0, 4)
results["faiss_index_size"] = index.ntotal

retriever = Retriever(embedder, indexer, chunks)
test_queries = [
    "What is K-Means Clustering?",
    "Explain how Gradient Boosting works.",
    "What are the assumptions of KNN?",
    "Summarize the chapter on Neural Networks.",
    "Compare Random Forest and Gradient Boosting.",
]

retrieval_times, gen_times, top_scores = [], [], []
for q in test_queries:
    t0 = time.perf_counter()
    retrieved = retriever.retrieve(q, k=5)
    retrieval_times.append(time.perf_counter() - t0)
    top_scores.append(retrieved[0]["score"])

    # --- Full RAG round trip including the LLM (adjust to your actual class/method names) ---
    # t0 = time.perf_counter()
    # prompt = PromptBuilder().build(q, retrieved)
    # answer = LLMGenerator().generate(prompt)
    # gen_times.append(time.perf_counter() - t0)

results["avg_retrieval_seconds"] = round(sum(retrieval_times) / len(retrieval_times), 5)
results["avg_top1_similarity_score"] = round(sum(top_scores) / len(top_scores), 4)
# results["avg_full_rag_seconds"] = round(sum(gen_times) / len(gen_times), 3)

print(json.dumps(results, indent=2))
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
