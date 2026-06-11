from src.ingestion.pdf_loader import PDFLoader
from src.chunking.sentence_chunker import SentenceChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.faiss_indexer import FAISSIndexer
from src.retrieval.retriever import Retriever


loader = PDFLoader(
    "data/pdfs/Introduction_to_Machine_Learning.pdf"
)

text = loader.extract_text()

chunker = SentenceChunker()

chunks = chunker.chunk_text(text)

generator = EmbeddingGenerator()

embeddings = generator.generate_embeddings(chunks)

indexer = FAISSIndexer()

indexer.build_index(embeddings)

retriever = Retriever(
    generator,
    indexer,
    chunks
)

query = "What is k nearest neighbors?"

results = retriever.retrieve(
    query,
    k=5
)

for i, chunk in enumerate(results):
    print(f"\nResult {i+1}")
    print("-" * 50)
    print(chunk[:500])