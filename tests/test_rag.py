from src.ingestion.pdf_loader import PDFLoader
from src.chunking.sentence_chunker import SentenceChunker
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.faiss_indexer import FAISSIndexer
from src.retrieval.retriever import Retriever
from src.rag.rag_pipeline import RAGPipeline


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
pipeline = RAGPipeline(
    retriever
)
result = pipeline.ask(
    "What is k nearest neighbors?"
)

print("\nAnswer:")
print("=" * 50)
print(result["answer"])

print("\nSources:")
print("=" * 50)

for source in result["sources"]:

    print(
        f"\nChunk ID: {source['chunk_id']}"
    )

    print(
        f"Score: {source['score']:.4f}"
    )

    print("-" * 50)

    print(
        source["text"][:300]
    )