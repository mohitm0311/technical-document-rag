from src.ingestion.pdf_loader import PDFLoader
from src.chunking.sentence_chunker import SentenceChunker
from src.embeddings.embedding_generator import EmbeddingGenerator


loader = PDFLoader(
    "data/pdfs/Introduction_to_Machine_Learning.pdf"
)

text = loader.extract_text()

chunker = SentenceChunker()

chunks = chunker.chunk_text(text)

generator = EmbeddingGenerator()

embeddings = generator.generate_embeddings(chunks)

print(embeddings.shape)