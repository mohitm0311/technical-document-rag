from src.ingestion.pdf_loader import PDFLoader
from src.chunking.sentence_chunker import SentenceChunker

loader = PDFLoader("data/pdfs/Introduction_to_Machine_Learning.pdf")

text = loader.extract_text()

chunker = SentenceChunker(
    max_chunk_size=1000,
    overlap_sentences=1
)

chunks = chunker.chunk_text(text)

print(f"Total Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0])

print("\nFirst Chunk Length:")
print(len(chunks[0]))