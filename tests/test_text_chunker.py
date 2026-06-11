from src.ingestion.pdf_loader import PDFLoader
from src.chunking.text_chunker import TextChunker

loader = PDFLoader("data/pdfs/Introduction_to_Machine_Learning.pdf")

text = loader.extract_text()

chunker = TextChunker()

chunks = chunker.chunk_text(text)

print(f"Total Chunks: {len(chunks)}")
print()
print(chunks[0])
