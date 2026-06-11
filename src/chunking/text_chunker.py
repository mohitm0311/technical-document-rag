class TextChunker:
    def __init__(self,
        chunk_size: int = 1000,
        overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        chunks = []

        start = 0

        while start < len(text):
            end = start + self.chunk_size

            chunk = text[start:end]

            chunks.append(chunk)

            start += (self.chunk_size - self.overlap)

        return chunks