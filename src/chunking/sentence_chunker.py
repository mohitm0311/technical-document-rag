import re

class SentenceChunker:
    def __init__(self, max_chunk_size: int = 1000, overlap_sentences: int = 1):
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk_text(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.max_chunk_size:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                overlap = current_chunk[-self.overlap_sentences:]
                current_chunk = overlap.copy()
                current_length = sum(len(s) for s in current_chunk)
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks