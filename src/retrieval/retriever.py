class Retriever:
    def __init__(
        self,
        embedding_generator,
        indexer,
        chunks
    ):
        self.embedding_generator = embedding_generator
        self.indexer = indexer
        self.chunks = chunks

    def retrieve(
        self,
        query: str,
        k: int = 5
    ):
        query_embedding = (
            self.embedding_generator
            .generate_embeddings([query])
        )

        distances, indices = (
            self.indexer.search(
                query_embedding,
                k
            )
        )

        results = []

        for score, idx in zip(
            distances[0],
            indices[0]
        ):
            results.append(
                {
                    "chunk_id": int(idx),
                    "score": float(score),
                    "text": self.chunks[idx]
                }
            )

        return results