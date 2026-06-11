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
        query_embedding = self.embedding_generator.generate_embeddings(
            [query]
        )

        distances, indices = self.indexer.search(
            query_embedding,
            k
        )

        retrieved_chunks = []

        for idx in indices[0]:
            retrieved_chunks.append(
                self.chunks[idx]
            )

        return retrieved_chunks