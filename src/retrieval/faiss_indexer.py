import faiss
import numpy as np


class FAISSIndexer:
    def __init__(self):
        self.index = None

    def build_index(self, embeddings: np.ndarray):

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

        return self.index
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)

        return distances, indices