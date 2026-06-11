import faiss
import numpy as np


class FAISSIndexer:
    def __init__(self):
        self.index = None

    def build_index(self, embeddings: np.ndarray):

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        return self.index
    
    def search(self, query_embedding: np.ndarray, k: int = 5):
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)

        return distances, indices