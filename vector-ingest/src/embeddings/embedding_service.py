from typing import List
from sentence_transformers import SentenceTransformer

from ..chunking.models import Chunk


class EmbeddingService:
    """Generate embeddings using BAAI/bge-small-en-v1.5."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        
        return chunks