"""Milvus document loader for Ragas test generation."""

import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

from langchain.docstore.document import Document
from tqdm import tqdm

# Add vector-ingest to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vector-ingest" / "src"))

from embeddings.milvus_config import MilvusConfig
from embeddings.milvus_store import MilvusVectorStore

logger = logging.getLogger(__name__)


class MilvusDocumentLoader:
    """Loads documents from Milvus and converts them to LangChain Document format."""
    
    def __init__(self, config: MilvusConfig = None):
        """Initialize Milvus connection."""
        self.config = config or MilvusConfig.default()
        self.store = MilvusVectorStore(config=self.config)
        
        # Connect to Milvus
        if not self.store.connect():
            raise ConnectionError(f"Failed to connect to Milvus at {self.config.host}:{self.config.port}")
        
        # Load collection
        if not self.store.load_collection():
            raise RuntimeError(f"Failed to load collection: {self.config.collection_name}")
        
        logger.info(f"Connected to Milvus collection: {self.config.collection_name}")
    
    def get_document_count(self) -> int:
        """Get total document count in collection."""
        try:
            if self.store.collection:
                self.store.collection.flush()
                return self.store.collection.num_entities
            return 0
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    def load_documents(
        self,
        max_documents: int = None,
        sample_strategy: str = "random"
    ) -> List[Document]:
        """
        Load documents from Milvus with specified sampling strategy.
        
        Time complexity: O(n) for sequential, O(n log n) for random sampling
        """
        total_docs = self.get_document_count()
        logger.info(f"Found {total_docs} documents in collection '{self.config.collection_name}'")
        
        if not total_docs:
            return []
        
        fetch_size = min(max_documents, total_docs) if max_documents else total_docs
        
        if sample_strategy == "random" and max_documents and max_documents < total_docs:
            return self._load_random_sample(fetch_size, total_docs)
        return self._load_sequential(fetch_size)
    
    def _load_sequential(self, max_docs: int) -> List[Document]:
        """Load documents sequentially. O(n) time complexity."""
        documents = []
        
        try:
            # Query Milvus collection
            # Get all fields except vector for efficiency
            output_fields = ["id", "text", "metadata", "chunk_id", "source", "token_count"]
            
            # Use query to get documents sequentially
            expr = "id >= 0"
            
            results = self.store.collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=min(max_docs, 16384)  # Milvus query limit
            )
            
            logger.info(f"Fetched {len(results)} documents from Milvus")
            
            for result in tqdm(results[:max_docs], desc="Converting documents"):
                doc = self._convert_to_document(result)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}", exc_info=True)
            return []
    
    def _load_random_sample(self, sample_size: int, total_docs: int) -> List[Document]:
        """Load random sample. O(n log n) for sorting."""
        try:
            # Generate random IDs to sample
            # Assume IDs are sequential starting from 0
            all_ids = list(range(total_docs))
            random.shuffle(all_ids)
            sampled_ids = all_ids[:sample_size]
            
            # Query by IDs
            expr = f"id in {sampled_ids}"
            output_fields = ["id", "text", "metadata", "chunk_id", "source", "token_count"]
            
            results = self.store.collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=sample_size
            )
            
            logger.info(f"Fetched {len(results)} random documents")
            
            documents = [
                self._convert_to_document(result)
                for result in tqdm(results, desc="Converting documents")
            ]
            
            return documents[:sample_size]
            
        except Exception as e:
            logger.error(f"Error in random sampling: {e}", exc_info=True)
            return []
    
    def _convert_to_document(self, result: Dict) -> Document:
        """
        Convert Milvus result to LangChain Document. O(1) operation.
        
        Args:
            result: Milvus query result dictionary
        
        Returns:
            LangChain Document object
        """
        # Extract text content
        text = result.get("text", "")
        
        # Build metadata efficiently
        metadata = {
            "id": result.get("id"),
            "chunk_id": result.get("chunk_id"),
            "source": result.get("source"),
            "token_count": result.get("token_count"),
        }
        
        # Parse metadata JSON if available
        if "metadata" in result and result["metadata"]:
            try:
                import json
                stored_metadata = json.loads(result["metadata"]) if isinstance(result["metadata"], str) else result["metadata"]
                
                # Add serializable fields from stored metadata
                for key, value in stored_metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                    elif isinstance(value, list) and value and isinstance(value[0], (str, int, float)):
                        metadata[key] = value
            except:
                pass
        
        return Document(page_content=text, metadata=metadata)
    
    def close(self):
        """Close Milvus connection."""
        try:
            self.store.disconnect()
            logger.info("Milvus connection closed")
        except Exception:
            pass


def load_documents_for_ragas(max_documents: int = 500, sample_strategy: str = "random", config: MilvusConfig = None) -> List[Document]:
    """
    Convenience function to load documents from Milvus for Ragas test generation.
    
    Args:
        max_documents: Maximum number of documents to load
        sample_strategy: Sampling strategy ('random' or 'sequential')
        config: Optional Milvus configuration
    
    Returns:
        List of LangChain Document objects
    """
    loader = MilvusDocumentLoader(config=config)
    try:
        documents = loader.load_documents(max_documents=max_documents, sample_strategy=sample_strategy)
        logger.info(f"Loaded {len(documents)} documents from Milvus")
        return documents
    finally:
        loader.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test document loading
    print("Testing Milvus document loader...")
    
    try:
        loader = MilvusDocumentLoader()
        
        # Get count
        count = loader.get_document_count()
        print(f"\nMilvus Collection Statistics:")
        print(f"  Collection: {loader.config.collection_name}")
        print(f"  Total documents: {count}")
        
        # Load sample
        print(f"\nLoading sample of 10 documents...")
        docs = loader.load_documents(max_documents=10, sample_strategy="random")
        
        print(f"\nLoaded {len(docs)} documents")
        if docs:
            print(f"\nFirst document preview:")
            print(f"  Content length: {len(docs[0].page_content)} chars")
            print(f"  Metadata keys: {list(docs[0].metadata.keys())}")
            print(f"  Content preview: {docs[0].page_content[:200]}...")
        
        loader.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

