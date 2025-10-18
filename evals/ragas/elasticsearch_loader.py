"""Elasticsearch document loader for Ragas test generation."""

import logging
import random
from typing import List, Dict, Any

from elasticsearch import Elasticsearch
from langchain.docstore.document import Document
from tqdm import tqdm

from config import ELASTICSEARCH_CONFIG

logger = logging.getLogger(__name__)


class ElasticsearchDocumentLoader:
    """Loads documents from Elasticsearch and converts them to LangChain Document format."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Elasticsearch connection."""
        config = config or ELASTICSEARCH_CONFIG
        self.index_name = config["index_name"]
        
        self.client = Elasticsearch(
            config["url"],
            basic_auth=(config["username"], config["password"]),
            verify_certs=config["verify_certs"],
            request_timeout=config["timeout"],
        )
        
        if not self.client.ping():
            raise ConnectionError(f"Failed to connect to Elasticsearch at {config['url']}")
        
        logger.info(f"Connected to Elasticsearch at {config['url']}")
    
    def get_document_count(self) -> int:
        """Get total document count in index."""
        try:
            result = self.client.count(index=self.index_name)
            return result["count"]
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    def load_documents(
        self,
        max_documents: int = None,
        sample_strategy: str = "random"
    ) -> List[Document]:
        """Load documents from Elasticsearch with specified sampling strategy."""
        total_docs = self.get_document_count()
        logger.info(f"Found {total_docs} documents in index '{self.index_name}'")
        
        if not total_docs:
            return []
        
        fetch_size = min(max_documents, total_docs) if max_documents else total_docs
        
        if sample_strategy == "random" and max_documents and max_documents < total_docs:
            return self._load_random_sample(fetch_size)
        return self._load_sequential(fetch_size)
    
    def _load_sequential(self, max_docs: int) -> List[Document]:
        """Load documents sequentially using scroll API. O(n) time complexity."""
        documents = []
        scroll_size = min(100, max_docs)
        
        response = self.client.search(
            index=self.index_name,
            query={"match_all": {}},
            size=scroll_size,
            scroll="2m",
        )
        
        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        
        with tqdm(total=max_docs, desc="Loading documents") as pbar:
            while hits and len(documents) < max_docs:
                for hit in hits:
                    if len(documents) >= max_docs:
                        break
                    documents.append(self._convert_to_document(hit))
                    pbar.update(1)
                
                if len(documents) >= max_docs:
                    break
                
                response = self.client.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]
        
        try:
            self.client.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass
        
        return documents
    
    def _load_random_sample(self, sample_size: int) -> List[Document]:
        """Load random sample using random_score. O(n log n) for sorting."""
        random_query = {
            "function_score": {
                "query": {"match_all": {}},
                "random_score": {"seed": random.randint(1, 1000000)},
            }
        }
        
        response = self.client.search(
            index=self.index_name,
            query=random_query,
            size=min(sample_size, 10000),
        )
        
        documents = [
            self._convert_to_document(hit)
            for hit in tqdm(response["hits"]["hits"], desc="Converting documents")
        ]
        
        return documents[:sample_size]
    
    def _convert_to_document(self, hit: Dict) -> Document:
        """Convert Elasticsearch hit to LangChain Document. O(1) operation."""
        source = hit["_source"]
        text = source.get("text", source.get("content", ""))
        
        # Build metadata efficiently
        metadata = {"id": hit["_id"], "index": hit["_index"]}
        
        # Add only serializable fields
        for key, value in source.items():
            if key not in ["text", "content", "embedding", "dense_vector"]:
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                elif isinstance(value, list) and value and isinstance(value[0], (str, int, float)):
                    metadata[key] = value
        
        return Document(page_content=text, metadata=metadata)
    
    def close(self):
        """Close Elasticsearch connection."""
        try:
            self.client.close()
        except Exception:
            pass


def load_documents_for_ragas(max_documents: int = 500, sample_strategy: str = "random") -> List[Document]:
    """Convenience function to load documents for Ragas test generation."""
    loader = ElasticsearchDocumentLoader()
    try:
        documents = loader.load_documents(max_documents=max_documents, sample_strategy=sample_strategy)
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    finally:
        loader.close()
