#!/usr/bin/env python3
"""
Elasticsearch Vector Store Client for RAG embeddings.
Independent implementation for uploading document chunks with embeddings to Elasticsearch.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

logger = logging.getLogger(__name__)


class ElasticsearchVectorStore:
    """Elasticsearch vector store for RAG document chunks."""
    
    def __init__(self, config: Dict[str, str]):
        """Initialize Elasticsearch client with configuration."""
        self.config = config
        self.client = None
        self.index_name = "rag-documents"
        
    def connect(self) -> bool:
        """Connect to Elasticsearch cluster."""
        try:
            self.client = Elasticsearch(
                hosts=[self.config["url"]],
                basic_auth=(self.config["username"], self.config["password"]),
                verify_certs=True,
                request_timeout=30
            )
            
            # Test connection
            if self.client.ping():
                logger.info(f"✅ Connected to Elasticsearch at {self.config['url']}")
                return True
            else:
                logger.error("❌ Failed to ping Elasticsearch")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
            return False
    
    def create_index(self) -> bool:
        """Create index with proper mapping for embeddings and metadata."""
        try:
            # Define index mapping for document chunks
            mapping = {
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "chunk_type": {"type": "keyword"},
                        "word_count": {"type": "integer"},
                        "section_path": {"type": "text"},
                        "page": {"type": "integer"},
                        "folder_path": {"type": "keyword"},
                        "product_version": {"type": "keyword"},
                        
                        # spaCy extraction fields
                        "organizations": {"type": "keyword"},
                        "locations": {"type": "keyword"},
                        "products": {"type": "keyword"},
                        "events": {"type": "keyword"},
                        
                        # Original metadata fields
                        "regions": {"type": "keyword"},
                        "metrics": {"type": "keyword"},
                        "orgs": {"type": "keyword"},
                        "time_periods": {"type": "keyword"},
                        
                        # Dense vector for semantic search
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 384,  # BGE-small-en-v1.5 dimension
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }
            
            # Create index if it doesn't exist
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"✅ Created Elasticsearch index: {self.index_name}")
            else:
                logger.info(f"📁 Index already exists: {self.index_name}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create Elasticsearch index: {e}")
            return False
    
    def clear_index(self) -> bool:
        """Clear all documents from the index."""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.delete_by_query(
                    index=self.index_name,
                    body={"query": {"match_all": {}}}
                )
                logger.info(f"🧹 Cleared all documents from index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear Elasticsearch index: {e}")
            return False
    
    def delete_index(self) -> bool:
        """Delete the entire index."""
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"🗑️ Deleted Elasticsearch index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to delete Elasticsearch index: {e}")
            return False
    
    def upload_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """Upload document chunks to Elasticsearch."""
        if not chunks:
            logger.warning("⚠️ No chunks to upload")
            return False
        
        try:
            # Prepare documents for bulk upload
            docs = []
            for chunk in chunks:
                # Extract spaCy extraction fields if present
                spacy_data = chunk.get('spacy_extraction', {})
                
                doc = {
                    "_index": self.index_name,
                    "_source": {
                        "chunk_id": chunk.get("chunk_id"),
                        "doc_id": chunk.get("doc_id"),
                        "content": chunk.get("content", "")[:32766],  # Elasticsearch text limit
                        "chunk_type": chunk.get("chunk_type", "text"),
                        "word_count": chunk.get("word_count", 0),
                        "section_path": str(chunk.get("section_path", "")),
                        "page": chunk.get("page"),
                        "folder_path": chunk.get("folder_path", []),
                        "product_version": chunk.get("product_version", "v1"),
                        
                        # spaCy extraction results (flattened for Elasticsearch)
                        "organizations": spacy_data.get("organizations", []),
                        "locations": spacy_data.get("locations", []),
                        "products": spacy_data.get("products", []),
                        "events": spacy_data.get("events", []),
                        
                        # Original metadata fields
                        "regions": chunk.get("regions", []),
                        "metrics": chunk.get("metrics", []),
                        "orgs": chunk.get("orgs", []),
                        "time_periods": chunk.get("time_periods", []),
                        
                        # Embedding vector
                        "embedding": chunk.get("embedding", [])
                    }
                }
                
                # Only add document if it has an embedding
                if chunk.get("embedding"):
                    docs.append(doc)
            
            if not docs:
                logger.warning("⚠️ No chunks with embeddings to upload")
                return False
            
            # Bulk upload to Elasticsearch
            logger.info(f"📤 Uploading {len(docs)} chunks to Elasticsearch...")
            
            success_count, failed_items = bulk(
                self.client,
                docs,
                index=self.index_name,
                chunk_size=100,
                request_timeout=60
            )
            
            if failed_items:
                logger.warning(f"⚠️ {len(failed_items)} items failed to upload")
                for item in failed_items[:3]:  # Show first 3 failures
                    logger.warning(f"Failed item: {item}")
            
            logger.info(f"✅ Successfully uploaded {success_count} documents to Elasticsearch")
            
            # Get index stats
            stats = self.client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            logger.info(f"📊 Index now contains {doc_count} total documents")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload chunks to Elasticsearch: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get total document count in the index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                return 0
            
            result = self.client.count(index=self.index_name)
            return result['count']
            
        except Exception as e:
            logger.error(f"❌ Failed to get document count: {e}")
            return 0
    
    def disconnect(self):
        """Disconnect from Elasticsearch."""
        if self.client:
            self.client.close()
            logger.info("📴 Disconnected from Elasticsearch")


def create_elasticsearch_store(config: Dict[str, str]) -> ElasticsearchVectorStore:
    """Factory function to create Elasticsearch vector store."""
    return ElasticsearchVectorStore(config)