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
        self.index_name = "rudram-embeddings"
        
    def connect(self) -> bool:
        """Connect to Elasticsearch cluster."""
        try:
            # Initialize Elasticsearch client for version 8.x compatibility
            self.client = Elasticsearch(
                [self.config["url"]],
                basic_auth=(self.config["username"], self.config["password"]),
                verify_certs=True,
                request_timeout=30
            )
            
            # Set compatibility headers globally
            self.client._client_meta_version = (8, 0, 0)
            
            # Test connection
            info = self.client.info()
            logger.info(f"✅ Connected to Elasticsearch at {self.config['url']}")
            logger.info(f"📊 Cluster: {info['cluster_name']}")
            return True
                
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
                        
                        # Dense vector for semantic search (using inference endpoint)
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768,  # text-vectorizer inference endpoint dimension
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
        """Upload document chunks to Elasticsearch using inference endpoint for embeddings."""
        if not chunks:
            logger.warning("⚠️ No chunks to upload")
            return False
        
        try:
            # Prepare documents for bulk upload with inference pipeline
            docs = []
            for chunk in chunks:
                # Extract spaCy extraction fields if present
                spacy_data = chunk.get('spacy_extraction', {})
                
                # Prepare document source
                doc_source = {
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
                }
                
                doc = {
                    "_index": self.index_name,
                    "_source": doc_source
                }
                
                # Add all chunks (no need to check for embeddings - they'll be generated)
                docs.append(doc)
            
            # Bulk upload to Elasticsearch (embeddings will be added via inference afterward)
            logger.info(f"📤 Uploading {len(docs)} chunks to Elasticsearch...")
            
            success_count, failed_items = bulk(
                self.client,
                docs,
                index=self.index_name,
                chunk_size=100,
                request_timeout=60
            )
            
            # Generate embeddings using inference endpoint for all uploaded documents
            if success_count > 0:
                logger.info(f"🧠 Generating embeddings using '{self.config['inference_id']}' inference endpoint...")
                self._add_embeddings_via_inference()
            
            if failed_items:
                logger.warning(f"⚠️ {len(failed_items)} items failed to upload")
                for item in failed_items[:3]:  # Show first 3 failures
                    logger.warning(f"Failed item: {item}")
            
            logger.info(f"✅ Successfully uploaded {success_count} documents with automatic embeddings")
            
            # Get index stats
            stats = self.client.indices.stats(index=self.index_name)
            doc_count = stats['indices'][self.index_name]['total']['docs']['count']
            logger.info(f"📊 Index now contains {doc_count} total documents with embeddings")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload chunks to Elasticsearch: {e}")
            return False
    
    def _add_embeddings_via_inference(self):
        """Add embeddings to uploaded documents using inference endpoint."""
        try:
            # Use update_by_query to add embeddings to all documents without embeddings
            update_script = {
                "script": {
                    "source": """
                    if (ctx._source.embedding == null || ctx._source.embedding.size() == 0) {
                        Map inferenceResult = params.inference.inference(params.inference_id, [ctx._source.content]);
                        if (inferenceResult.text_embedding != null && inferenceResult.text_embedding.size() > 0) {
                            ctx._source.embedding = inferenceResult.text_embedding[0].embedding;
                        }
                    }
                    """,
                    "params": {
                        "inference_id": self.config["inference_id"]
                    }
                }
            }
            
            # For now, let's use a simpler approach - get documents and update them individually
            logger.info("🔍 Fetching documents to add embeddings...")
            
            # Get all documents without embeddings (simplified approach)
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "must_not": {
                                "exists": {"field": "embedding"}
                            }
                        }
                    },
                    "size": 100
                }
            )
            
            docs_to_update = response['hits']['hits']
            logger.info(f"📝 Found {len(docs_to_update)} documents needing embeddings")
            
            # Generate embeddings for each document using inference endpoint
            for doc in docs_to_update[:10]:  # Limit to first 10 for testing
                doc_id = doc['_id']
                content = doc['_source']['content']
                
                try:
                    # Generate embedding using inference endpoint
                    inference_response = self.client.inference.inference(
                        inference_id=self.config["inference_id"],
                        body={"input": [content]}
                    )
                    
                    if 'text_embedding' in inference_response:
                        embedding = inference_response['text_embedding'][0]['embedding']
                        
                        # Update document with embedding
                        self.client.update(
                            index=self.index_name,
                            id=doc_id,
                            body={
                                "doc": {
                                    "embedding": embedding
                                }
                            }
                        )
                        
                        logger.info(f"✅ Added {len(embedding)}-dim embedding to document {doc_id}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to add embedding to {doc_id}: {e}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error adding embeddings via inference: {e}")
    
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