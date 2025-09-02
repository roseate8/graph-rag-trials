#!/usr/bin/env python3
"""
Elasticsearch Chunk Dumper
Downloads all chunks from the rudram-embeddings collection and saves to JSON file.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from elasticsearch_client import create_elasticsearch_store

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Elasticsearch Configuration
ELASTICSEARCH_CONFIG = {
    "url": "https://1600c6e333fd4bdb8c8e9b9dec5c5fef.us-west-2.aws.found.io:443",
    "username": "elastic", 
    "password": "XI6rIccvUKLCgVnX11QPI8CV",
    "inference_id": "text-vectorizer"
}


def dump_elasticsearch_chunks(output_file: str = "elastic-embeddings.json") -> bool:
    """
    Download all chunks from Elasticsearch rudram-embeddings collection.
    
    Args:
        output_file: Name of output JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("🔍 Connecting to Elasticsearch to dump chunks...")
        
        # Connect to Elasticsearch
        store = create_elasticsearch_store(ELASTICSEARCH_CONFIG)
        if not store.connect():
            logger.error("❌ Failed to connect to Elasticsearch")
            return False
        
        # Check if index exists
        if not store.client.indices.exists(index=store.index_name):
            logger.warning(f"⚠️ Index '{store.index_name}' does not exist")
            store.disconnect()
            return False
        
        # Get total document count
        doc_count = store.get_document_count()
        logger.info(f"📊 Found {doc_count} documents in '{store.index_name}' collection")
        
        if doc_count == 0:
            logger.warning("⚠️ No documents found in collection")
            store.disconnect()
            return False
        
        # Fetch all documents using scroll API for efficient retrieval
        logger.info("📥 Downloading all chunks from Elasticsearch...")
        
        all_chunks = []
        
        # Use scroll API to get all documents efficiently including embeddings
        response = store.client.search(
            index=store.index_name,
            scroll='2m',
            size=1000,  # Batch size
            body={
                "query": {"match_all": {}},
                "_source": True,  # Include all source fields including embeddings
                "stored_fields": ["*"]  # Include all stored fields
            }
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        
        # Process first batch
        for hit in hits:
            source = hit['_source']
            # Include document ID and any computed fields like embeddings
            chunk_data = dict(source)
            chunk_data['_id'] = hit['_id']
            chunk_data['_score'] = hit.get('_score', 0)
            all_chunks.append(chunk_data)
        
        # Continue scrolling until all documents are retrieved
        while len(hits) > 0:
            response = store.client.scroll(
                scroll_id=scroll_id,
                scroll='2m'
            )
            
            hits = response['hits']['hits']
            for hit in hits:
                source = hit['_source']
                # Include document ID and any computed fields like embeddings
                chunk_data = dict(source)
                chunk_data['_id'] = hit['_id']
                chunk_data['_score'] = hit.get('_score', 0)
                all_chunks.append(chunk_data)
            
            logger.info(f"📥 Downloaded {len(all_chunks)}/{doc_count} chunks...")
        
        # Clear scroll context
        store.client.clear_scroll(scroll_id=scroll_id)
        
        logger.info(f"✅ Successfully downloaded {len(all_chunks)} chunks from Elasticsearch")
        
        # Analyze chunk types and embeddings
        table_chunks = sum(1 for c in all_chunks if c.get('chunk_type') == 'table')
        text_chunks = sum(1 for c in all_chunks if c.get('chunk_type') == 'text')
        chunks_with_embeddings = sum(1 for c in all_chunks if c.get('embedding'))
        
        # Check embedding dimensions from Elasticsearch inference
        embedding_dims = []
        for chunk in all_chunks[:5]:  # Check first 5 chunks
            if chunk.get('embedding'):
                embedding_dims.append(len(chunk['embedding']))
        
        avg_embedding_dim = sum(embedding_dims) / len(embedding_dims) if embedding_dims else 0
        
        logger.info(f"📊 Chunk analysis:")
        logger.info(f"  📄 Text chunks: {text_chunks}")
        logger.info(f"  📊 Table chunks: {table_chunks}")
        logger.info(f"  🔮 Chunks with embeddings: {chunks_with_embeddings}")
        logger.info(f"  📏 Average embedding dimension: {avg_embedding_dim:.0f}")
        
        # Check spaCy extractions
        chunks_with_spacy = sum(1 for c in all_chunks if any(c.get(field, []) for field in ['organizations', 'locations', 'products', 'events']))
        logger.info(f"  🧠 Chunks with spaCy entities: {chunks_with_spacy}")
        
        # Save to JSON file
        output_path = Path(output_file)
        dump_data = {
            "collection_name": store.index_name,
            "total_chunks": len(all_chunks),
            "table_chunks": table_chunks,
            "text_chunks": text_chunks,
            "chunks_with_embeddings": chunks_with_embeddings,
            "elasticsearch_config": {
                "url": ELASTICSEARCH_CONFIG["url"],
                "inference_id": ELASTICSEARCH_CONFIG["inference_id"]
            },
            "chunks": all_chunks
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(all_chunks)} chunks to: {output_path}")
        logger.info(f"📊 File size: {output_path.stat().st_size:,} bytes")
        
        store.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error dumping Elasticsearch chunks: {e}")
        return False


def main():
    """Main entry point for chunk dumping."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dump chunks from Elasticsearch rudram-embeddings collection")
    parser.add_argument(
        "--output-file", "-o",
        default="elastic-embeddings.json",
        help="Output JSON file name (default: elastic-embeddings.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 Starting Elasticsearch chunk dump...")
    
    success = dump_elasticsearch_chunks(args.output_file)
    
    if success:
        logger.info("✅ Chunk dump completed successfully!")
    else:
        logger.error("❌ Chunk dump failed!")
        exit(1)


if __name__ == "__main__":
    main()