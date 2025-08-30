#!/usr/bin/env python3
"""
Milvus Cloud Setup and Integration Script
=========================================

This script helps you:
1. Install required dependencies 
2. Configure Milvus connection
3. Create collections and indexes
4. Upload existing chunks to Milvus
5. Test similarity search

Usage:
    python setup_milvus.py --setup      # Initial setup
    python setup_milvus.py --upload     # Upload chunks to Milvus
    python setup_milvus.py --test       # Test search functionality
    python setup_milvus.py --stats      # Show collection stats
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from embeddings.vector_store import MilvusVectorStore
from embeddings.config import MilvusConfig
from embeddings.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required dependencies for Milvus."""
    try:
        import subprocess
        
        print("Installing pymilvus...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pymilvus>=2.3.0"])
        
        print("Installing python-dotenv for environment variables...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def load_environment():
    """Load environment variables from .env file."""
    try:
        from dotenv import load_dotenv
        
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            print("✅ Loaded environment variables from .env")
        else:
            print("⚠️ No .env file found. Please create one based on .env.example")
            
    except ImportError:
        print("⚠️ python-dotenv not installed. Using system environment variables.")


def get_milvus_config() -> Dict[str, Any]:
    """Get Milvus configuration from environment."""
    # Try cloud config first
    cloud_config = MilvusConfig.get_cloud_config()
    
    # Check if cloud config is properly set
    if cloud_config.get("uri") or cloud_config.get("host"):
        return cloud_config
    
    # Fallback to default local config
    return MilvusConfig.get_default_config()


def setup_collection(vector_store: MilvusVectorStore) -> bool:
    """Setup Milvus collection and index."""
    print("\n🏗️ Setting up Milvus collection...")
    
    # Connect
    if not vector_store.connect():
        print("❌ Failed to connect to Milvus")
        return False
    
    # Create collection
    if not vector_store.create_collection():
        print("❌ Failed to create collection")
        return False
    
    # Create index
    print("Creating HNSW index for optimal search performance...")
    if not vector_store.create_index(index_type="HNSW", metric_type="IP"):
        print("❌ Failed to create index")
        return False
    
    print("✅ Collection and index created successfully!")
    return True


def upload_chunks_to_milvus(vector_store: MilvusVectorStore) -> bool:
    """Upload existing chunks to Milvus."""
    chunks_file = Path("output/processed_chunks.json")
    
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("   Run 'python main.py' first to process documents")
        return False
    
    print(f"\n📤 Loading chunks from {chunks_file}...")
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        if not chunks:
            print("❌ No chunks found in file")
            return False
        
        print(f"Found {len(chunks)} chunks to upload")
        
        # Convert chunks to Milvus format
        milvus_chunks = []
        for chunk in chunks:
            if "embedding" in chunk:
                milvus_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["document_id"],
                    "content": chunk["content"][:65000],  # Truncate if needed
                    "word_count": chunk["word_count"],
                    "section_path": str(chunk.get("section_info", {})),
                    "embedding": chunk["embedding"]
                })
        
        print(f"Prepared {len(milvus_chunks)} chunks for upload")
        
        # Upload in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(milvus_chunks), batch_size):
            batch = milvus_chunks[i:i+batch_size]
            
            if vector_store.insert_chunks(batch):
                total_uploaded += len(batch)
                print(f"Uploaded batch {i//batch_size + 1}: {len(batch)} chunks")
            else:
                print(f"❌ Failed to upload batch {i//batch_size + 1}")
                return False
        
        print(f"✅ Successfully uploaded {total_uploaded} chunks to Milvus!")
        return True
        
    except Exception as e:
        print(f"❌ Error uploading chunks: {e}")
        return False


def test_similarity_search(vector_store: MilvusVectorStore) -> bool:
    """Test similarity search functionality."""
    print("\n🔍 Testing similarity search...")
    
    try:
        # Create embedding service
        embedding_service = EmbeddingService()
        
        # Test query
        test_query = "What are the main risk factors for the business?"
        print(f"Query: '{test_query}'")
        
        # Generate query embedding
        query_embedding = embedding_service.embed_text(test_query)
        
        # Search
        results = vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=3
        )
        
        if results:
            print(f"✅ Found {len(results)} similar chunks:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['similarity_score']:.3f}")
                print(f"   Chunk: {result['chunk_id']}")
                print(f"   Content: {result['content'][:200]}...")
        else:
            print("❌ No results found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


def show_collection_stats(vector_store: MilvusVectorStore):
    """Show collection statistics."""
    print("\n📊 Collection Statistics:")
    
    if not vector_store.connect():
        print("❌ Failed to connect to Milvus")
        return
    
    stats = vector_store.get_collection_stats()
    
    if "error" in stats:
        print(f"❌ Error getting stats: {stats['error']}")
        return
    
    print(f"Collection Name: {stats['name']}")
    print(f"Number of Entities: {stats['num_entities']:,}")
    print(f"Indexes: {len(stats.get('indexes', []))}")


def main():
    parser = argparse.ArgumentParser(description="Milvus Cloud Setup and Integration")
    parser.add_argument("--setup", action="store_true", help="Setup collection and index")
    parser.add_argument("--upload", action="store_true", help="Upload chunks to Milvus")
    parser.add_argument("--test", action="store_true", help="Test similarity search")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
        return
    
    # Load environment variables
    load_environment()
    
    # Get configuration
    config = get_milvus_config()
    print(f"Using configuration: {config['collection_name']} collection")
    
    # Create vector store instance
    vector_store = MilvusVectorStore(**config)
    
    try:
        if args.setup:
            setup_collection(vector_store)
        
        elif args.upload:
            if vector_store.connect():
                upload_chunks_to_milvus(vector_store)
            
        elif args.test:
            if vector_store.connect():
                test_similarity_search(vector_store)
        
        elif args.stats:
            show_collection_stats(vector_store)
        
        else:
            print("Please specify an action: --setup, --upload, --test, --stats, or --install")
            print("Use --help for more information")
    
    finally:
        vector_store.disconnect()


if __name__ == "__main__":
    main()