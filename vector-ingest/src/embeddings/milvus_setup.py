#!/usr/bin/env python3
"""
Milvus Standalone Setup and Management Script
============================================

This script helps you set up and manage a Milvus standalone instance for document embeddings.

Usage:
    python milvus_setup.py install       # Install dependencies
    python milvus_setup.py start         # Start Milvus with Docker
    python milvus_setup.py stop          # Stop Milvus
    python milvus_setup.py init          # Initialize collection and index
    python milvus_setup.py upload        # Upload embeddings from processed chunks
    python milvus_setup.py search        # Test similarity search
    python milvus_setup.py stats         # Show collection statistics
    python milvus_setup.py health        # Check health status
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add parent directories to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
root_dir = src_dir.parent
sys.path.extend([str(src_dir), str(root_dir)])

try:
    # Try relative imports first (when run as module)
    from .milvus_config import MilvusConfig, get_config
    from .milvus_store import MilvusVectorStore
    from .embedding_service import EmbeddingService
except ImportError:
    # Fallback to direct imports (when run as script)
    from milvus_config import MilvusConfig, get_config
    from milvus_store import MilvusVectorStore
    from embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required dependencies for Milvus standalone."""
    print("Installing Milvus dependencies...")
    
    dependencies = [
        "pymilvus>=2.3.0",
        "docker>=6.0.0"
    ]
    
    try:
        for dep in dependencies:
            print(f"   Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False


def start_milvus():
    """Start Milvus standalone using Docker."""
    print("🚀 Starting Milvus standalone with Docker...")
    
    try:
        # Check if Docker is available
        subprocess.check_call([
            "docker", "--version"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("   Docker found, starting Milvus...")
        
        # Run Milvus standalone container
        cmd = [
            "docker", "run", "-d",
            "--name", "milvus-standalone",
            "-p", "19530:19530",
            "-p", "9091:9091", 
            "-v", "milvus_data:/var/lib/milvus",
            "milvusdb/milvus:latest",
            "milvus", "run", "standalone"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Milvus started successfully!")
            print("   Container: milvus-standalone")
            print("   Port: 19530")
            print("   Waiting for Milvus to be ready...")
            
            # Wait for Milvus to be ready
            time.sleep(10)
            return True
        else:
            # Check if container already exists
            if "already in use" in result.stderr:
                print("ℹ️  Milvus container already running")
                return True
            else:
                print(f"❌ Failed to start Milvus: {result.stderr}")
                return False
    
    except subprocess.CalledProcessError:
        print("❌ Docker not found. Please install Docker first.")
        print("   Visit: https://docs.docker.com/get-docker/")
        return False
    except Exception as e:
        print(f"❌ Error starting Milvus: {e}")
        return False


def stop_milvus():
    """Stop Milvus standalone container."""
    print("🛑 Stopping Milvus standalone...")
    
    try:
        # Stop container
        subprocess.check_call([
            "docker", "stop", "milvus-standalone"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Remove container
        subprocess.check_call([
            "docker", "rm", "milvus-standalone"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ Milvus stopped successfully!")
        return True
        
    except subprocess.CalledProcessError:
        print("ℹ️  Milvus container not found or already stopped")
        return True
    except Exception as e:
        print(f"❌ Error stopping Milvus: {e}")
        return False


def initialize_collection(config: MilvusConfig):
    """Initialize Milvus collection and index."""
    print(f"🏗️  Initializing collection: {config.collection_name}")
    
    with MilvusVectorStore(config) as store:
        # Connect
        if not store.connect():
            print("❌ Failed to connect to Milvus")
            return False
        
        # Create collection
        if not store.create_collection():
            print("❌ Failed to create collection")
            return False
        
        # Create index
        if not store.create_index():
            print("❌ Failed to create index")
            return False
        
        print("✅ Collection initialized successfully!")
        return True


def upload_embeddings(config: MilvusConfig):
    """Upload embeddings from processed chunks file."""
    # Look for processed chunks file
    chunks_file = root_dir / "output" / "processed_chunks.json"
    
    if not chunks_file.exists():
        print(f"❌ Chunks file not found: {chunks_file}")
        print("   Run the main processing pipeline first to generate embeddings")
        return False
    
    print(f"📤 Loading chunks from {chunks_file}")
    
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        if not chunks:
            print("❌ No chunks found in file")
            return False
        
        print(f"   Found {len(chunks)} chunks to upload")
        
        # Filter chunks with embeddings
        chunks_with_embeddings = [
            chunk for chunk in chunks 
            if "embedding" in chunk and chunk["embedding"]
        ]
        
        if not chunks_with_embeddings:
            print("❌ No chunks with embeddings found")
            return False
        
        print(f"   {len(chunks_with_embeddings)} chunks have embeddings")
        
        # Upload to Milvus
        with MilvusVectorStore(config) as store:
            if not store.connect():
                print("❌ Failed to connect to Milvus")
                return False
            
            # Convert chunk format for Milvus
            milvus_chunks = []
            for chunk in chunks_with_embeddings:
                milvus_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"], 
                    "content": chunk["content"][:65535],  # Truncate if needed
                    "word_count": chunk["word_count"],
                    "section_path": str(chunk.get("section_path", "")),
                    "embedding": chunk["embedding"]
                })
            
            # Upload in batches
            batch_size = 100
            total_uploaded = 0
            
            for i in range(0, len(milvus_chunks), batch_size):
                batch = milvus_chunks[i:i + batch_size]
                
                if store.insert_chunks(batch):
                    total_uploaded += len(batch)
                    print(f"   Uploaded batch {i//batch_size + 1}: {len(batch)} chunks")
                else:
                    print(f"❌ Failed to upload batch {i//batch_size + 1}")
                    return False
            
            print(f"✅ Successfully uploaded {total_uploaded} chunks!")
            return True
    
    except Exception as e:
        print(f"❌ Error uploading chunks: {e}")
        return False


def test_search(config: MilvusConfig):
    """Test similarity search functionality."""
    print("🔍 Testing similarity search...")
    
    try:
        # Create embedding service
        embedding_service = EmbeddingService()
        
        # Test queries
        test_queries = [
            "What are the main business risks?",
            "Revenue and financial performance",
            "Company strategy and future plans"
        ]
        
        with MilvusVectorStore(config) as store:
            if not store.connect():
                print("❌ Failed to connect to Milvus")
                return False
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n🔎 Test Query {i}: '{query}'")
                
                # Generate query embedding
                query_embedding = embedding_service.embed_text(query)
                
                # Search for similar chunks
                results = store.search_similar(
                    query_embedding=query_embedding,
                    top_k=3
                )
                
                if results:
                    print(f"   Found {len(results)} similar chunks:")
                    for j, result in enumerate(results, 1):
                        score = result['similarity_score']
                        chunk_id = result['chunk_id']
                        content = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                        print(f"   {j}. Score: {score:.3f} | {chunk_id}")
                        print(f"      Content: {content}")
                else:
                    print("   No results found")
        
        print("\n✅ Search test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False


def show_stats(config: MilvusConfig):
    """Show collection statistics."""
    print("📊 Collection Statistics")
    
    with MilvusVectorStore(config) as store:
        if not store.connect():
            print("❌ Failed to connect to Milvus")
            return False
        
        stats = store.get_stats()
        
        if "error" in stats:
            print(f"❌ Error getting stats: {stats['error']}")
            return False
        
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Entities: {stats['num_entities']:,}")
        print(f"   Index: {stats['index_type']} ({stats['metric_type']})")
        print(f"   Dimensions: {stats['embedding_dim']}")
        
        return True


def health_check(config: MilvusConfig):
    """Check Milvus health status."""
    print("🏥 Health Check")
    
    with MilvusVectorStore(config) as store:
        if not store.connect():
            print("❌ Connection failed")
            return False
        
        health = store.health_check()
        
        print(f"   Connected: {'✅' if health['connected'] else '❌'}")
        print(f"   Collection exists: {'✅' if health['collection_exists'] else '❌'}")
        print(f"   Collection loaded: {'✅' if health['collection_loaded'] else '❌'}")
        print(f"   Entities: {health['num_entities']:,}")
        
        if health['errors']:
            print("   Errors:")
            for error in health['errors']:
                print(f"     - {error}")
        
        return len(health['errors']) == 0


def main():
    parser = argparse.ArgumentParser(description="Milvus Standalone Setup and Management")
    parser.add_argument("action", choices=[
        "install", "start", "stop", "init", "upload", 
        "search", "stats", "health"
    ], help="Action to perform")
    parser.add_argument("--config", choices=["development", "production", "testing"],
                       default="production", help="Configuration profile")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.config)
    
    print(f"Using {args.config} configuration")
    print(f"   Host: {config.host}:{config.port}")
    print(f"   Collection: {config.collection_name}")
    
    try:
        if args.action == "install":
            success = install_dependencies()
        
        elif args.action == "start":
            success = start_milvus()
        
        elif args.action == "stop":
            success = stop_milvus()
        
        elif args.action == "init":
            success = initialize_collection(config)
        
        elif args.action == "upload":
            success = upload_embeddings(config)
        
        elif args.action == "search":
            success = test_search(config)
        
        elif args.action == "stats":
            success = show_stats(config)
        
        elif args.action == "health":
            success = health_check(config)
        
        else:
            print(f"❌ Unknown action: {args.action}")
            success = False
        
        if success:
            print(f"\n{args.action.title()} completed successfully!")
        else:
            print(f"\n{args.action.title()} failed!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()