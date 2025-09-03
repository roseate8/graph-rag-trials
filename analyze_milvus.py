#!/usr/bin/env python3
"""
Analyze duplicate chunks in Milvus database.
"""

import sys
import warnings
from pathlib import Path
from collections import Counter, defaultdict

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths
VECTOR_INGEST_PATH = Path(__file__).parent / "vector-ingest" / "src"
sys.path.insert(0, str(VECTOR_INGEST_PATH))

from embeddings.milvus_store import MilvusVectorStore
from embeddings.milvus_config import get_config


def analyze_duplicates():
    """Analyze duplicate chunks in Milvus collection."""
    print("Analyzing duplicates in Milvus...")
    
    try:
        # Connect to Milvus
        config = get_config()
        config.collection_name = "document_chunks"
        store = MilvusVectorStore(config)
        
        if not store.connect():
            print("ERROR: Failed to connect to Milvus")
            return
        
        print("Connected to Milvus successfully")
        
        # Get collection info
        print("Getting collection info...")
        collection = store.collection
        collection.load()
        
        # Get total count
        total_entities = collection.num_entities
        print(f"Total entities in collection: {total_entities}")
        
        # Query all chunk IDs in batches
        print("Retrieving chunk IDs...")
        chunk_id_results = []
        batch_size = 1000
        
        for offset in range(0, total_entities, batch_size):
            limit = min(batch_size, total_entities - offset)
            batch = collection.query(
                expr="",
                output_fields=["chunk_id"],
                limit=limit,
                offset=offset
            )
            chunk_id_results.extend(batch)
            print(f"  Retrieved {len(chunk_id_results)}/{total_entities} chunk IDs...")
        
        chunk_ids = [r['chunk_id'] for r in chunk_id_results]
        print(f"Retrieved {len(chunk_ids)} chunk IDs")
        
        # Count duplicates
        chunk_id_counts = Counter(chunk_ids)
        duplicate_chunk_ids = {chunk_id: count for chunk_id, count in chunk_id_counts.items() if count > 1}
        
        print(f"\nANALYSIS RESULTS:")
        print(f"  Total chunks: {len(chunk_ids)}")
        print(f"  Unique chunk IDs: {len(chunk_id_counts)}")
        print(f"  Duplicate chunk IDs: {len(duplicate_chunk_ids)}")
        
        if duplicate_chunk_ids:
            total_duplicates = sum(duplicate_chunk_ids.values()) - len(duplicate_chunk_ids)
            print(f"  Total duplicate entries: {total_duplicates}")
            
            print(f"\nTop 10 most duplicated chunks:")
            for chunk_id, count in sorted(duplicate_chunk_ids.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {chunk_id}: {count} copies")
        else:
            print("  No duplicates found!")
            
        return duplicate_chunk_ids
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 50)
    print("Milvus Duplicate Analysis Tool")
    print("=" * 50)
    analyze_duplicates()