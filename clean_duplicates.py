#!/usr/bin/env python3
"""
Clean duplicate chunks from Milvus using the existing RAG system.
"""

import sys
import warnings
from pathlib import Path
from collections import Counter

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths
NAIVE_RAG_PATH = Path(__file__).parent / "naive-rag"
VECTOR_INGEST_PATH = Path(__file__).parent / "vector-ingest" / "src"
sys.path.insert(0, str(NAIVE_RAG_PATH))
sys.path.insert(0, str(VECTOR_INGEST_PATH))

from core import create_rag_system


def clean_milvus_duplicates(dry_run=True):
    """Clean duplicates from Milvus by recreating the collection."""
    print("=" * 50)
    print("Milvus Duplicate Cleanup")
    print("=" * 50)
    
    try:
        # Create RAG system to access Milvus
        print("1. Connecting to RAG system...")
        rag = create_rag_system(llm_type="mock", enable_reranking=False)
        
        if not rag.connect():
            print("ERROR: Could not connect to Milvus")
            return
        
        print("   Connected successfully")
        
        # Get retriever to access Milvus directly
        retriever = rag.retriever
        milvus_store = retriever.milvus_store
        collection = milvus_store.collection
        
        # Get all data
        print("2. Retrieving all chunks from Milvus...")
        all_data = collection.query(
            expr="", 
            output_fields=["chunk_id", "doc_id", "content", "word_count", "section_path", "embedding"],
            limit=1000  # Start with first 1000 to test
        )
        
        print(f"   Retrieved {len(all_data)} chunks")
        
        # Analyze duplicates
        chunk_ids = [item['chunk_id'] for item in all_data]
        chunk_id_counts = Counter(chunk_ids)
        duplicates = {chunk_id: count for chunk_id, count in chunk_id_counts.items() if count > 1}
        
        print(f"3. Analysis results:")
        print(f"   Total chunks: {len(all_data)}")
        print(f"   Unique chunk IDs: {len(chunk_id_counts)}")
        print(f"   Duplicate chunk IDs: {len(duplicates)}")
        
        if duplicates:
            total_duplicate_entries = sum(duplicates.values()) - len(duplicates)
            print(f"   Total duplicate entries: {total_duplicate_entries}")
            
            print(f"\n   Top duplicated chunks:")
            for chunk_id, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     {chunk_id}: {count} copies")
        
        if not duplicates:
            print("   No duplicates found!")
            return
        
        if dry_run:
            print(f"\n4. DRY RUN - Would remove {total_duplicate_entries} duplicate entries")
            print("   To actually clean, run with: python clean_duplicates.py --force")
            return
        
        # Clean duplicates by keeping first occurrence of each chunk_id
        print(f"4. Cleaning duplicates...")
        
        # Group by chunk_id and keep first occurrence
        seen_chunk_ids = set()
        unique_data = []
        
        for item in all_data:
            chunk_id = item['chunk_id']
            if chunk_id not in seen_chunk_ids:
                unique_data.append(item)
                seen_chunk_ids.add(chunk_id)
        
        print(f"   Deduplicated: {len(all_data)} -> {len(unique_data)} chunks")
        
        # Drop and recreate collection with clean data
        print("5. Recreating collection with clean data...")
        collection_name = milvus_store.config.collection_name
        
        # Drop existing collection
        collection.drop()
        print(f"   Dropped collection: {collection_name}")
        
        # Recreate collection
        milvus_store.create_collection(drop_if_exists=False)
        print(f"   Recreated collection: {collection_name}")
        
        # Reconnect to new collection
        milvus_store.connect()
        
        # Insert clean data
        if unique_data:
            # Prepare data for insertion
            chunk_ids = [item['chunk_id'] for item in unique_data]
            doc_ids = [item['doc_id'] for item in unique_data]
            contents = [item['content'] for item in unique_data]
            word_counts = [item['word_count'] for item in unique_data]
            section_paths = [item['section_path'] for item in unique_data]
            embeddings = [item['embedding'] for item in unique_data]
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(unique_data), batch_size):
                end_idx = min(i + batch_size, len(unique_data))
                batch_data = [
                    chunk_ids[i:end_idx],
                    doc_ids[i:end_idx], 
                    contents[i:end_idx],
                    word_counts[i:end_idx],
                    section_paths[i:end_idx],
                    embeddings[i:end_idx]
                ]
                
                milvus_store.collection.insert(batch_data)
                print(f"   Inserted batch: {end_idx}/{len(unique_data)} chunks")
        
        print("6. SUCCESS: Duplicates removed and collection recreated!")
        print(f"   Final count: {len(unique_data)} unique chunks")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean Milvus duplicates")
    parser.add_argument("--force", action="store_true", help="Actually perform cleanup")
    
    args = parser.parse_args()
    
    clean_milvus_duplicates(dry_run=not args.force)


if __name__ == "__main__":
    main()