#!/usr/bin/env python3
"""
Clean duplicate chunks from Milvus database.
"""

import sys
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths
VECTOR_INGEST_PATH = Path(__file__).parent / "vector-ingest" / "src"
sys.path.insert(0, str(VECTOR_INGEST_PATH))

from embeddings.milvus_store import MilvusVectorStore
from embeddings.milvus_config import get_config


def analyze_duplicates(collection_name: str = "document_chunks"):
    """Analyze duplicate chunks in Milvus collection."""
    print(f"Analyzing duplicates in collection: {collection_name}")
    
    try:
        # Connect to Milvus
        config = get_config()
        config.collection_name = collection_name
        store = MilvusVectorStore(config)
        
        if not store.connect():
            print("❌ Failed to connect to Milvus")
            return
        
        print("Connected to Milvus")
        
        # Get collection stats
        stats = store.get_collection_stats()
        total_entities = stats.get('num_entities', 0)
        print(f"Total entities in collection: {total_entities}")
        
        # Query all chunk data in batches
        print("Retrieving all chunks...")
        all_chunks = []
        batch_size = 1000
        offset = 0
        
        while offset < total_entities:
            limit = min(batch_size, total_entities - offset)
            batch = store.collection.query(
                expr="",
                output_fields=["chunk_id", "doc_id", "content", "word_count", "section_path"],
                limit=limit,
                offset=offset
            )
            all_chunks.extend(batch)
            offset += limit
            print(f"  Retrieved {len(all_chunks)}/{total_entities} chunks...")
        
        print(f"✅ Retrieved {len(all_chunks)} total chunks")
        
        # Analyze duplicates by chunk_id
        chunk_id_counts = Counter(chunk['chunk_id'] for chunk in all_chunks)
        duplicate_chunk_ids = {chunk_id: count for chunk_id, count in chunk_id_counts.items() if count > 1}
        
        print(f"\n📈 Analysis Results:")
        print(f"  Total chunks: {len(all_chunks)}")
        print(f"  Unique chunk IDs: {len(chunk_id_counts)}")
        print(f"  Duplicate chunk IDs: {len(duplicate_chunk_ids)}")
        print(f"  Total duplicate entries: {sum(duplicate_chunk_ids.values()) - len(duplicate_chunk_ids)}")
        
        if duplicate_chunk_ids:
            print(f"\n🔍 Top 10 most duplicated chunks:")
            for chunk_id, count in sorted(duplicate_chunk_ids.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {chunk_id}: {count} copies")
        
        # Group duplicates by content hash to find exact duplicates
        content_groups = defaultdict(list)
        for chunk in all_chunks:
            content_hash = hash(chunk['content'])
            content_groups[content_hash].append(chunk)
        
        exact_duplicates = {h: chunks for h, chunks in content_groups.items() if len(chunks) > 1}
        print(f"  Chunks with identical content: {len(exact_duplicates)}")
        
        return {
            'total_chunks': len(all_chunks),
            'duplicate_chunk_ids': duplicate_chunk_ids,
            'exact_duplicates': exact_duplicates,
            'all_chunks': all_chunks
        }
        
    except Exception as e:
        print(f"❌ Error analyzing duplicates: {e}")
        return None


def clean_duplicates(collection_name: str = "document_chunks", dry_run: bool = True):
    """Clean duplicate chunks from Milvus collection."""
    print(f"🧹 Cleaning duplicates in collection: {collection_name}")
    print(f"   Dry run: {'Yes' if dry_run else 'No'}")
    
    # First analyze
    analysis = analyze_duplicates(collection_name)
    if not analysis:
        return
    
    duplicate_chunk_ids = analysis['duplicate_chunk_ids']
    all_chunks = analysis['all_chunks']
    
    if not duplicate_chunk_ids:
        print("✅ No duplicates found! Database is clean.")
        return
    
    print(f"\n🎯 Identifying chunks to remove...")
    
    # Group chunks by chunk_id
    chunks_by_id = defaultdict(list)
    for chunk in all_chunks:
        chunks_by_id[chunk['chunk_id']].append(chunk)
    
    # For each duplicate chunk_id, keep the first occurrence and mark others for deletion
    chunks_to_delete = []
    for chunk_id, count in duplicate_chunk_ids.items():
        chunk_group = chunks_by_id[chunk_id]
        # Sort by primary key to ensure consistent selection
        chunk_group.sort(key=lambda x: x.get('id', 0))
        # Keep first, delete rest
        chunks_to_delete.extend(chunk_group[1:])
    
    print(f"📋 Chunks to delete: {len(chunks_to_delete)}")
    
    if dry_run:
        print("🔍 DRY RUN - Would delete these chunks:")
        for i, chunk in enumerate(chunks_to_delete[:10]):  # Show first 10
            print(f"  {i+1}. {chunk['chunk_id']} (ID: {chunk.get('id', 'N/A')})")
        if len(chunks_to_delete) > 10:
            print(f"  ... and {len(chunks_to_delete) - 10} more")
        
        print(f"\n💾 After cleanup:")
        print(f"  Total chunks: {analysis['total_chunks']} → {analysis['total_chunks'] - len(chunks_to_delete)}")
        print(f"  Unique chunks: {len(duplicate_chunk_ids)} duplicated chunk_ids would become unique")
        
        return
    
    # Perform actual deletion
    try:
        config = get_config()
        config.collection_name = collection_name
        store = MilvusVectorStore(config)
        
        if not store.connect():
            print("❌ Failed to connect to Milvus for deletion")
            return
        
        print("🗑️  Deleting duplicate chunks...")
        
        # Delete in batches
        batch_size = 100
        deleted_count = 0
        
        for i in range(0, len(chunks_to_delete), batch_size):
            batch = chunks_to_delete[i:i + batch_size]
            ids_to_delete = [str(chunk.get('id', '')) for chunk in batch if chunk.get('id')]
            
            if ids_to_delete:
                # Delete by primary key
                store.collection.delete(expr=f"id in {ids_to_delete}")
                deleted_count += len(ids_to_delete)
                print(f"  Deleted batch: {deleted_count}/{len(chunks_to_delete)} chunks")
        
        print(f"✅ Successfully deleted {deleted_count} duplicate chunks")
        
        # Verify cleanup
        print("🔍 Verifying cleanup...")
        final_analysis = analyze_duplicates(collection_name)
        
        if final_analysis and not final_analysis['duplicate_chunk_ids']:
            print("🎉 SUCCESS: All duplicates removed!")
        else:
            remaining = len(final_analysis['duplicate_chunk_ids']) if final_analysis else "unknown"
            print(f"⚠️  Warning: {remaining} duplicate chunk IDs still remain")
    
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")


def main():
    """Main cleanup script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean duplicate chunks from Milvus")
    parser.add_argument("--collection", default="document_chunks", help="Collection name")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't clean")
    parser.add_argument("--force", action="store_true", help="Actually delete duplicates (not dry run)")
    
    args = parser.parse_args()
    
    print("Milvus Duplicate Cleanup Tool")
    print("=" * 50)
    
    if args.analyze_only:
        analyze_duplicates(args.collection)
    else:
        clean_duplicates(args.collection, dry_run=not args.force)
        
        if not args.force:
            print("\n💡 To actually perform the cleanup, run with --force flag")


if __name__ == "__main__":
    main()