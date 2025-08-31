"""
Script to extract random sample chunks from Milvus and save them to a JSON file.
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from embeddings.milvus_store import MilvusVectorStore
    from embeddings.milvus_config import MilvusConfig
except ImportError:
    logger.error("Failed to import required modules")
    sys.exit(1)


def get_collection_schema(store) -> List[str]:
    """Get the actual field names from the collection schema."""
    try:
        if store.collection is None:
            logger.error("Collection not loaded")
            return []
        
        # Get field names from schema
        fields = [field.name for field in store.collection.schema.fields]
        logger.info(f"Available fields in collection: {fields}")
        return fields
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        return []


def extract_sample_chunks(sample_size: int = 100, include_embeddings: bool = True) -> List[Dict[str, Any]]:
    """
    Extract random sample chunks from Milvus database.
    
    Args:
        sample_size: Number of chunks to extract
        include_embeddings: Whether to include embedding vectors (makes file larger)
        
    Returns:
        List of chunk dictionaries
    """
    logger.info(f"Extracting {sample_size} random chunks from Milvus")
    
    try:
        # Initialize Milvus store with default config
        store = MilvusVectorStore()
        
        # Connect to Milvus
        if not store.connect():
            logger.error("Failed to connect to Milvus")
            return []
        
        logger.info("Connected to Milvus successfully")
        
        # Check if collection exists
        if not store.collection_exists():
            logger.error("Collection does not exist")
            return []
        
        # Get total entity count
        total_entities = store.get_entity_count()
        logger.info(f"Total entities in collection: {total_entities}")
        
        if total_entities == 0:
            logger.error("No entities found in collection")
            return []
        
        # Adjust sample size if needed
        sample_size = min(sample_size, total_entities)
        logger.info(f"Adjusted sample size: {sample_size}")
        
        # Load collection
        store.collection.load()
        
        # Get actual fields from schema
        schema_fields = get_collection_schema(store)
        if not schema_fields:
            logger.error("Could not get schema fields")
            return []
        
        # Filter fields - exclude primary key and vector fields for metadata extraction
        metadata_fields = [f for f in schema_fields if f not in ['id', 'embedding']]
        
        # Include embedding if requested
        output_fields = metadata_fields + (['embedding'] if include_embeddings else [])
        
        # For dynamic fields, we need to specify them explicitly since they're not in schema
        # Add known dynamic fields that might exist - pre-allocated list for efficiency
        output_fields.extend([
            # Chunk type and metadata containers
            'chunk_type', 'embedding_dim', 'structural_metadata', 'table_metadata', 'entity_metadata',
            
            # Specific table fields
            'table_id', 'column_headers', 'table_title', 'table_caption',
            
            # Reference fields
            'outbound_refs', 'inbound_refs',
            
            # Entity fields
            'regions', 'metrics', 'time_periods', 'dates',
            
            # JSON metadata fields that might contain nested data
            'metadata', 'table_metadata', 'structural_metadata', 'entity_metadata'
        ])
        
        logger.info(f"Querying fields: {output_fields}")
        
        # Optimized chunk retrieval with intelligent sampling
        logger.info("Fetching chunks from collection")
        
        # Calculate optimal limit - balance between over-sampling and query efficiency
        if sample_size >= total_entities:
            # If we need all or most entities, get them all
            limit = total_entities
        elif sample_size < 10:
            # For small samples, get more for better randomness
            limit = min(sample_size * 5, total_entities)
        else:
            # For larger samples, more conservative multiplier
            limit = min(sample_size * 2, total_entities)
        
        all_chunks = store.collection.query(
            expr="",  # Empty expression to get all
            output_fields=output_fields,
            limit=limit
        )
        
        logger.info(f"Retrieved {len(all_chunks)} chunks from collection")
        
        # Optimized sampling - avoid unnecessary work if we got exactly what we need
        if len(all_chunks) > sample_size:
            logger.info(f"Randomly sampling {sample_size} chunks from {len(all_chunks)} total chunks")
            # Use random.sample which is more efficient than manual shuffling
            all_chunks = random.sample(all_chunks, sample_size)
        elif len(all_chunks) < sample_size:
            logger.warning(f"Only retrieved {len(all_chunks)} chunks, less than requested {sample_size}")
        
        logger.info(f"Final sample size: {len(all_chunks)} chunks")
        
        # Add embedding dimension info if embeddings are included - single pass optimization
        if include_embeddings and all_chunks:
            # Pre-calculate to avoid repeated attribute access
            chunks_count = len(all_chunks)
            for i in range(chunks_count):
                embedding = all_chunks[i].get('embedding')
                if embedding:  # More efficient than checking key existence + truthiness
                    all_chunks[i]['embedding_dim'] = len(embedding)
        
         # Decode JSON-encoded metadata fields - optimized for minimal iterations
        # Pre-define excluded fields and JSON start chars as sets for O(1) lookup
        excluded_fields = {'chunk_id', 'doc_id', 'content', 'section_path'}
        json_start_chars = {'{', '['}
         
        # Add special handling for known metadata fields
        metadata_field_names = {
            'metadata', 'table_metadata', 'structural_metadata', 'entity_metadata',
            'outbound_refs', 'inbound_refs', 'column_headers', 'regions', 
            'metrics', 'time_periods', 'dates'
        }
        
        logger.info("Processing and decoding JSON metadata fields...")
        
        # Process all chunks in single optimized loop
        for chunk in all_chunks:
            # Collect keys to modify to avoid dict modification during iteration
            keys_to_decode = []
            for key, value in chunk.items():
                # Check if this is a JSON string that needs decoding
                if (isinstance(value, str) and 
                    key not in excluded_fields and 
                    value and 
                    (value[0] in json_start_chars or key in metadata_field_names)):
                    keys_to_decode.append(key)
            
            # Batch decode JSON strings - separate loop to avoid dict modification issues
            for key in keys_to_decode:
                try:
                    chunk[key] = json.loads(chunk[key])
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Keep as string if not valid JSON - silent fail for performance
                    continue
                    
            # Special handling for metadata field if it exists
            if 'metadata' in chunk and isinstance(chunk['metadata'], dict):
                # Extract table-specific fields from metadata if they exist
                metadata = chunk['metadata']
                
                # Copy table fields to top level for visibility
                for field in ['table_id', 'column_headers', 'table_title', 'table_caption']:
                    if field in metadata and field not in chunk:
                        chunk[field] = metadata[field]
                        
                # Copy entity fields to top level
                for field in ['regions', 'metrics', 'time_periods', 'dates']:
                    if field in metadata and field not in chunk:
                        chunk[field] = metadata[field]
        
        logger.info(f"Successfully extracted {len(all_chunks)} chunks")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error extracting chunks: {e}", exc_info=True)
        return []
    finally:
        # Disconnect from Milvus
        if 'store' in locals() and store.connected:
            store.disconnect()


def save_to_json(chunks: List[Dict[str, Any]], output_path: str = "sample_chunks.json"):
    """
    Save chunks to JSON file - optimized for performance.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to output JSON file
    """
    try:
        # Create output directory if it doesn't exist - single Path operation
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data structure once
        output_data = {
            "sample_size": len(chunks),
            "chunks": chunks
        }
        
        # Save to JSON with optimized settings
        with open(output_file, 'w', encoding='utf-8', buffering=8192) as f:  # Larger buffer for better I/O performance
            json.dump(output_data, f, indent=2, ensure_ascii=False, separators=(',', ': '))  # Faster separators
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving chunks to JSON: {e}")
        return False


def main():
    """Main function to extract and save sample chunks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract sample chunks from Milvus")
    parser.add_argument("--sample-size", type=int, default=100, 
                       help="Number of chunks to extract (default: 100)")
    parser.add_argument("--output", type=str, 
                       default=str(Path(__file__).parent.parent.parent.parent / "sample_chunks.json"),
                       help="Output JSON file path")
    parser.add_argument("--no-embeddings", action="store_true",
                       help="Exclude embedding vectors (smaller file size)")
    parser.add_argument("--metadata-only", action="store_true", 
                       help="Extract only metadata fields without content")
    
    args = parser.parse_args()
    
    logger.info(f"Starting sample chunk extraction")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Include embeddings: {not args.no_embeddings}")
    logger.info(f"Metadata only: {args.metadata_only}")
    logger.info(f"Output path: {args.output}")
    
    # Extract chunks
    chunks = extract_sample_chunks(
        sample_size=args.sample_size, 
        include_embeddings=not args.no_embeddings
    )
    
    if not chunks:
        logger.error("No chunks extracted")
        return
    
    # Remove content if metadata-only mode - optimized processing
    if args.metadata_only:
        logger.info("Removing content field for metadata-only extraction")
        for chunk in chunks:
            content = chunk.pop('content', None)  # More efficient than checking + deleting
            if content:
                # Keep just first 200 chars as preview - single operation
                chunk['content_preview'] = content[:200] + ("..." if len(content) > 200 else "")
    
    # Add extraction metadata - simplified and more efficient
    import datetime
    extraction_info = {
        "extraction_time": json.dumps({"timestamp": datetime.datetime.now().isoformat()}),
        "total_extracted": len(chunks),
        "include_embeddings": not args.no_embeddings,
        "metadata_only": args.metadata_only
    }
    
    # Save to JSON
    save_to_json_with_metadata(chunks, args.output, extraction_info)
    
    logger.info("Done!")


def save_to_json_with_metadata(chunks: List[Dict[str, Any]], output_path: str, 
                              extraction_info: Dict[str, Any]):
    """Save chunks with extraction metadata - optimized performance."""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data with metadata - single dict construction
        output_data = {
            "extraction_info": extraction_info,
            "sample_size": len(chunks),
            "chunks": chunks
        }
        
        # Optimized file writing with larger buffer and faster JSON settings
        with open(output_file, 'w', encoding='utf-8', buffering=16384) as f:  # Larger buffer for big files
            json.dump(output_data, f, indent=2, ensure_ascii=False, separators=(',', ': '))
        
        # Get file size efficiently in single stat call
        file_size = output_file.stat().st_size
        
        # Combined logging for efficiency
        logger.info(f"Saved {len(chunks)} chunks to {output_path} ({file_size / 1024 / 1024:.2f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving chunks to JSON: {e}")
        return False


if __name__ == "__main__":
    main()
