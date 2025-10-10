#!/usr/bin/env python3
"""
Check all collections in Milvus database and display their statistics.
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add paths
VECTOR_INGEST_PATH = Path(__file__).parent / "vector-ingest" / "src"
sys.path.insert(0, str(VECTOR_INGEST_PATH))

from pymilvus import connections, utility
from embeddings.milvus_config import get_config


def check_milvus_collections():
    """Check all collections in Milvus and display their statistics."""
    print("=" * 80)
    print("Milvus Collections Statistics")
    print("=" * 80)

    try:
        # Connect to Milvus
        config = get_config()
        print(f"\nConnecting to Milvus at {config.host}:{config.port}...")

        connections.connect(
            alias="default",
            host=config.host,
            port=config.port
        )
        print("Connected successfully")

        # List all collections
        print("\n" + "=" * 80)
        collections = utility.list_collections()
        print(f"\nTotal Collections: {len(collections)}")
        print("=" * 80)

        if not collections:
            print("\nNo collections found in Milvus database.")
            return

        # Get details for each collection
        for i, collection_name in enumerate(collections, 1):
            print(f"\n[{i}] Collection: {collection_name}")
            print("-" * 80)

            try:
                from pymilvus import Collection
                collection = Collection(collection_name)

                # Load collection to get accurate count
                collection.load()

                # Get statistics
                num_entities = collection.num_entities

                # Get schema information
                schema = collection.schema
                print(f"  Description: {schema.description or 'N/A'}")
                print(f"  Total Entities: {num_entities:,}")

                # Get field information
                print("\n  Fields:")
                for field in schema.fields:
                    print(f"    - {field.name}")
                    print(f"      Type: {field.dtype}")
                    if hasattr(field, 'params') and field.params:
                        if 'dim' in field.params:
                            print(f"      Dimension: {field.params['dim']}")
                        print(f"      Params: {field.params}")
                    if field.is_primary:
                        print(f"      Primary Key: Yes")
                    if hasattr(field, 'auto_id') and field.auto_id:
                        print(f"      Auto ID: Yes")

                # Get index information
                indexes = collection.indexes
                if indexes:
                    print("\n  Indexes:")
                    for index in indexes:
                        print(f"    - Field: {index.field_name}")
                        print(f"      Type: {index.params.get('index_type', 'N/A')}")
                        print(f"      Metric: {index.params.get('metric_type', 'N/A')}")
                        if 'params' in index.params:
                            print(f"      Params: {index.params['params']}")

                # Get loaded status
                print(f"\n  Loaded: {collection.is_loaded}")

            except Exception as e:
                print(f"  ERROR: Failed to get collection details: {e}")

        print("\n" + "=" * 80)

        # Disconnect
        connections.disconnect("default")
        print("\nDisconnected from Milvus")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_milvus_collections()
