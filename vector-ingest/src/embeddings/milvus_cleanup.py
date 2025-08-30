#!/usr/bin/env python3
"""
Milvus cleanup module for clearing embeddings before processing.
Can be easily plugged in/out of the main pipeline.
"""

import logging
from typing import Optional
from pathlib import Path

try:
    from .milvus_config import get_config
    from .milvus_store import MilvusVectorStore
except ImportError:
    # Handle standalone execution
    from milvus_config import get_config
    from milvus_store import MilvusVectorStore

logger = logging.getLogger(__name__)


class MilvusCleanup:
    """Handles cleanup of Milvus collections before processing."""
    
    def __init__(self, config_type: str = "production"):
        """Initialize cleanup with specified config."""
        self.config = get_config(config_type)
        self.store = MilvusVectorStore(self.config)
    
    def clear_collection(self, confirm: bool = False) -> bool:
        """
        Clear all data from the Milvus collection.
        
        Args:
            confirm: If True, skip confirmation prompt
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"🧹 Connecting to Milvus at {self.config.host}:{self.config.port}")
            
            if not self.store.connect():
                logger.error("❌ Failed to connect to Milvus")
                return False
            
            # Check if collection exists
            collection_name = self.config.collection_name
            if not self.store.collection_exists():
                logger.info(f"📂 Collection '{collection_name}' doesn't exist - nothing to clear")
                return True
            
            # Get current count
            current_count = self.store.get_entity_count()
            logger.info(f"📊 Current collection '{collection_name}' contains {current_count:,} entities")
            
            if current_count == 0:
                logger.info("✅ Collection is already empty")
                return True
            
            # Confirmation prompt (unless bypassed)
            if not confirm:
                logger.warning(f"⚠️  This will DELETE ALL {current_count:,} entities from collection '{collection_name}'")
                response = input("Continue? (y/N): ").strip().lower()
                if response != 'y':
                    logger.info("❌ Cleanup cancelled by user")
                    return False
            
            # Perform cleanup
            logger.info(f"🗑️  Clearing collection '{collection_name}'...")
            success = self.store.clear_collection()
            
            if success:
                # Verify cleanup
                new_count = self.store.get_entity_count()
                logger.info(f"✅ Successfully cleared collection. Entities: {current_count:,} → {new_count:,}")
                return True
            else:
                logger.error("❌ Failed to clear collection")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {str(e)}")
            return False
        finally:
            # Always disconnect
            if hasattr(self.store, 'client') and self.store.client:
                self.store.disconnect()
    
    def drop_collection(self, confirm: bool = False) -> bool:
        """
        Drop the entire collection (more thorough than clear).
        
        Args:
            confirm: If True, skip confirmation prompt
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"🧹 Connecting to Milvus at {self.config.host}:{self.config.port}")
            
            if not self.store.connect():
                logger.error("❌ Failed to connect to Milvus")
                return False
            
            collection_name = self.config.collection_name
            if not self.store.collection_exists():
                logger.info(f"📂 Collection '{collection_name}' doesn't exist - nothing to drop")
                return True
            
            current_count = self.store.get_entity_count()
            logger.info(f"📊 Collection '{collection_name}' contains {current_count:,} entities")
            
            # Confirmation prompt (unless bypassed)
            if not confirm:
                logger.warning(f"⚠️  This will DROP the entire collection '{collection_name}' ({current_count:,} entities)")
                logger.warning("⚠️  Collection structure and indexes will be destroyed")
                response = input("Continue? (y/N): ").strip().lower()
                if response != 'y':
                    logger.info("❌ Collection drop cancelled by user")
                    return False
            
            # Perform drop
            logger.info(f"🗑️  Dropping collection '{collection_name}'...")
            success = self.store.drop_collection()
            
            if success:
                logger.info(f"✅ Successfully dropped collection '{collection_name}'")
                return True
            else:
                logger.error("❌ Failed to drop collection")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during collection drop: {str(e)}")
            return False
        finally:
            # Always disconnect
            if hasattr(self.store, 'client') and self.store.client:
                self.store.disconnect()


def clear_milvus_collection(config_type: str = "production", confirm: bool = False) -> bool:
    """
    Convenience function to clear Milvus collection.
    
    Args:
        config_type: Configuration type to use
        confirm: If True, skip confirmation prompt
        
    Returns:
        bool: True if successful, False otherwise
    """
    cleanup = MilvusCleanup(config_type)
    return cleanup.clear_collection(confirm=confirm)


def drop_milvus_collection(config_type: str = "production", confirm: bool = False) -> bool:
    """
    Convenience function to drop Milvus collection.
    
    Args:
        config_type: Configuration type to use
        confirm: If True, skip confirmation prompt
        
    Returns:
        bool: True if successful, False otherwise
    """
    cleanup = MilvusCleanup(config_type)
    return cleanup.drop_collection(confirm=confirm)


def main():
    """CLI interface for cleanup operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up Milvus collections")
    parser.add_argument("--action", choices=["clear", "drop"], default="clear",
                       help="Action to perform (default: clear)")
    parser.add_argument("--config", default="production",
                       help="Configuration type (default: production)")
    parser.add_argument("--confirm", action="store_true",
                       help="Skip confirmation prompt")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== MILVUS CLEANUP ===")
    
    # Perform requested action
    if args.action == "clear":
        success = clear_milvus_collection(args.config, args.confirm)
    else:  # drop
        success = drop_milvus_collection(args.config, args.confirm)
    
    if success:
        print("✅ Cleanup completed successfully")
        return 0
    else:
        print("❌ Cleanup failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())