"""
Optimized metadata enrichment using GLiNER for essential entity extraction.

Extracts 4 core entity types: organizations, locations, products, events.
Optimized for performance with efficient batch processing and entity mapping.
"""

import logging
from typing import Dict, Any, Set, List
import time

logger = logging.getLogger(__name__)


class GlinerMetadataExtractor:
    """GLiNER-based metadata extractor for 4 core entity types."""
    
    # Class-level constants for performance
    MAX_CONTENT_LENGTH = 1200
    ENTITY_LIMITS = {
        'organizations': 3, 'locations': 3, 'products': 3, 'events': 3
    }
    SKIP_PREFIXES = ('the ', 'and ', 'page ', 'item ', 'form ')
    
    # GLiNER label mappings to our 4 categories
    GLINER_LABELS = [
        "Organization", "Company", "Corporation", "Business",  # organizations
        "Location", "Place", "City", "Country", "State",      # locations  
        "Product", "Software", "Service", "Brand", "Tool",    # products
        "Event", "Meeting", "Conference", "Launch", "Announcement"  # events
    ]
    
    # Mapping GLiNER labels to our categories
    LABEL_MAPPING = {
        # Organizations
        "Organization": "organizations",
        "Company": "organizations", 
        "Corporation": "organizations",
        "Business": "organizations",
        
        # Locations
        "Location": "locations",
        "Place": "locations",
        "City": "locations", 
        "Country": "locations",
        "State": "locations",
        
        # Products
        "Product": "products",
        "Software": "products",
        "Service": "products", 
        "Brand": "products",
        "Tool": "products",
        
        # Events
        "Event": "events",
        "Meeting": "events",
        "Conference": "events",
        "Launch": "events", 
        "Announcement": "events"
    }
    
    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1"):
        """Initialize with GLiNER model and performance optimizations."""
        self.logger = logging.getLogger(__name__ + '.GlinerMetadataExtractor')
        self.model = None
        self.model_name = model_name
        
        try:
            # Lazy import to avoid dependency issues if GLiNER not installed
            from gliner import GLiNER
            
            self.logger.info(f"Loading GLiNER model: {model_name}")
            start_time = time.time()
            
            self.model = GLiNER.from_pretrained(model_name)
            
            load_time = time.time() - start_time
            self.logger.info(f"✅ GLiNER model loaded in {load_time:.2f}s")
            
        except ImportError:
            self.logger.error("GLiNER not installed. Install with: pip install gliner")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load GLiNER model '{model_name}': {e}")
            raise
    
    def process_chunk_content(self, content: str) -> Dict[str, Any]:
        """Extract 4 essential entity types with optimized performance."""
        if not self.model or not content.strip():
            return self._empty_result()
        
        try:
            # Truncate content for performance (class constant)
            text = content[:self.MAX_CONTENT_LENGTH]
            
            # Use GLiNER to predict entities with our defined labels
            entities = self.model.predict_entities(
                text, 
                self.GLINER_LABELS, 
                threshold=0.5  # Confidence threshold
            )
            
            # Pre-allocate sets for O(1) deduplication
            entity_sets = {
                'organizations': set(),
                'locations': set(), 
                'products': set(),
                'events': set()
            }
            
            # Process GLiNER results and map to our categories
            for entity in entities:
                entity_text = entity.get("text", "").strip()
                entity_label = entity.get("label", "")
                
                if not entity_text or len(entity_text) < 2:
                    continue
                
                # Skip common prefixes that aren't useful
                if entity_text.lower().startswith(self.SKIP_PREFIXES):
                    continue
                
                # Map GLiNER label to our category
                category = self.LABEL_MAPPING.get(entity_label)
                if category and category in entity_sets:
                    entity_sets[category].add(entity_text)
                    
                    # Early termination if we have enough organizations (most common)
                    if len(entity_sets['organizations']) >= 5:
                        break
            
            # Post-process to fix common misclassifications
            self._fix_entity_misclassifications(entity_sets)
            
            # Convert to limited result lists
            return self._build_result(entity_sets)
            
        except Exception as e:
            self.logger.error(f"GLiNER processing error: {e}")
            return self._empty_result()
    
    def _fix_entity_misclassifications(self, entity_sets: Dict[str, Set[str]]):
        """Fix common entity misclassifications between categories."""
        # Common product patterns that might get misclassified as organizations
        product_indicators = [
            'iphone', 'ipad', 'macbook', 'imac', 'airpods', 'apple watch',
            'windows', 'office', 'excel', 'word', 'powerpoint', 'teams', 'azure',
            'chrome', 'gmail', 'maps', 'youtube', 'android',
            'aws', 'ec2', 's3', 'lambda', 'prime',
            'model s', 'model 3', 'model x', 'model y', 'cybertruck',
            'elasticsearch', 'kibana', 'logstash'
        ]
        
        # Check organizations for misclassified products
        orgs_to_remove = set()
        products_to_add = set()
        
        for org in list(entity_sets['organizations']):
            org_lower = org.lower()
            
            # Check if this organization name matches known product patterns
            for pattern in product_indicators:
                if pattern in org_lower:
                    orgs_to_remove.add(org)
                    products_to_add.add(org)
                    break
            
            # Check for version/model patterns indicating products
            version_patterns = [' pro', ' plus', ' premium', ' enterprise', ' 365', ' 11', ' 15']
            if any(pattern in org_lower for pattern in version_patterns):
                # Don't move if it has clear company indicators
                company_indicators = [' inc', ' corp', ' ltd', ' llc', ' company', ' co.']
                if not any(indicator in org_lower for indicator in company_indicators):
                    orgs_to_remove.add(org)
                    products_to_add.add(org)
        
        # Apply the fixes
        for org in orgs_to_remove:
            entity_sets['organizations'].discard(org)
        
        for product in products_to_add:
            entity_sets['products'].add(product)
    
    def _build_result(self, entity_sets: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Build final result with entity limits applied."""
        return {
            "organizations": list(entity_sets['organizations'])[:self.ENTITY_LIMITS['organizations']],
            "locations": list(entity_sets['locations'])[:self.ENTITY_LIMITS['locations']],
            "products": list(entity_sets['products'])[:self.ENTITY_LIMITS['products']],
            "events": list(entity_sets['events'])[:self.ENTITY_LIMITS['events']]
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure for consistent interface."""
        return {
            "organizations": [], "locations": [], "products": [], "events": []
        }


def create_gliner_extractor(model_name: str = "urchade/gliner_medium-v2.1") -> GlinerMetadataExtractor:
    """Factory function to create GLiNER extractor."""
    return GlinerMetadataExtractor(model_name=model_name)


# Simple test
if __name__ == "__main__":
    extractor = create_gliner_extractor()
    
    sample_text = """
    Elastic N.V. reported revenue of $16.1 billion for fiscal year 2022. 
    The company operates in the United States and Netherlands.
    Apple Inc. announced the new iPhone 15 launch event in California.
    Microsoft Teams will host the quarterly business meeting next month.
    """
    
    result = extractor.process_chunk_content(sample_text)
    print("GLiNER 4-entity extraction result:")
    for category, entities in result.items():
        print(f"{category}: {entities}")
