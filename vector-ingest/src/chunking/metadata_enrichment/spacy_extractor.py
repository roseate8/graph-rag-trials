"""
Optimized metadata enrichment using spaCy for essential entity extraction.

Extracts 4 core entity types: organizations, locations, products, events.
Optimized for performance with early termination and efficient deduplication.
"""

import spacy
import logging
from typing import Dict, Any, Set, Tuple

logger = logging.getLogger(__name__)


class SpacyMetadataExtractor:
    """Optimized spaCy-based metadata extractor for 4 core entity types."""
    
    # Class-level constants for performance
    MAX_CONTENT_LENGTH = 1200
    ENTITY_LIMITS = {
        'organizations': 3, 'locations': 3, 'products': 3, 'events': 3
    }
    SKIP_PREFIXES = ('the ', 'and ', 'page ', 'item ', 'form ')
    LOCATION_LABELS = frozenset(['GPE', 'LOC'])
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize with spaCy model and performance optimizations."""
        self.logger = logging.getLogger(__name__ + '.SpacyMetadataExtractor')
        self.nlp = None
        try:
            self.nlp = spacy.load(model_name)
            # Disable unnecessary pipeline components for speed
            self.nlp.disable_pipes(['parser', 'tagger'])
            self.logger.info(f"Loaded optimized spaCy model: {model_name}")
        except OSError:
            self.logger.error(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
            raise
    
    def process_chunk_content(self, content: str) -> Dict[str, Any]:
        """Extract 4 essential entity types with optimized performance."""
        if not self.nlp or not content.strip():
            return self._empty_result()
        
        try:
            # Truncate content for performance (class constant)
            text = content[:self.MAX_CONTENT_LENGTH]
            doc = self.nlp(text)
            
            # Pre-allocate sets for O(1) deduplication
            entity_sets = {
                'organizations': set(),
                'locations': set(), 
                'products': set(),
                'events': set()
            }
            
            # Optimized single-pass entity extraction
            for ent in doc.ents:
                text_clean = ent.text.strip()
                if not text_clean:
                    continue
                    
                label = ent.label_
                text_len = len(text_clean)
                
                # Use optimized classification with early exit patterns
                if self._classify_entity(label, text_clean, text_len, entity_sets):
                    # Early termination if we have enough organizations (most common)
                    if len(entity_sets['organizations']) >= 5:
                        break
            
            # Post-process to fix common misclassifications before building result
            self._fix_product_misclassifications(entity_sets)
            
            # Convert to limited result lists
            return self._build_result(entity_sets)
            
        except Exception as e:
            self.logger.error(f"spaCy processing error: {e}")
            return self._empty_result()
    
    def _classify_entity(self, label: str, text_clean: str, text_len: int, 
                        entity_sets: Dict[str, Set[str]]) -> bool:
        """Classify entities using spaCy's default model capabilities."""
        # ORGANIZATIONS: Company names, regulatory bodies
        if label == 'ORG' and text_len > 3:
            if not text_clean.lower().startswith(self.SKIP_PREFIXES):
                entity_sets['organizations'].add(text_clean)
                return True
        
        # LOCATIONS: Geographic entities  
        elif label in self.LOCATION_LABELS and text_len > 2:
            entity_sets['locations'].add(text_clean)
        
        # PRODUCTS: Commercial products, software, brands, services
        elif label == 'PRODUCT' and text_len > 2:
            entity_sets['products'].add(text_clean)
        
        # EVENTS: Business events, meetings, announcements, incidents
        elif label == 'EVENT' and text_len > 3:
            entity_sets['events'].add(text_clean)
        
        return False
    
    def _fix_product_misclassifications(self, entity_sets: Dict[str, Set[str]]):
        """Fix common product misclassifications from organizations to products."""
        # Common product patterns that get misclassified as ORG
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


def create_spacy_extractor(model_name: str = "en_core_web_lg") -> SpacyMetadataExtractor:
    """Factory function to create simplified spaCy extractor."""
    return SpacyMetadataExtractor(model_name=model_name)


# Simple test
if __name__ == "__main__":
    extractor = create_spacy_extractor()
    
    sample_text = """
    Elastic N.V. reported revenue of $16.1 billion for fiscal year 2022. 
    The company operates in the United States and Netherlands.
    Apple Inc. announced the new iPhone 15 launch event.
    """
    
    result = extractor.process_chunk_content(sample_text)
    print("4-entity extraction result:")
    print(result)  # Expected: {"organizations": ["Elastic N.V.", "Apple Inc."], "locations": ["United States", "Netherlands"], "products": ["iPhone 15"], "events": ["launch event"]}