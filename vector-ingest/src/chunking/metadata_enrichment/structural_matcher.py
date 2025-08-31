import logging
import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

from .models import MatchingResult, MatchingMethod, EnrichmentContext
from ..processors.doc_structure import DocumentElement, ElementType, BoundingBox
from ..models import StructuralMetadata

logger = logging.getLogger(__name__)


class StructuralMatcher:
    def __init__(self, content_threshold: float = 0.7, position_tolerance: int = 2):
        self.content_threshold = content_threshold
        self.position_tolerance = position_tolerance
        self._element_cache: Dict[str, List[DocumentElement]] = {}
        
    def match_chunks_to_structure(self, chunks: List[Any], context: EnrichmentContext) -> List[MatchingResult]:
        if not context.has_structure_data():
            return [MatchingResult(chunk_id=chunk.metadata.chunk_id) for chunk in chunks]
        
        elements = self._load_document_elements(context)
        if not elements:
            return [MatchingResult(chunk_id=chunk.metadata.chunk_id) for chunk in chunks]
        
        # Pre-normalize element content for O(1) lookup
        element_lookup = {}
        for i, element in enumerate(elements):
            if element.content:
                normalized = re.sub(r'\s+', ' ', element.content.lower().strip())
                element_lookup[normalized] = (i, element)
        
        results = []
        for chunk in chunks:
            result = self._match_single_chunk(chunk, elements, element_lookup)
            results.append(result)
        
        matched_count = sum(1 for r in results if r.has_match)
        logger.info(f"Matched {matched_count}/{len(chunks)} chunks")
        return results
    
    def _load_document_elements(self, context: EnrichmentContext) -> List[DocumentElement]:
        if context.doc_id in self._element_cache:
            return self._element_cache[context.doc_id]
        
        elements = []
        
        # Process JSON elements first (actual document content) - optimized
        if context.json_elements:
            logger.info(f"🔍 Processing JSON elements: {len(context.json_elements)}")
            text_elements = 0
            
            # Pre-allocate elements list for better performance
            json_elements = context.json_elements
            elements_to_add = []
            
            for i, item in enumerate(json_elements):
                if isinstance(item, dict):
                    text = item.get('text')
                    if text:
                        content = text.strip()
                        if content and len(content) > 5:  # Skip very short content
                            text_elements += 1
                            # Cache level calculation
                            level = content.count('#') if content.startswith('#') else item.get('level', 0)
                            
                            # Batch create elements for better memory efficiency
                            elements_to_add.append(DocumentElement(
                                content=content,
                                element_type=self._classify_element_type(level),
                                level=level,
                                page=self._get_page_number(item),
                                bbox=self._extract_bbox_from_prov(item)
                            ))
                            
                            # Limit processing for performance
                            if text_elements >= 3000:  # Reasonable limit
                                logger.info(f"🔍 Reached element processing limit ({text_elements})")
                                break
            
            # Batch extend for better performance than individual appends
            elements.extend(elements_to_add)
        
        # Debug: Show some sample elements  
        if elements:
            logger.info(f"🔍 Created {len(elements)} document elements")
            for i, elem in enumerate(elements[:3]):  # Show first 3
                logger.info(f"🔍   Element {i}: '{elem.content[:50]}...' (type: {elem.element_type})")
        
        self._element_cache[context.doc_id] = elements
        return elements
    
    def _get_page_number(self, item: Dict) -> int:
        """Extract page number from item, checking multiple possible locations."""
        # Check direct page fields
        if 'page' in item:
            return item['page']
        if 'page_no' in item:
            return item['page_no']
        
        # Check prov array for page number
        prov = item.get('prov')
        if prov and isinstance(prov, list) and len(prov) > 0:
            first_prov = prov[0]
            if isinstance(first_prov, dict):
                return first_prov.get('page_no', first_prov.get('page', 0))
        
        return 0
    
    def _extract_bbox_from_prov(self, item: Dict) -> Optional[BoundingBox]:
        """Extract bounding box from prov array structure."""
        prov = item.get('prov')
        if not prov or not isinstance(prov, list) or len(prov) == 0:
            return None
        
        first_prov = prov[0]
        if not isinstance(first_prov, dict):
            return None
        
        bbox_data = first_prov.get('bbox')
        if not bbox_data:
            return None
        
        return BoundingBox(
            x=bbox_data.get('l', bbox_data.get('x', 0.0)),
            y=bbox_data.get('t', bbox_data.get('y', 0.0)),
            width=bbox_data.get('r', 0.0) - bbox_data.get('l', 0.0),
            height=bbox_data.get('b', 0.0) - bbox_data.get('t', 0.0),
            page=first_prov.get('page_no', first_prov.get('page', 0))
        )
    
    def _extract_bbox(self, bbox_data: Optional[Dict]) -> Optional[BoundingBox]:
        """Legacy bbox extraction for backward compatibility."""
        if not bbox_data:
            return None
        return BoundingBox(
            x=bbox_data.get('l', bbox_data.get('x', 0.0)),
            y=bbox_data.get('t', bbox_data.get('y', 0.0)),
            width=bbox_data.get('width', 0.0),
            height=bbox_data.get('height', 0.0),
            page=bbox_data.get('page', 0)
        )
    
    def _match_single_chunk(self, chunk: Any, elements: List[DocumentElement], 
                           element_lookup: Dict) -> MatchingResult:
        chunk_id = chunk.metadata.chunk_id
        chunk_content = chunk.content.strip()
        chunk_page = getattr(chunk.metadata, 'page', None)
        
        # Debug first chunk only
        if chunk_id.endswith('_chunk_1'):
            logger.info(f"🔍 Matching chunk_1 content: '{chunk_content[:100]}...'")
            logger.info(f"🔍 Available {len(element_lookup)} elements in lookup")
        
        # Fast exact match using lookup table
        chunk_normalized = re.sub(r'\s+', ' ', chunk_content.lower().strip())
        if chunk_normalized in element_lookup:
            i, element = element_lookup[chunk_normalized]
            return MatchingResult(
                chunk_id=chunk_id,
                matched_element_id=f"element_{i}",
                method=MatchingMethod.CONTENT_EXACT,
                confidence=1.0,
                element_data=self._extract_element_data(element)
            )
        
        # Fast containment check for headings (optimized with early termination)
        chunk_len = len(chunk_normalized)
        if chunk_len == 0:  # Handle empty chunk case
            return MatchingResult(chunk_id=chunk_id)
            
        for normalized, (i, element) in element_lookup.items():
            normalized_len = len(normalized)
            # Optimized: skip if too short or too long compared to chunk (safe division)
            max_normalized_len = max(chunk_len // 2, 20)  # Prevent division issues
            if (10 <= normalized_len <= max_normalized_len and 
                normalized in chunk_normalized):
                return MatchingResult(
                    chunk_id=chunk_id,
                    matched_element_id=f"element_{i}",
                    method=MatchingMethod.CONTENT_EXACT,
                    confidence=0.9,
                    element_data=self._extract_element_data(element)
                )
        
        # Optimized fuzzy matching with early termination and limits
        best_confidence = 0.0
        best_element_idx = None
        checked_count = 0
        max_checks = min(50, len(elements))  # Limit fuzzy matching attempts
        
        for i, element in enumerate(elements):
            if checked_count >= max_checks:
                break
                
            if not element.content:
                continue
                
            element_len = len(element.content)
            # More restrictive length filter for performance
            if element_len > 150 or element_len < 10:
                continue
            
            checked_count += 1
            
            # Pre-normalize element content once for reuse
            element_normalized = re.sub(r'\s+', ' ', element.content.lower().strip())
            
            # Quick length-based filter before expensive similarity calculation
            len_ratio = min(len(chunk_normalized), len(element_normalized)) / max(len(chunk_normalized), len(element_normalized))
            if len_ratio < 0.3:  # Skip if lengths are too different
                continue
            
            # Use faster similarity calculation for short strings
            if len(element_normalized) < 50:
                # Simple containment check for very short elements
                if element_normalized in chunk_normalized or chunk_normalized in element_normalized:
                    similarity = 0.8
                else:
                    similarity = SequenceMatcher(None, chunk_normalized, element_normalized).ratio()
            else:
                similarity = SequenceMatcher(None, chunk_normalized, element_normalized).ratio()
            
            if similarity > best_confidence and similarity >= self.content_threshold:
                best_confidence = similarity
                best_element_idx = i
                # Early termination if we find a very good match
                if similarity > 0.95:
                    break
        
        if best_element_idx is not None:
            element = elements[best_element_idx]
            return MatchingResult(
                chunk_id=chunk_id,
                matched_element_id=f"element_{best_element_idx}",
                method=MatchingMethod.CONTENT_FUZZY,
                confidence=best_confidence,
                element_data=self._extract_element_data(element)
            )
        
        # Optimized position matching as fallback (with early termination)
        if chunk_page is not None:
            for i, element in enumerate(elements[:20]):  # Limit position search
                if (element.page is not None and 
                    abs(chunk_page - element.page) <= self.position_tolerance):
                    confidence = 0.6 if chunk_page == element.page else 0.4
                    return MatchingResult(
                        chunk_id=chunk_id,
                        matched_element_id=f"element_{i}",
                        method=MatchingMethod.POSITION_PAGE,
                        confidence=confidence,
                        element_data=self._extract_element_data(element)
                    )
        
        return MatchingResult(chunk_id=chunk_id)
    
    def _extract_element_data(self, element: DocumentElement) -> Dict[str, Any]:
        data = {
            'element_type': element.element_type.value if element.element_type else None,
            'level': element.level,
            'page': element.page,
            'is_heading': element.is_heading
        }
        if element.bbox:
            data['bbox'] = element.bbox.to_dict()
        return data
    
    def _classify_element_type(self, level: int) -> ElementType:
        return {1: ElementType.TITLE, 2: ElementType.SECTION}.get(level, 
                ElementType.SUBSECTION if level >= 3 else ElementType.PARAGRAPH)
    
    def create_structural_metadata(self, matching_result: MatchingResult) -> Optional[StructuralMetadata]:
        if not matching_result.has_match or not matching_result.element_data:
            return None
        
        data = matching_result.element_data
        bbox_data = data.get('bbox')
        
        return StructuralMetadata(
            element_type=data.get('element_type'),
            element_level=data.get('level', 0),
            page_number=data.get('page'),
            bbox_coords={
                'x': bbox_data.get('x', 0.0), 'y': bbox_data.get('y', 0.0),
                'width': bbox_data.get('width', 0.0), 'height': bbox_data.get('height', 0.0)
            } if bbox_data else None,
            is_heading=data.get('is_heading', False),
            matching_method=matching_result.method.value,
            matching_confidence=matching_result.confidence
        )