from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class MatchingMethod(Enum):
    CONTENT_EXACT = "exact"
    CONTENT_FUZZY = "fuzzy"  
    POSITION_PAGE = "page"
    NO_MATCH = "none"


@dataclass
class MatchingResult:
    chunk_id: str
    matched_element_id: Optional[str] = None
    method: MatchingMethod = MatchingMethod.NO_MATCH
    confidence: float = 0.0
    element_data: Optional[Dict[str, Any]] = None
    
    @property
    def has_match(self) -> bool:
        return self.confidence > 0.3


@dataclass 
class StructuralMetadata:
    element_type: Optional[str] = None
    element_level: int = 0
    page_number: Optional[int] = None
    bbox_coords: Optional[Dict[str, float]] = None
    is_heading: bool = False
    matching_method: MatchingMethod = MatchingMethod.NO_MATCH
    matching_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            'element_type': self.element_type,
            'element_level': self.element_level if self.element_level > 0 else None,
            'page_number': self.page_number,
            'bbox_coords': self.bbox_coords,
            'is_heading': self.is_heading if self.is_heading else None,
            'matching_method': self.matching_method.value,
            'matching_confidence': self.matching_confidence
        }.items() if v is not None}


@dataclass
class EnrichmentContext:
    doc_id: str
    document_structure: Optional[Any] = None
    json_elements: Optional[List[Dict[str, Any]]] = None
    md_content: Optional[str] = None
    total_chunks: int = 0
    
    def has_structure_data(self) -> bool:
        return self.document_structure is not None or self.json_elements is not None