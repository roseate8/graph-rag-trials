from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class StructuralMetadata(BaseModel):
    """Pydantic version of structural metadata for document chunks."""
    element_type: Optional[str] = None
    element_level: int = 0
    page_number: Optional[int] = None
    bbox_coords: Optional[Dict[str, float]] = None
    is_heading: bool = False
    matching_method: str = "none"
    matching_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in {
            'element_type': self.element_type,
            'element_level': self.element_level if self.element_level > 0 else None,
            'page_number': self.page_number,
            'bbox_coords': self.bbox_coords,
            'is_heading': self.is_heading if self.is_heading else None,
            'matching_method': self.matching_method,
            'matching_confidence': self.matching_confidence
        }.items() if v is not None}


class DocumentMetadata(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source_type: Literal["pdf", "docx", "html", "md", "txt"]
    author: Optional[str] = None
    date: Optional[str] = None
    page_count: Optional[int] = None


class Reference(BaseModel):
    ref_type: Literal["table", "figure", "appendix", "section"]
    label: str
    target_anchor: Optional[str] = None
    page: Optional[int] = None


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_type: Literal["text", "table", "figure"] = "text"
    page: Optional[int] = None
    section_path: List[str] = Field(default_factory=list)
    word_count: int
    
    # References - use Field for better performance
    outbound_refs: List[Reference] = Field(default_factory=list)
    inbound_refs: List[str] = Field(default_factory=list)
    
    # Entities (optional enrichment) - use Field for better performance  
    regions: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    time_periods: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    
    # Table-specific metadata
    table_id: Optional[str] = None
    column_headers: List[str] = Field(default_factory=list)
    table_title: Optional[str] = None
    table_shape: Optional[Dict[str, int]] = None  # {"rows": X, "cols": Y}
    
    # Organization entities (mapped from existing entity extraction)
    orgs: List[str] = Field(default_factory=list)
    
    # Time context for financial data
    time_context: Optional[Dict[str, str]] = None  # {"start": "...", "end": "...", "granularity": "..."}
    
    # Product versioning
    product_version: str = "v1"
    
    # File hierarchy path (comma-separated folder structure)
    folder_path: List[str] = Field(default_factory=list)
    
    # Structural metadata (Phase 1 enrichment)
    structural_metadata: Optional[StructuralMetadata] = None


class Chunk(BaseModel):
    metadata: ChunkMetadata
    content: str
    embedding: Optional[List[float]] = None


class DocumentStructure(BaseModel):
    toc_sections: List[Dict[str, Any]] = []
    headings: List[Dict[str, str]] = []
    tables: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []