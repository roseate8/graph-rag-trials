from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


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
    table_caption: Optional[str] = None


class Chunk(BaseModel):
    metadata: ChunkMetadata
    content: str
    embedding: Optional[List[float]] = None


class DocumentStructure(BaseModel):
    toc_sections: List[Dict[str, Any]] = []
    headings: List[Dict[str, str]] = []
    tables: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []