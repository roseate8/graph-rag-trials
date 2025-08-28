from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    doc_id: str
    title: Optional[str] = None
    source_type: Literal["pdf", "docx", "html", "md"]
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
    chunk_type: Literal["text", "table", "figure"]
    page: Optional[int] = None
    section_path: List[str] = []
    word_count: int
    
    # References
    outbound_refs: List[Reference] = []
    inbound_refs: List[str] = []
    
    # Entities (optional enrichment)
    regions: List[str] = []
    metrics: List[str] = []
    time_periods: List[str] = []
    dates: List[str] = []
    
    # Table-specific metadata
    table_id: Optional[str] = None
    column_headers: List[str] = []
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