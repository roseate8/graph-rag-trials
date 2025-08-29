"""Document structure definitions and element types for content processing."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ElementType(Enum):
    """Document element types for classification."""
    TITLE = "title"              # H1-level headings and document titles
    SECTION = "section"          # H2-level headings or major topical areas within the document
    SUBSECTION = "subsection"    # H3+ level headings
    PARAGRAPH = "paragraph"      # Text paragraphs
    FIGURE = "figure"           # Figures and images
    TABLE = "table"             # Tables and structured data
    LIST = "list"               # Lists (ordered and unordered)
    CONTENT = "content"         # General content


@dataclass
class BoundingBox:
    """Represents bounding box coordinates for document elements."""
    x: float
    y: float
    width: float
    height: float
    page: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'page': self.page
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        """Create from dictionary representation."""
        return cls(
            x=data.get('x', 0),
            y=data.get('y', 0),
            width=data.get('width', 0),
            height=data.get('height', 0),
            page=data.get('page', 0)
        )


@dataclass
class DocumentElement:
    """Represents a document structure element with content and metadata."""
    content: str
    element_type: ElementType
    level: int = 0  # heading level (1-6) or 0 for non-heading elements
    bbox: Optional[BoundingBox] = None
    page: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'content': self.content,
            'type': self.element_type.value,
            'level': self.level,
            'page': self.page
        }
        
        if self.bbox:
            result['bbox'] = self.bbox.to_dict()
        
        if self.metadata:
            result['metadata'] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentElement':
        """Create from dictionary representation."""
        bbox = None
        if 'bbox' in data:
            bbox = BoundingBox.from_dict(data['bbox'])
        
        return cls(
            content=data['content'],
            element_type=ElementType(data['type']),
            level=data.get('level', 0),
            bbox=bbox,
            page=data.get('page', 0),
            metadata=data.get('metadata')
        )
    
    @property
    def is_heading(self) -> bool:
        """Check if element is a heading."""
        return self.element_type in [ElementType.TITLE, ElementType.SECTION, ElementType.SUBSECTION]
    
    @property
    def heading_prefix(self) -> str:
        """Get markdown heading prefix based on level."""
        if not self.is_heading or self.level <= 0:
            return ""
        return "#" * self.level


class DocumentStructure:
    """Document structure analysis and management."""
    
    # Element type mapping for classification
    HEADING_TYPES = {
        1: ElementType.TITLE,
        2: ElementType.SECTION,
        3: ElementType.SUBSECTION,
        4: ElementType.SUBSECTION,
        5: ElementType.SUBSECTION,
        6: ElementType.SUBSECTION
    }
    
    @staticmethod
    def classify_heading_type(level: int) -> ElementType:
        """Classify heading type based on level."""
        return DocumentStructure.HEADING_TYPES.get(level, ElementType.SUBSECTION)
    
    @staticmethod
    def extract_structure_info(elements: List[DocumentElement]) -> Dict[str, Any]:
        """Extract structural information from elements."""
        structure_info = {
            'titles': [],           # H1-level headings and document titles
            'sections': [],         # H2-level headings or major topical areas
            'subsections': [],      # H3+ level headings
            'content_types': [],    # All element types present
            'bounding_boxes': [],   # Bounding box coordinates
            'pages': set()          # Page numbers
        }
        
        for element in elements:
            if element.element_type == ElementType.TITLE:
                structure_info['titles'].append(element.content)
            elif element.element_type == ElementType.SECTION:
                structure_info['sections'].append(element.content)
            elif element.element_type == ElementType.SUBSECTION:
                structure_info['subsections'].append(element.content)
            
            structure_info['content_types'].append(element.element_type.value)
            
            if element.bbox:
                structure_info['bounding_boxes'].append(element.bbox.to_dict())
            
            structure_info['pages'].add(element.page)
        
        structure_info['pages'] = list(structure_info['pages'])
        return structure_info
    
    @staticmethod
    def get_heading_hierarchy(elements: List[DocumentElement]) -> List[Dict[str, Any]]:
        """Extract heading hierarchy from elements."""
        headings = []
        
        for element in elements:
            if element.is_heading:
                headings.append({
                    'content': element.content,
                    'level': element.level,
                    'type': element.element_type.value,
                    'page': element.page,
                    'bbox': element.bbox.to_dict() if element.bbox else None
                })
        
        return headings