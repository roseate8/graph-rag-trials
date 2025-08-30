"""Document structure definitions and element types for content processing."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class ElementType(Enum):
    """Document element types for classification."""
    TITLE = "title"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    FIGURE = "figure"
    TABLE = "table"
    LIST = "list"
    CONTENT = "content"


@dataclass
class BoundingBox:
    """Represents bounding box coordinates for document elements."""
    x: float
    y: float
    width: float
    height: float
    page: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - optimized with asdict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoundingBox':
        """Create from dictionary - optimized with direct unpacking."""
        return cls(
            x=data.get('x', 0.0),
            y=data.get('y', 0.0),
            width=data.get('width', 0.0),
            height=data.get('height', 0.0),
            page=data.get('page', 0)
        )


@dataclass
class DocumentElement:
    """Represents a document structure element with content and metadata."""
    content: str
    element_type: ElementType
    level: int = 0
    bbox: Optional[BoundingBox] = None
    page: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    # Cache heading types set for O(1) lookup
    _HEADING_TYPES = frozenset([ElementType.TITLE, ElementType.SECTION, ElementType.SUBSECTION])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - optimized."""
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
        """Create from dictionary - optimized."""
        bbox = BoundingBox.from_dict(data['bbox']) if 'bbox' in data else None
        
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
        """Check if element is a heading - O(1) lookup."""
        return self.element_type in self._HEADING_TYPES
    
    @property
    def heading_prefix(self) -> str:
        """Get markdown heading prefix based on level."""
        return "#" * self.level if self.is_heading and self.level > 0 else ""


class DocumentStructure:
    """Document structure analysis and management."""
    
    # Optimized heading type mapping - use tuple for better performance
    HEADING_TYPES = (
        ElementType.SUBSECTION,  # default (index 0, for levels > 6)
        ElementType.TITLE,       # level 1
        ElementType.SECTION,     # level 2
        ElementType.SUBSECTION,  # level 3
        ElementType.SUBSECTION,  # level 4
        ElementType.SUBSECTION,  # level 5
        ElementType.SUBSECTION   # level 6
    )
    
    @staticmethod
    def classify_heading_type(level: int) -> ElementType:
        """Classify heading type based on level - O(1) lookup."""
        return DocumentStructure.HEADING_TYPES[level if 1 <= level <= 6 else 0]
    
    @staticmethod
    def extract_structure_info(elements: List[DocumentElement]) -> Dict[str, Any]:
        """Extract structural information - optimized with list comprehensions."""
        titles = []
        sections = []
        subsections = []
        content_types = []
        bounding_boxes = []
        pages = set()
        
        # Single pass through elements for better performance
        for element in elements:
            element_type = element.element_type
            
            if element_type == ElementType.TITLE:
                titles.append(element.content)
            elif element_type == ElementType.SECTION:
                sections.append(element.content)
            elif element_type == ElementType.SUBSECTION:
                subsections.append(element.content)
            
            content_types.append(element_type.value)
            
            if element.bbox:
                bounding_boxes.append(element.bbox.to_dict())
            
            pages.add(element.page)
        
        return {
            'titles': titles,
            'sections': sections,
            'subsections': subsections,
            'content_types': content_types,
            'bounding_boxes': bounding_boxes,
            'pages': sorted(pages)  # Return sorted list for consistency
        }
    
    @staticmethod
    def get_heading_hierarchy(elements: List[DocumentElement]) -> List[Dict[str, Any]]:
        """Extract heading hierarchy - optimized with list comprehension."""
        return [
            {
                'content': element.content,
                'level': element.level,
                'type': element.element_type.value,
                'page': element.page,
                'bbox': element.bbox.to_dict() if element.bbox else None
            }
            for element in elements if element.is_heading
        ]