"""Text chunking processor for splitting text into coherent chunks."""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .base import BaseProcessor
from .doc_structure import DocumentElement, BoundingBox, ElementType, DocumentStructure


class TextChunker(BaseProcessor):
    """Splits text into chunks based on word limits while preserving sentence boundaries."""
    
    def __init__(
        self,
        target_words: int = 700,
        min_words: int = 20,
        overlap_words: int = 15,
        max_words: int = 800
    ):
        """
        Initialize the text chunker.
        
        Args:
            target_words: Target number of words per chunk (600-800 range)
            min_words: Minimum words required for a valid chunk
            overlap_words: Number of words to overlap between chunks (10-20)
            max_words: Maximum words allowed per chunk
        """
        super().__init__()
        self.target_words = target_words
        self.min_words = min_words
        self.overlap_words = overlap_words
        self.max_words = max_words
    
    def process(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process text content into chunks with document structure analysis.
        Supports dual JSON/MD format with precedence rules.
        
        Args:
            content: Text content to chunk (MD format preferred for headings)
            metadata: Optional metadata including JSON data and file paths
            
        Returns:
            List of chunk dictionaries with content, structure, and metadata
        """
        if not content or not content.strip():
            return []
        
        # Load JSON data if available for enhanced metadata
        json_data = self._load_json_data(metadata)
        
        # Identify document structure elements with dual format support
        elements = self._identify_dual_format_elements(content, metadata, json_data)
        
        # Group elements into chunks while preserving structure
        chunks = self._create_structure_aware_chunks(elements)
        
        result = []
        for i, chunk_elements in enumerate(chunks):
            chunk_content = self._combine_elements_content(chunk_elements)
            structure_info = self._extract_structure_info(chunk_elements)
            
            chunk_dict = {
                'content': chunk_content,
                'type': 'text',
                'chunk_id': i,
                'word_count': len(chunk_content.split()),
                'elements': [self._element_to_dict(elem) for elem in chunk_elements],
                'structure': structure_info,
                'metadata': metadata or {},
                'sources': {
                    'md_file': metadata.get('md_file') if metadata else None,
                    'json_file': metadata.get('json_file') if metadata else None
                }
            }
            result.append(chunk_dict)
        
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Pattern to split on sentence boundaries while preserving abbreviations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text.strip())
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence.split()) >= 3:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _create_chunks(self, sentences: List[str]) -> List[str]:
        """Create chunks from sentences while respecting word limits."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If adding this sentence would exceed max_words, finalize current chunk
            if current_word_count + sentence_words > self.max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_word_count = len(overlap_text.split()) if overlap_text else 0
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
            # If we've reached target words and have a good breaking point
            if current_word_count >= self.target_words:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_word_count = len(overlap_text.split()) if overlap_text else 0
        
        # Add remaining content if it meets minimum requirements
        if current_chunk and current_word_count >= self.min_words:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_text(self, chunk_sentences: List[str]) -> str:
        """Extract overlap text from the end of current chunk."""
        if not chunk_sentences:
            return ""
        
        # Take last few sentences to create overlap
        overlap_text = ""
        word_count = 0
        
        for sentence in reversed(chunk_sentences):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= self.overlap_words:
                overlap_text = sentence + " " + overlap_text if overlap_text else sentence
                word_count += sentence_words
            else:
                break
        
        return overlap_text.strip()
    
    def _extract_heading_context(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Extract heading and subheading context from metadata."""
        if not metadata:
            return ""
        
        context_parts = []
        
        # Add main heading
        if 'heading' in metadata:
            context_parts.append(f"# {metadata['heading']}")
        
        # Add subheading
        if 'subheading' in metadata:
            context_parts.append(f"## {metadata['subheading']}")
        
        return "\n".join(context_parts)
    
    def _add_heading_context(self, content: str, heading_context: str) -> str:
        """Add heading context to chunk content."""
        if not heading_context:
            return content
        
        return f"{heading_context}\n\n{content}"
    
    def _load_json_data(self, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Load JSON data if available for enhanced metadata."""
        if not metadata:
            return None
        
        # Try to get JSON data directly from metadata
        if 'json_data' in metadata:
            return metadata['json_data']
        
        # Try to load from JSON file path
        json_file = metadata.get('json_file')
        if json_file and Path(json_file).exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return None
    
    def _identify_dual_format_elements(self, content: str, metadata: Optional[Dict[str, Any]] = None, json_data: Optional[Dict[str, Any]] = None) -> List[DocumentElement]:
        """Identify document structure elements using both MD and JSON data with precedence rules."""
        elements = []
        lines = content.split('\n')
        
        # Build JSON lookup for bounding boxes, tables, and figures
        json_lookup = self._build_json_lookup(json_data) if json_data else {}
        
        current_page = metadata.get('page', 0) if metadata else 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            element = self._classify_dual_format_line(line, i, json_lookup, current_page)
            if element:
                elements.append(element)
        
        return elements
    
    def _build_json_lookup(self, json_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Build lookup table from JSON data for bounding boxes, tables, and figures."""
        lookup = {}
        
        # Process elements from JSON that have precedence for certain data
        elements = json_data.get('elements', [])
        for element in elements:
            content = element.get('content', '').strip()
            element_type = element.get('type', '')
            
            # JSON takes precedence for bounding boxes, tables, and figures
            if element_type in ['table', 'figure'] or 'bbox' in element:
                lookup[content] = element
        
        return lookup
    
    def _classify_dual_format_line(self, line: str, line_idx: int, json_lookup: Dict[str, Dict[str, Any]], page: int) -> Optional[DocumentElement]:
        """Classify a line using dual format with precedence rules."""
        # MD takes precedence for headings (H1, H2, H3, H4, etc.)
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            content = line.lstrip('#').strip()
            
            element_type = DocumentStructure.classify_heading_type(level)
            
            # Check JSON for additional metadata but use MD content
            json_element = json_lookup.get(content, {})
            bbox = self._extract_bbox_from_json(json_element)
            
            return DocumentElement(
                content=content,
                element_type=element_type,
                level=level,
                bbox=bbox,
                page=json_element.get('page', page),
                metadata={'source': 'md_primary', 'json_enhanced': bool(json_element)}
            )
        
        # Check JSON first for tables and figures (JSON takes precedence)
        json_element = json_lookup.get(line)
        if json_element:
            element_type_str = json_element.get('type', '')
            if element_type_str in ['table', 'figure']:
                bbox = self._extract_bbox_from_json(json_element)
                element_type = ElementType.TABLE if element_type_str == 'table' else ElementType.FIGURE
                
                return DocumentElement(
                    content=line,
                    element_type=element_type,
                    level=0,
                    bbox=bbox,
                    page=json_element.get('page', page),
                    metadata={'source': 'json_primary', 'enhanced_data': json_element}
                )
        
        # Fallback to MD-based classification for tables and figures
        if '|' in line and line.count('|') >= 2:
            return DocumentElement(
                content=line,
                element_type=ElementType.TABLE,
                level=0,
                bbox=None,
                page=page,
                metadata={'source': 'md_fallback'}
            )
        
        if re.search(r'\b(figure|fig|image|chart|graph)\s*\d*\b', line, re.IGNORECASE):
            return DocumentElement(
                content=line,
                element_type=ElementType.FIGURE,
                level=0,
                bbox=None,
                page=page,
                metadata={'source': 'md_fallback'}
            )
        
        # Check for list items
        if re.match(r'^[\*\-\+]\s+|^\d+\.\s+', line):
            bbox = self._extract_bbox_from_json(json_lookup.get(line, {}))
            return DocumentElement(
                content=line,
                element_type=ElementType.LIST,
                level=0,
                bbox=bbox,
                page=page,
                metadata={'source': 'md_primary'}
            )
        
        # Default to paragraph with JSON enhancement if available
        json_element = json_lookup.get(line, {})
        bbox = self._extract_bbox_from_json(json_element)
        
        return DocumentElement(
            content=line,
            element_type=ElementType.PARAGRAPH,
            level=0,
            bbox=bbox,
            page=json_element.get('page', page),
            metadata={'source': 'md_primary', 'json_enhanced': bool(json_element)}
        )
    
    def _extract_bbox_from_json(self, json_element: Dict[str, Any]) -> Optional[BoundingBox]:
        """Extract bounding box from JSON element data."""
        if not json_element or 'bbox' not in json_element:
            return None
        
        bbox_data = json_element['bbox']
        if isinstance(bbox_data, dict):
            return BoundingBox(
                x=bbox_data.get('x', 0),
                y=bbox_data.get('y', 0),
                width=bbox_data.get('width', 0),
                height=bbox_data.get('height', 0),
                page=bbox_data.get('page', 0)
            )
        
        return None
    
    def process_file_pair(self, md_file_path: str, json_file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process a paired MD/JSON file combination."""
        # Load MD content
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
        except IOError:
            raise ValueError(f"Could not read MD file: {md_file_path}")
        
        # Prepare metadata with file paths
        metadata = {
            'md_file': md_file_path,
            'json_file': json_file_path
        }
        
        # Load JSON data if provided
        if json_file_path:
            json_data = self._load_json_data({'json_file': json_file_path})
            if json_data:
                metadata['json_data'] = json_data
        
        return self.process(md_content, metadata)
    
    def _create_structure_aware_chunks(self, elements: List[DocumentElement]) -> List[List[DocumentElement]]:
        """Create chunks while respecting document structure boundaries."""
        if not elements:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for element in elements:
            element_words = len(element.content.split())
            
            # Check if we should start a new chunk
            should_break = False
            
            # Always break on titles (H1)
            if element.element_type == ElementType.TITLE:
                should_break = current_chunk and current_word_count > 0
            
            # Break on sections (H2) if we're getting close to target
            elif element.element_type == ElementType.SECTION and current_word_count > self.target_words * 0.7:
                should_break = True
            
            # Break if adding this element would exceed max words
            elif current_word_count + element_words > self.max_words and current_chunk:
                should_break = True
            
            if should_break:
                if current_word_count >= self.min_words:
                    chunks.append(current_chunk.copy())
                
                # Start new chunk with overlap if applicable
                current_chunk = self._create_overlap_chunk(current_chunk)
                current_word_count = sum(len(elem.content.split()) for elem in current_chunk)
            
            current_chunk.append(element)
            current_word_count += element_words
            
            # Check if we've reached target and can break at a good boundary
            if (current_word_count >= self.target_words and 
                element.element_type in [ElementType.SECTION, ElementType.SUBSECTION, ElementType.PARAGRAPH]):
                chunks.append(current_chunk.copy())
                current_chunk = self._create_overlap_chunk(current_chunk)
                current_word_count = sum(len(elem.content.split()) for elem in current_chunk)
        
        # Add remaining elements if they meet minimum requirements
        if current_chunk and current_word_count >= self.min_words:
            chunks.append(current_chunk)
        
        return chunks
    
    def _create_overlap_chunk(self, chunk: List[DocumentElement]) -> List[DocumentElement]:
        """Create overlap elements for the next chunk."""
        if not chunk:
            return []
        
        overlap_elements = []
        word_count = 0
        
        # Take elements from the end to create overlap
        for element in reversed(chunk):
            element_words = len(element.content.split())
            if word_count + element_words <= self.overlap_words:
                overlap_elements.insert(0, element)
                word_count += element_words
            else:
                break
        
        return overlap_elements
    
    def _combine_elements_content(self, elements: List[DocumentElement]) -> str:
        """Combine content from multiple elements into a single string."""
        content_parts = []
        
        for element in elements:
            if element.is_heading:
                # Add appropriate markdown formatting for headings
                prefix = element.heading_prefix or '#'
                content_parts.append(f"{prefix} {element.content}")
            else:
                content_parts.append(element.content)
        
        return '\n\n'.join(content_parts)
    
    def _extract_structure_info(self, elements: List[DocumentElement]) -> Dict[str, Any]:
        """Extract structural information from chunk elements."""
        return DocumentStructure.extract_structure_info(elements)
    
    def _element_to_dict(self, element: DocumentElement) -> Dict[str, Any]:
        """Convert DocumentElement to dictionary representation."""
        return element.to_dict()