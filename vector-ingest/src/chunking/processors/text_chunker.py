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
        
        # Pre-compile sentence splitting regex for better performance
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
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
        
        metadata = metadata or {}
        
        # Handle dual JSON/MD format if metadata contains JSON data
        if 'json_data' in metadata:
            return self._process_dual_format(content, metadata)
        
        # Default processing for single format
        return self._process_single_format(content, metadata)
    
    def _process_dual_format(self, md_content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process content with both JSON and MD formats available."""
        json_data = metadata.get('json_data', {})
        
        # Prefer MD content for headings and structure
        if md_content.strip():
            chunks = self._create_chunks_from_markdown(md_content, metadata)
            
            # Enhance with JSON data if available
            if json_data and 'elements' in json_data:
                chunks = self._enhance_with_json_data(chunks, json_data)
            
            return chunks
        
        # Fallback to JSON if MD is empty
        if json_data and 'elements' in json_data:
            return self._create_chunks_from_json(json_data, metadata)
        
        return []
    
    def _process_single_format(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process content in single format."""
        return self._create_chunks_from_markdown(content, metadata)
    
    def _create_chunks_from_markdown(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from markdown content with structure awareness."""
        sentences = self._split_into_sentences(content)
        if not sentences:
            return []
        
        chunks = self._group_sentences_into_chunks(sentences)
        doc_id = metadata.get('doc_id', 'unknown')
        
        # Use list comprehension for better performance
        return [
            {
                'chunk_id': f"{doc_id}_chunk_{i+1}",
                'content': chunk_text,
                'word_count': len(chunk_text.split()),
                'doc_id': doc_id,
                'section_path': []
            }
            for i, chunk_text in enumerate(chunks)
        ]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure."""
        # Use pre-compiled regex and list comprehension for better performance
        sentences = self.sentence_pattern.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks based on word limits."""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        # Pre-calculate word counts for all sentences to avoid repeated splitting
        sentence_word_counts = [len(sentence.split()) for sentence in sentences]
        
        for sentence, sentence_words in zip(sentences, sentence_word_counts):
            # If adding this sentence would exceed max_words, finalize current chunk
            if current_word_count + sentence_words > self.max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                if overlap_text:
                    current_chunk = [overlap_text]
                    current_word_count = len(overlap_text.split())
                else:
                    current_chunk = []
                    current_word_count = 0
            
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
            # If we've reached target words, finalize chunk
            if current_word_count >= self.target_words:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                if overlap_text:
                    current_chunk = [overlap_text]
                    current_word_count = len(overlap_text.split())
                else:
                    current_chunk = []
                    current_word_count = 0
        
        # Add remaining content if it meets minimum requirements
        if current_chunk and current_word_count >= self.min_words:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_text(self, chunk_sentences: List[str]) -> str:
        """Extract overlap text from the end of current chunk."""
        if not chunk_sentences:
            return ""
        
        # Build overlap text more efficiently with list and join
        overlap_sentences = []
        word_count = 0
        
        for sentence in reversed(chunk_sentences):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= self.overlap_words:
                overlap_sentences.insert(0, sentence)
                word_count += sentence_words
            else:
                break
        
        return ' '.join(overlap_sentences)
    
    def _create_chunks_from_json(self, json_data: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from JSON structure data."""
        elements = json_data.get('elements', [])
        if not elements:
            return []
        
        # Group elements into chunks
        chunks = self._group_elements_into_chunks(elements)
        
        # Convert to chunk dictionaries
        chunk_dicts = []
        doc_id = metadata.get('doc_id', 'unknown')
        
        for i, chunk_elements in enumerate(chunks):
            chunk_content = self._elements_to_text(chunk_elements)
            if chunk_content.strip():
                chunk_id = f"{doc_id}_chunk_{i+1}"
                chunk_dicts.append({
                    'chunk_id': chunk_id,
                    'content': chunk_content,
                    'word_count': len(chunk_content.split()),
                    'doc_id': doc_id,
                    'section_path': []
                })
        
        return chunk_dicts
    
    def _group_elements_into_chunks(self, elements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group JSON elements into chunks based on word limits."""
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        # Pre-calculate word counts to avoid repeated operations
        element_word_counts = [(elem, len(elem.get('text', '').split())) for elem in elements]
        
        for element, element_words in element_word_counts:
            # If adding this element would exceed max_words, finalize current chunk
            if current_word_count + element_words > self.max_words and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_word_count = 0
            
            current_chunk.append(element)
            current_word_count += element_words
            
            # If we've reached target words, finalize chunk
            if current_word_count >= self.target_words:
                chunks.append(current_chunk)
                current_chunk = []
                current_word_count = 0
        
        # Add remaining elements
        if current_chunk and current_word_count >= self.min_words:
            chunks.append(current_chunk)
        
        return chunks
    
    def _elements_to_text(self, elements: List[Dict[str, Any]]) -> str:
        """Convert JSON elements to text content."""
        # Use generator expression with filter for better memory efficiency
        texts = (element.get('text', '').strip() for element in elements)
        return '\n\n'.join(text for text in texts if text)
    
    def _enhance_with_json_data(self, chunks: List[Dict[str, Any]], json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance MD-based chunks with JSON structure data."""
        # This is a placeholder for more sophisticated enhancement
        # In practice, you might align chunks with JSON elements, extract tables, etc.
        return chunks