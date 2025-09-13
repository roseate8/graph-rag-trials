"""Text chunking processor for splitting text into coherent chunks."""

import re
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .base import BaseProcessor
from .doc_structure import DocumentElement, BoundingBox, ElementType, DocumentStructure


class TokenEstimator:
    """Token counting with caching and batch processing."""
    
    def __init__(self, max_cache_size: int = 10000):
        self._tokenizer = None
        self._token_cache = {}
        self._max_cache_size = max_cache_size
        self._word_split_cache = {}
    
    def _get_tokenizer(self):
        """Lazy load tokenizer to avoid import overhead."""
        if self._tokenizer is None:
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._tokenizer = None
        return self._tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text with optimized caching."""
        if not text:
            return 0
            
        text_hash = hash(text)
        if text_hash in self._token_cache:
            return self._token_cache[text_hash]
        
        if len(self._token_cache) >= self._max_cache_size:
            self._token_cache.clear()  # Simple cache reset when full
        
        tokenizer = self._get_tokenizer()
        if tokenizer:
            token_count = len(tokenizer.encode(text))
        else:
            if text_hash in self._word_split_cache:
                words, chars = self._word_split_cache[text_hash]
            else:
                words = len(text.split())
                chars = len(text)
                if len(self._word_split_cache) < self._max_cache_size // 2:
                    self._word_split_cache[text_hash] = (words, chars)
            token_count = int(words * 1.3 + chars * 0.1)
        
        self._token_cache[text_hash] = token_count
        return token_count
    
    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Efficiently count tokens for multiple texts."""
        if not texts:
            return []
            
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if not text:
                results.append(0)
                continue
                
            text_hash = hash(text)
            if text_hash in self._token_cache:
                results.append(self._token_cache[text_hash])
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        if not uncached_texts:
            return results
        
        # Batch process uncached texts
        tokenizer = self._get_tokenizer()
        if tokenizer:
            # Use batch encoding if available
            try:
                encoded = tokenizer.encode_batch(uncached_texts)
                token_counts = [len(enc) for enc in encoded]
            except AttributeError:
                token_counts = [len(tokenizer.encode(text)) for text in uncached_texts]
        else:
            token_counts = []
            for text in uncached_texts:
                words = len(text.split())
                chars = len(text)
                token_counts.append(int(words * 1.3 + chars * 0.1))
        
        # Update cache and results
        for i, (text, count) in enumerate(zip(uncached_texts, token_counts)):
            text_hash = hash(text)
            if len(self._token_cache) < self._max_cache_size:
                self._token_cache[text_hash] = count
            results[uncached_indices[i]] = count
        
        return results


class TextChunker(BaseProcessor):
    """Splits text into chunks based on token limits while preserving sentence boundaries."""
    
    def __init__(
        self,
        target_tokens: int = 340,  # ~440 tokens ideally, but targeting middle of range
        min_tokens: int = 50,
        overlap_percentage: float = 0.20,  # 20% overlap
        max_tokens_soft: int = 450,
        max_tokens_hard: int = 512
    ):
        """
        Initialize the text chunker with token-based limits.
        
        Args:
            target_tokens: Target number of tokens per chunk (256-448 range, ideally ~440)
            min_tokens: Minimum tokens required for a valid chunk (50 tokens)
            overlap_percentage: Percentage of overlap between chunks (~20%)
            max_tokens_soft: Soft maximum tokens per chunk (450 tokens)
            max_tokens_hard: Hard maximum tokens per chunk (512 tokens)
        """
        super().__init__()
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.overlap_percentage = overlap_percentage
        self.max_tokens_soft = max_tokens_soft
        self.max_tokens_hard = max_tokens_hard
        
        # Pre-compile sentence splitting regex for better performance
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        # Initialize token estimator
        self.token_estimator = TokenEstimator()
        
        # Cache for content analysis to avoid repeated processing
        self._content_analysis_cache = {}
    
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
        
        # Extract heading structure for prepending
        headings_info = self._extract_heading_structure(content)
        
        chunks = self._group_sentences_into_chunks(sentences)
        if not chunks:
            return []
        
        doc_id = metadata.get('doc_id', 'unknown')
        
        # Pre-calculate all final contents for batch token counting
        final_contents = []
        chunk_headings_list = []
        
        for chunk_text in chunks:
            chunk_headings = self._find_relevant_headings(chunk_text, headings_info, content)
            chunk_headings_list.append(chunk_headings)
            final_content = self._prepend_headings_to_chunk(chunk_text, chunk_headings)
            final_contents.append(final_content)
        
        # Batch count tokens for all final contents
        token_counts = self.token_estimator.count_tokens_batch(final_contents)
        
        # Build result chunks
        processed_chunks = []
        for i, (final_content, token_count, chunk_headings) in enumerate(
            zip(final_contents, token_counts, chunk_headings_list)
        ):
            processed_chunks.append({
                'chunk_id': f"{doc_id}_chunk_{i+1}",
                'content': final_content,
                'token_count': token_count,
                'word_count': len(final_content.split()),
                'doc_id': doc_id,
                'section_path': [h['text'] for h in chunk_headings],
                'heading_context': chunk_headings
            })
        
        return processed_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving headings and structure."""
        lines = text.strip().split('\n')
        sentences = []
        current_sentence = ""
        
        for line in lines:
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
                
            if line_stripped.startswith('#'):
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                sentences.append(line_stripped)
            else:
                current_sentence = (current_sentence + " " + line_stripped) if current_sentence else line_stripped
        
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Split regular sentences within each non-heading item
        final_sentences = []
        for sentence in sentences:
            if sentence.startswith('#'):
                final_sentences.append(sentence)
            else:
                sub_sentences = self.sentence_pattern.split(sentence)
                final_sentences.extend(s.strip() for s in sub_sentences if s.strip())
        
        return final_sentences
    
    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks based on token limits with optimized processing."""
        if not sentences:
            return []
        
        # Filter out headings and prepare content sentences
        content_sentences = [s for s in sentences if not s.startswith('#')]
        if not content_sentences:
            return []
        
        # Pre-calculate all token counts in batch
        sentence_token_counts = self.token_estimator.count_tokens_batch(content_sentences)
        total_tokens = sum(sentence_token_counts)
        
        if total_tokens < self.min_tokens:
            return [' '.join(content_sentences)]
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        # Pre-analyze content complexity to avoid repeated analysis
        content_analysis = self._analyze_content_complexity(' '.join(content_sentences))
        
        for sentence, sentence_tokens in zip(content_sentences, sentence_token_counts):
            max_tokens = content_analysis['max_tokens']
            
            # Check if we need to finalize current chunk
            if current_token_count + sentence_tokens > max_tokens and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate overlap more efficiently
                chunk_indices = [content_sentences.index(s) for s in current_chunk if s in content_sentences]
                chunk_tokens = [sentence_token_counts[i] for i in chunk_indices if i < len(sentence_token_counts)]
                overlap_sentences, overlap_tokens = self._get_overlap_sentences(
                    current_chunk, chunk_tokens, current_token_count
                )
                current_chunk = overlap_sentences
                current_token_count = overlap_tokens
            
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
            
            # Check if we've reached target tokens
            if current_token_count >= self.target_tokens and len(current_chunk) > 1:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Calculate overlap more efficiently
                chunk_indices = [content_sentences.index(s) for s in current_chunk if s in content_sentences]
                chunk_tokens = [sentence_token_counts[i] for i in chunk_indices if i < len(sentence_token_counts)]
                overlap_sentences, overlap_tokens = self._get_overlap_sentences(
                    current_chunk, chunk_tokens, current_token_count
                )
                current_chunk = overlap_sentences
                current_token_count = overlap_tokens
        
        # Add remaining content
        if current_chunk and current_token_count >= self.min_tokens:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _analyze_content_complexity(self, content: str) -> Dict[str, Any]:
        """
        Analyze content complexity once and cache results.
        """
        if not content:
            return {'max_tokens': self.max_tokens_soft, 'is_single_topic': False}
        
        # Pre-compute all metrics at once
        word_count = len(content.split())
        has_paragraph_breaks = '\n\n' in content.strip()
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        
        # Determine if single-topic based on multiple factors
        is_single_topic = (
            sentence_count <= 3 or 
            (not has_paragraph_breaks and word_count < 100)
        )
        
        max_tokens = self.max_tokens_hard if is_single_topic else self.max_tokens_soft
        
        return {
            'max_tokens': max_tokens,
            'is_single_topic': is_single_topic,
            'word_count': word_count,
            'sentence_count': sentence_count
        }
    
    def _get_overlap_sentences(self, chunk_sentences: List[str], 
                             sentence_tokens: List[int], current_tokens: int) -> Tuple[List[str], int]:
        """Extract overlap sentences more efficiently without re-tokenizing."""
        if not chunk_sentences:
            return [], 0
        
        target_overlap_tokens = int(current_tokens * self.overlap_percentage)
        if target_overlap_tokens == 0:
            return [], 0
        
        overlap_sentences = []
        token_count = 0
        
        # Work backwards through sentences
        for i in range(len(chunk_sentences) - 1, -1, -1):
            sentence = chunk_sentences[i]
            # Use pre-calculated token count if available
            if i < len(sentence_tokens):
                sentence_token_count = sentence_tokens[i]
            else:
                sentence_token_count = self.token_estimator.count_tokens(sentence)
            
            if token_count + sentence_token_count <= target_overlap_tokens:
                overlap_sentences.insert(0, sentence)
                token_count += sentence_token_count
            else:
                break
        
        return overlap_sentences, token_count
    
    def _extract_heading_structure(self, content: str) -> List[Dict[str, Any]]:
        """Extract heading structure from markdown content with optimized parsing."""
        headings = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                # Count heading level more efficiently
                level = len(line_stripped) - len(line_stripped.lstrip('#'))
                
                # Extract heading text
                heading_text = line_stripped[level:].strip()
                if heading_text:
                    headings.append({
                        'text': heading_text,
                        'level': level,
                        'line_number': i,
                        'markdown': line_stripped
                    })
        
        return headings
    
    def _find_relevant_headings(self, chunk_text: str, headings_info: List[Dict[str, Any]], full_content: str) -> List[Dict[str, Any]]:
        """Find the most relevant headings for a chunk with optimized search."""
        if not headings_info:
            return []
        
        # Use a smaller sample for faster matching
        chunk_start = ' '.join(chunk_text.split()[:5])  # Reduced from 10 to 5 words
        if len(chunk_start) < 10:  # If too short, use more context
            chunk_start = chunk_text[:50]  # First 50 characters
        
        # Simplified approach: find chunk position directly
        chunk_pos = full_content.find(chunk_start)
        if chunk_pos == -1:
            # Try with normalized whitespace
            normalized_chunk = ' '.join(chunk_start.split())
            normalized_content = ' '.join(full_content.split())
            chunk_pos = normalized_content.find(normalized_chunk)
            
        if chunk_pos == -1:
            return self._build_heading_hierarchy(headings_info)
        
        # Convert character position to approximate line number
        lines_before = full_content[:chunk_pos].count('\n')
        
        # Find the most relevant heading using binary search approach
        relevant_heading = None
        for heading in reversed(headings_info):
            if heading['line_number'] <= lines_before:
                relevant_heading = heading
                break
        
        if not relevant_heading:
            relevant_heading = headings_info[0] if headings_info else None
        
        if not relevant_heading:
            return []
        
        return self._build_heading_hierarchy(headings_info, relevant_heading)
    
    def _build_heading_hierarchy(self, headings_info: List[Dict[str, Any]], target_heading: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Build a hierarchy of up to 3 headings (target + 2 parents)."""
        if not headings_info:
            return []
        
        if not target_heading:
            # Use the deepest/most specific heading
            target_heading = headings_info[-1]
        
        # Build hierarchy: find 2 parent headings
        hierarchy = [target_heading]
        current_level = target_heading['level']
        target_index = headings_info.index(target_heading)
        
        # Look for parent headings (lower level numbers) before the target
        for heading in reversed(headings_info[:target_index]):
            if heading['level'] < current_level and len(hierarchy) < 3:
                hierarchy.insert(0, heading)
                current_level = heading['level']
        
        return hierarchy
    
    def _prepend_headings_to_chunk(self, chunk_text: str, headings: List[Dict[str, Any]]) -> str:
        """Prepend headings to chunk content in the specified format."""
        if not headings:
            return chunk_text
        
        # Build heading prefix
        heading_prefix = []
        for heading in headings:
            level_marker = '#' * heading['level']
            heading_line = f"{level_marker} {heading['text']} [H{heading['level']}]"
            heading_prefix.append(heading_line)
        
        # Combine headings with chunk content
        if heading_prefix:
            return '\n\n'.join(heading_prefix) + '\n\n' + chunk_text
        else:
            return chunk_text
    
    def _create_chunks_from_json(self, json_data: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from JSON structure data."""
        elements = json_data.get('elements', [])
        if not elements:
            return []
        
        chunks = self._group_elements_into_chunks(elements)
        if not chunks:
            return []
        
        doc_id = metadata.get('doc_id', 'unknown')
        
        # Pre-process all chunk contents
        chunk_contents = []
        for chunk_elements in chunks:
            content = self._elements_to_text(chunk_elements)
            if content.strip():
                chunk_contents.append(content)
        
        if not chunk_contents:
            return []
        
        # Batch count tokens for all contents
        token_counts = self.token_estimator.count_tokens_batch(chunk_contents)
        
        # Build result chunks
        chunk_dicts = []
        for i, (content, token_count) in enumerate(zip(chunk_contents, token_counts)):
            chunk_dicts.append({
                'chunk_id': f"{doc_id}_chunk_{i+1}",
                'content': content,
                'token_count': token_count,
                'word_count': len(content.split()),
                'doc_id': doc_id,
                'section_path': []
            })
        
        return chunk_dicts
    
    def _group_elements_into_chunks(self, elements: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group JSON elements into chunks with optimized processing."""
        if not elements:
            return []
        
        # Batch calculate token counts
        element_texts = [elem.get('text', '') for elem in elements]
        element_token_counts = self.token_estimator.count_tokens_batch(element_texts)
        
        # Early return for single small element
        if len(elements) == 1 and element_token_counts[0] < self.min_tokens:
            return [elements]
        
        # Pre-analyze content complexity for the entire set
        full_text = '\n'.join(element_texts)
        content_analysis = self._analyze_content_complexity(full_text)
        max_tokens = content_analysis['max_tokens']
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for element, element_tokens in zip(elements, element_token_counts):
            # Check if we need to finalize current chunk
            if current_token_count + element_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_token_count = 0
            
            current_chunk.append(element)
            current_token_count += element_tokens
            
            # Check if we've reached target tokens
            if current_token_count >= self.target_tokens:
                chunks.append(current_chunk)
                current_chunk = []
                current_token_count = 0
        
        # Add remaining elements
        if current_chunk and current_token_count >= self.min_tokens:
            chunks.append(current_chunk)
        
        return chunks
    
    def _elements_to_text(self, elements: List[Dict[str, Any]]) -> str:
        """Convert JSON elements to text content efficiently."""
        if not elements:
            return ''
        
        # Pre-filter and collect non-empty texts in one pass
        texts = []
        for element in elements:
            text = element.get('text', '').strip()
            if text:
                texts.append(text)
        
        return '\n\n'.join(texts)
    
    def _enhance_with_json_data(self, chunks: List[Dict[str, Any]], json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance MD-based chunks with JSON structure data."""
        # This is a placeholder for more sophisticated enhancement
        # In practice, you might align chunks with JSON elements, extract tables, etc.
        return chunks