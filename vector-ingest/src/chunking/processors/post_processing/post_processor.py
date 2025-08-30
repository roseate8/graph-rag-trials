import re
import logging
from typing import List, Dict, Any, Optional

from .chunk_cleaner import ChunkCleaner

logger = logging.getLogger(__name__)


class PostProcessor:
    """Post-process text chunks after chunking is complete."""
    
    def __init__(self):
        self.cleaner = ChunkCleaner()
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process all chunks with cleanup and merging."""
        # Clean all chunks first
        cleaned_chunks = self.cleanup_chunks(chunks)
        
        # Apply merging logic
        merged_chunks = self.merge_chunks(cleaned_chunks)
        
        return merged_chunks
    
    def cleanup_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean up all chunks using the chunk cleaner."""
        cleaned_chunks = []
        
        for chunk in chunks:
            if isinstance(chunk, dict) and 'content' in chunk:
                cleaned_content = self.cleaner.clean_chunk(chunk['content'])
                if cleaned_content:  # Only keep non-empty chunks after cleaning
                    cleaned_chunk = chunk.copy()
                    cleaned_chunk['content'] = cleaned_content
                    cleaned_chunks.append(cleaned_chunk)
            else:
                # Handle chunks that might be simple strings
                if isinstance(chunk, str):
                    cleaned_content = self.cleaner.clean_chunk(chunk)
                    if cleaned_content:
                        cleaned_chunks.append({'content': cleaned_content})
                else:
                    # Keep other chunk formats as-is
                    cleaned_chunks.append(chunk)
        
        return cleaned_chunks
    
    def merge_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge small chunks based on hierarchy compatibility and similarity."""
        if len(chunks) <= 1:
            return chunks
        
        min_chunk_tokens = 50
        similarity_threshold = 0.6
        
        merged_chunks = []
        i = 0
        merge_operations = 0
        
        # Count small chunks for logging
        small_chunk_count = sum(1 for chunk in chunks 
                               if len(chunk.get('content', '').split()) < min_chunk_tokens)
        
        logger.info(f"🔗 Starting merge process: {len(chunks)} total chunks, "
                   f"{small_chunk_count} small chunks (<{min_chunk_tokens} tokens)")
        
        while i < len(chunks):
            current_chunk = chunks[i]
            current_content = current_chunk.get('content', '')
            current_tokens = len(current_content.split())
            
            # Try to merge if current chunk is small
            if current_tokens < min_chunk_tokens and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                
                if self._can_merge_chunks(current_chunk, next_chunk, similarity_threshold):
                    # Merge the chunks
                    merged_chunk = self._merge_two_chunks(current_chunk, next_chunk)
                    merged_chunks.append(merged_chunk)
                    merge_operations += 1
                    i += 2  # Skip both chunks since we merged them
                    continue
            
            # If we couldn't merge, add the current chunk as-is
            merged_chunks.append(current_chunk)
            i += 1
        
        logger.info(f"🔗 Merge complete: {len(chunks)} → {len(merged_chunks)} chunks "
                   f"({merge_operations} merged, {merge_operations} operations)")
        
        return merged_chunks
    
    def _can_merge_chunks(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any], 
                         similarity_threshold: float) -> bool:
        """Determine if two chunks can be merged based on hierarchy and similarity."""
        
        # Check section path compatibility first (hierarchy-based merging)
        path1 = chunk1.get('section_path', [])
        path2 = chunk2.get('section_path', [])
        
        if path1 and path2:
            # Same section - can merge
            if path1 == path2:
                return True
            
            # Adjacent sections in same parent - can merge
            if self._are_compatible_sections(path1, path2):
                return True
        
        # If no clear hierarchy, check content similarity
        content1 = chunk1.get('content', '')
        content2 = chunk2.get('content', '')
        
        if content1 and content2:
            similarity = self._calculate_jaccard_similarity(content1, content2)
            return similarity > similarity_threshold
        
        # Default: allow merge if no strong indicators against it
        return True
    
    def _are_compatible_sections(self, path1: List[str], path2: List[str]) -> bool:
        """Check if two section paths represent compatible/adjacent sections."""
        if not path1 or not path2:
            return True
        
        # If different lengths, check if one is a subsection of the other
        if len(path1) != len(path2):
            shorter = path1 if len(path1) < len(path2) else path2
            longer = path2 if len(path1) < len(path2) else path1
            
            # Check if shorter path is a prefix of longer path
            return longer[:len(shorter)] == shorter
        
        # Same length - check how many levels differ
        differences = sum(1 for p1, p2 in zip(path1, path2) if p1 != p2)
        
        # Allow merge if they differ in at most one level (adjacent sections)
        return differences <= 1
    
    def _calculate_jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Convert to word sets (lowercase for comparison)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity: intersection / union
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_two_chunks(self, chunk1: Dict[str, Any], chunk2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two chunks into a single chunk."""
        # Combine content
        content1 = chunk1.get('content', '')
        content2 = chunk2.get('content', '')
        merged_content = content1 + '\n\n' + content2
        
        # Calculate new word count
        merged_word_count = len(merged_content.split())
        
        # Create merged chunk based on first chunk's metadata
        merged_chunk = chunk1.copy()
        merged_chunk['content'] = merged_content
        merged_chunk['word_count'] = merged_word_count
        
        # Handle section paths - prefer more specific path
        path1 = chunk1.get('section_path', [])
        path2 = chunk2.get('section_path', [])
        
        if path1 and path2:
            # Use the longer (more specific) path, or first if same length
            merged_chunk['section_path'] = path1 if len(path1) >= len(path2) else path2
        elif path1 or path2:
            merged_chunk['section_path'] = path1 or path2
        
        return merged_chunk