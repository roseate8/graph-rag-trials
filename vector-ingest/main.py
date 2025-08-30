#!/usr/bin/env python3
"""
Graph-RAG Document Processing Pipeline
Main entry point for document chunking with TOC detection.
"""

import logging
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.chunking.models import Chunk, DocumentMetadata, DocumentStructure, ChunkMetadata
from src.chunking.processors.toc_detector import TableOfContentsDetector
from src.chunking.processors.text_chunker import TextChunker
from src.chunking.processors import PostProcessor, ChunkCleaner
from src.chunking.processors.entity_extractor import EntityExtractor
from src.embeddings import EmbeddingService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenTracker:
    """Track token consumption throughout the processing pipeline."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.processing_tokens = 0
        self.embedding_tokens = 0
        self.openai_input_tokens = 0
        self.openai_output_tokens = 0
        self.openai_cost = 0.0
        
    def add_input_tokens(self, count: int):
        """Add to input token count."""
        self.input_tokens += count
        
    def add_output_tokens(self, count: int):
        """Add to output token count."""
        self.output_tokens += count
        
    def add_processing_tokens(self, count: int):
        """Add to processing token count."""
        self.processing_tokens += count
        
    def add_embedding_tokens(self, count: int):
        """Add to embedding token count."""
        self.embedding_tokens += count
    
    def add_openai_tokens(self, input_tokens: int, output_tokens: int, cost: float):
        """Add OpenAI API token consumption."""
        self.openai_input_tokens += input_tokens
        self.openai_output_tokens += output_tokens
        self.openai_cost += cost
    
    def get_openai_total(self) -> int:
        """Get total OpenAI API tokens consumed."""
        return self.openai_input_tokens + self.openai_output_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (1 token ≈ 4 characters)."""
        if not text:
            return 0
        return len(text) // 4
    
    def get_total(self) -> int:
        """Get total tokens consumed."""
        return self.input_tokens + self.output_tokens + self.processing_tokens + self.embedding_tokens
    
    def log_summary(self):
        """Log token consumption summary."""
        logger.info(f"📊 Token Consumption Summary:")
        logger.info(f"  📥 Input tokens: {self.input_tokens:,}")
        logger.info(f"  📤 Output tokens: {self.output_tokens:,}")
        logger.info(f"  Processing tokens: {self.processing_tokens:,}")
        logger.info(f"  🔮 Embedding tokens: {self.embedding_tokens:,}")
        logger.info(f"  🎯 Total tokens: {self.get_total():,}")


class DocumentProcessor:
    """Streamlined document processing with integrated TOC detection."""
    
    def __init__(self, input_dir: Path = None, output_dir: Path = None, use_cache: bool = True):
        self.toc_detector = TableOfContentsDetector()
        self.text_chunker = TextChunker(target_words=700, max_words=800, overlap_words=15)
        self.post_processor = PostProcessor()
        self.entity_extractor = EntityExtractor()
        self.embedding_service = EmbeddingService()
        self.token_tracker = TokenTracker()
        self.input_dir = input_dir or Path.cwd() / "input"
        self.output_dir = output_dir or Path.cwd() / "output"
        self.use_cache = use_cache
        self.cache_dir = self.output_dir / ".cache"
        if use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # File type mapping for efficiency
        self.type_map = {
            '.pdf': 'pdf', '.docx': 'docx', '.doc': 'docx',
            '.html': 'html', '.htm': 'html', '.md': 'md', '.txt': 'md'
        }
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate hash for file to detect changes."""
        stat = file_path.stat()
        content = f"{file_path.name}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, file_path: Path) -> Path:
        """Get cache file path for a document."""
        file_hash = self._get_file_hash(file_path)
        return self.cache_dir / f"{file_path.stem}_{file_hash}.json"
    
    def _load_from_cache(self, file_path: Path) -> List[Chunk]:
        """Load processed chunks from cache if available."""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(file_path)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            chunks = []
            for chunk_data in cache_data.get('chunks', []):
                metadata = ChunkMetadata(
                    chunk_id=chunk_data['metadata']['chunk_id'],
                    doc_id=chunk_data['metadata']['doc_id'],
                    chunk_type=chunk_data['metadata']['chunk_type'],
                    word_count=chunk_data['metadata']['word_count'],
                    section_path=chunk_data['metadata'].get('section_path', [])
                )
                
                chunk = Chunk(metadata=metadata, content=chunk_data['content'])
                if 'embedding' in chunk_data:
                    chunk.embedding = chunk_data['embedding']
                chunks.append(chunk)
                
            logger.info(f"📦 Loaded {len(chunks)} chunks from cache for {file_path.name}")
            return chunks
            
        except Exception as e:
            logger.warning(f"Failed to load cache for {file_path.name}: {str(e)}")
            return None
    
    def _save_to_cache(self, file_path: Path, chunks: List[Chunk]):
        """Save processed chunks to cache."""
        if not self.use_cache or not chunks:
            return
            
        cache_path = self._get_cache_path(file_path)
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'chunks': []
            }
            
            for chunk in chunks:
                chunk_data = {
                    'content': chunk.content,
                    'metadata': {
                        'chunk_id': chunk.metadata.chunk_id,
                        'doc_id': chunk.metadata.doc_id,
                        'chunk_type': chunk.metadata.chunk_type,
                        'word_count': chunk.metadata.word_count,
                        'section_path': chunk.metadata.section_path or []
                    }
                }
                
                if hasattr(chunk, 'embedding') and chunk.embedding:
                    chunk_data['embedding'] = chunk.embedding
                    
                cache_data['chunks'].append(chunk_data)
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to save cache for {file_path.name}: {str(e)}")
    
    def process_all_documents(self, max_workers: int = None) -> List[Chunk]:
        """Process all documents in input directory with parallel processing."""
        # Ensure input directory exists
        if not self.input_dir.exists():
            self.input_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created input directory: {self.input_dir}")
            logger.info(f"Please place your documents in {self.input_dir} and run again.")
            return []
        
        files = self._discover_input_files(self.input_dir)
        
        if not files:
            logger.warning(f"No supported files found in {self.input_dir}")
            logger.info("Supported formats: PDF, DOCX, HTML, MD, TXT")
            return []
        
        # Check cache first
        cached_files = []
        files_to_process = []
        
        for file_path in files:
            cached_chunks = self._load_from_cache(file_path)
            if cached_chunks:
                cached_files.append((file_path, cached_chunks))
            else:
                files_to_process.append(file_path)
        
        logger.info(f"Found {len(files)} files: {len(cached_files)} cached, {len(files_to_process)} to process")
        
        all_chunks = []
        
        # Add cached chunks
        for file_path, chunks in cached_files:
            all_chunks.extend(chunks)
            # Update token tracking for cached files
            for chunk in chunks:
                self.token_tracker.add_input_tokens(self.token_tracker.estimate_tokens(chunk.content))
        
        # Process remaining files in parallel
        if files_to_process:
            if max_workers is None:
                max_workers = min(len(files_to_process), cpu_count())
            
            logger.info(f"Processing {len(files_to_process)} files with {max_workers} workers")
            
            # For parallel processing, we need a standalone function
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files for processing
                future_to_file = {
                    executor.submit(process_single_document, file_path, self.input_dir, self.output_dir, self.use_cache): file_path
                    for file_path in files_to_process
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        chunks, token_stats = future.result()
                        all_chunks.extend(chunks)
                        
                        # Update token tracking
                        self.token_tracker.add_input_tokens(token_stats['input_tokens'])
                        self.token_tracker.add_processing_tokens(token_stats['processing_tokens'])
                        self.token_tracker.add_embedding_tokens(token_stats['embedding_tokens'])
                        
                        logger.info(f"Generated {len(chunks)} chunks from {file_path.name}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process {file_path.name}: {str(e)}")
                        continue
        
        logger.info(f"🎯 Total chunks generated: {len(all_chunks)}")
        return all_chunks
    
    def process_document(self, file_path: Path) -> List[Chunk]:
        """Process a single document with TOC detection and chunking."""
        # 1. Read content from file
        logger.info(f"Reading {file_path.name}...")
        cleaned_content = self._read_file(file_path)
        
        if not cleaned_content.strip():
            logger.warning(f"No content after reading: {file_path.name}")
            return []
        
        # Track input tokens
        input_tokens = self.token_tracker.estimate_tokens(cleaned_content)
        self.token_tracker.add_input_tokens(input_tokens)
        logger.info(f"📄 Content loaded: {len(cleaned_content)} chars, ~{input_tokens:,} tokens")
        
        # 2. Create document metadata
        metadata = self._create_document_metadata(file_path)
        
        # 3. Detect TOC and analyze structure
        logger.info("Analyzing document structure...")
        structure = self._analyze_structure(cleaned_content, file_path)
        
        # 4. Create initial chunks using TextChunker
        logger.info("🔪 Creating text chunks...")
        raw_chunks = self._create_raw_chunks(cleaned_content, structure, metadata)
        
        # 5. Post-process chunks (clean and merge small chunks)
        logger.info("🧹 Post-processing chunks...")
        processed_chunks = self._post_process_chunks(raw_chunks)
        
        # 6. Extract entities and populate metadata
        logger.info("🏷️  Extracting entities...")
        chunks_with_entities = self._extract_entities(processed_chunks)
        
        # 7. Generate embeddings
        logger.info("🔮 Generating embeddings...")
        final_chunks = self._generate_embeddings(chunks_with_entities)
        
        return final_chunks
    
    def _discover_input_files(self, input_dir: Path) -> List[Path]:
        """Discover all supported files in input directory."""
        if not input_dir.exists():
            return []
        
        supported_exts = {'.txt', '.md', '.html', '.htm'}
        files = [f for f in input_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in supported_exts]
        
        return sorted(files)
    
    def _supports_file_type(self, file_path: Path) -> bool:
        """Check if file type is supported."""
        supported_exts = {'.txt', '.md', '.html', '.htm'}
        return file_path.suffix.lower() in supported_exts
    
    def _read_file(self, file_path: Path) -> str:
        """Read file content with encoding fallback."""
        if not self._supports_file_type(file_path):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        try:
            # Handle different encodings efficiently
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            
            for encoding in encodings:
                try:
                    content = file_path.read_text(encoding=encoding)
                    return content
                except UnicodeDecodeError:
                    continue
            
            # Fallback with error handling
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            return content
            
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {str(e)}")

    def _create_document_metadata(self, file_path: Path) -> DocumentMetadata:
        """Create document metadata from file."""
        return DocumentMetadata(
            doc_id=file_path.stem,
            title=file_path.stem.replace('_', ' ').replace('-', ' ').title(),
            source_type=self.type_map.get(file_path.suffix.lower(), 'md'),
            author=None,
            date=None,
            page_count=None
        )
    
    def _analyze_structure(self, content: str, file_path: Path = None) -> DocumentStructure:
        """Analyze document structure using our systematic TOC detector."""
        logger.info("🔍 Detecting Table of Contents...")
        
        # Determine JSON path if available
        json_path = None
        if file_path:
            json_file = file_path.with_suffix('.json')
            if json_file.exists():
                json_path = str(json_file)
        
        # Use our systematic TOC detector with JSON-first strategy
        toc_entries = self.toc_detector.detect_toc(content, json_path=json_path)
        
        if toc_entries:
            logger.info(f"Found TOC with {len(toc_entries)} entries")
            
            # Convert TOC entries to structure format
            toc_sections = []
            for entry in toc_entries:
                toc_sections.append({
                    "level": entry.level,
                    "title": entry.title,
                    "page": entry.page,
                    "line_number": entry.line_number,
                    "confidence": entry.confidence
                })
            
            # Log some TOC entries for visibility
            for i, entry in enumerate(toc_entries[:3]):  # Show first 3
                logger.info(f"  {i+1}. {entry.title} (level {entry.level})")
            if len(toc_entries) > 3:
                logger.info(f"  ... and {len(toc_entries)-3} more entries")
        
        else:
            logger.info("❌ No Table of Contents detected")
            toc_sections = []
        
        return DocumentStructure(toc_sections=toc_sections)
    
    def _create_raw_chunks(self, content: str, structure: DocumentStructure, doc_metadata: DocumentMetadata) -> List[Chunk]:
        """Create initial chunks using TextChunker."""
        if not content.strip():
            return []
        
        # Use TextChunker to split content into proper chunks
        chunk_dicts = self.text_chunker.process(content)
        
        chunks = []
        for i, chunk_dict in enumerate(chunk_dicts):
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{doc_metadata.doc_id}_chunk_{i+1}",
                doc_id=doc_metadata.doc_id,
                chunk_type="text",
                word_count=len(chunk_dict['content'].split()),
                section_path=chunk_dict.get('section_path', [])
            )
            
            chunks.append(Chunk(metadata=chunk_metadata, content=chunk_dict['content']))
        
        # Track processing tokens for chunks
        total_chunk_tokens = sum(self.token_tracker.estimate_tokens(chunk.content) for chunk in chunks)
        self.token_tracker.add_processing_tokens(total_chunk_tokens)
        
        logger.info(f"🔪 Created {len(chunks)} initial chunks (~{total_chunk_tokens:,} tokens)")
        return chunks
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Clean and merge chunks using PostProcessor."""
        if not chunks:
            return chunks
        
        # Convert Chunk objects to dict format for PostProcessor
        chunk_dicts = []
        for chunk in chunks:
            chunk_dicts.append({
                'content': chunk.content,
                'metadata': {
                    'chunk_id': chunk.metadata.chunk_id,
                    'doc_id': chunk.metadata.doc_id,
                    'chunk_type': chunk.metadata.chunk_type,
                    'word_count': chunk.metadata.word_count,
                    'section_path': chunk.metadata.section_path or []
                }
            })
        
        # Process chunks (clean and merge)
        processed_dicts = self.post_processor.process_chunks(chunk_dicts)
        
        # Convert back to Chunk objects
        processed_chunks = []
        for i, chunk_dict in enumerate(processed_dicts):
            metadata_dict = chunk_dict.get('metadata', {})
            
            chunk_metadata = ChunkMetadata(
                chunk_id=metadata_dict.get('chunk_id', f"processed_chunk_{i+1}"),
                doc_id=metadata_dict.get('doc_id', ''),
                chunk_type=metadata_dict.get('chunk_type', 'text'),
                word_count=len(chunk_dict['content'].split()),
                section_path=metadata_dict.get('section_path', [])
            )
            
            processed_chunks.append(Chunk(metadata=chunk_metadata, content=chunk_dict['content']))
        
        # Track token changes from post-processing
        final_tokens = sum(self.token_tracker.estimate_tokens(chunk.content) for chunk in processed_chunks)
        logger.info(f"🧹 Post-processed to {len(processed_chunks)} chunks (~{final_tokens:,} tokens)")
        return processed_chunks
    
    def _extract_entities(self, chunks: List[Chunk]) -> List[Chunk]:
        """Extract entities and populate chunk metadata."""
        if not chunks:
            return chunks
        
        for chunk in chunks:
            # Extract entities from content
            entities = self.entity_extractor.process(chunk.content)
            
            # Add entities to chunk metadata (extend ChunkMetadata if needed)
            if hasattr(chunk.metadata, 'entities'):
                chunk.metadata.entities = entities
            # For now, we can store in section_path or create a simple string representation
            if entities.get('people') or entities.get('organizations') or entities.get('dates'):
                entity_info = []
                if entities.get('people'):
                    entity_info.append(f"People: {', '.join(list(entities['people'])[:3])}")
                if entities.get('organizations'):
                    entity_info.append(f"Orgs: {', '.join(list(entities['organizations'])[:3])}")
                if entities.get('dates'):
                    entity_info.append(f"Dates: {', '.join(list(entities['dates'])[:3])}")
                
                # Add to section_path for now (since that's what we have in ChunkMetadata)
                if not chunk.metadata.section_path:
                    chunk.metadata.section_path = entity_info[:1]  # Just take first for brevity
        
        logger.info(f"🏷️  Extracted entities for {len(chunks)} chunks")
        return chunks
    
    def _generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for all chunks."""
        if not chunks:
            return chunks
        
        # Track embedding tokens (approximate - embeddings process the text)
        embedding_tokens = sum(self.token_tracker.estimate_tokens(chunk.content) for chunk in chunks)
        self.token_tracker.add_embedding_tokens(embedding_tokens)
        
        # Generate embeddings using EmbeddingService
        chunks_with_embeddings = self.embedding_service.embed_chunks(chunks)
        
        logger.info(f"🔮 Generated embeddings for {len(chunks_with_embeddings)} chunks (~{embedding_tokens:,} tokens processed)")
        return chunks_with_embeddings


def process_single_document(file_path: Path, input_dir: Path, output_dir: Path, use_cache: bool = True):
    """
    Standalone function for processing a single document.
    This function is designed to be used with ProcessPoolExecutor.
    
    Returns:
        Tuple[List[Chunk], Dict[str, int]]: (chunks, token_stats)
    """
    # Initialize a processor for this worker
    processor = DocumentProcessor(input_dir, output_dir, use_cache)
    
    # Process the document
    chunks = processor.process_document(file_path)
    
    # Return chunks and token statistics
    token_stats = {
        'input_tokens': processor.token_tracker.input_tokens,
        'processing_tokens': processor.token_tracker.processing_tokens,
        'embedding_tokens': processor.token_tracker.embedding_tokens
    }
    
    # Save to cache if enabled
    if use_cache and chunks:
        processor._save_to_cache(file_path, chunks)
    
    return chunks, token_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process documents with TOC detection")
    parser.add_argument(
        "--input-dir", 
        type=Path, 
        default=Path.cwd() / "input",
        help="Directory containing input documents (default: ./input)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "output", 
        help="Directory to save processed chunks (default: ./output)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear processing cache before running"
    )
    parser.add_argument(
        "--cache-only",
        action="store_true", 
        help="Only clear cache and exit (don't process documents)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle cache clearing
    cache_dir = args.output_dir / ".cache"
    
    if args.clear_cache or args.cache_only:
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared cache directory: {cache_dir}")
        else:
            logger.info(f"Cache directory doesn't exist: {cache_dir}")
    
    if args.cache_only:
        logger.info("Cache cleared. Exiting (--cache-only specified).")
        return
    
    logger.info("Starting Graph-RAG Document Processing Pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize processor
    processor = DocumentProcessor(input_dir=args.input_dir, use_cache=not args.clear_cache)
    
    # Process all documents
    chunks = processor.process_all_documents()
    
    if not chunks:
        logger.warning("No chunks generated. Check your input directory and files.")
        return
    
    # Log token consumption summary
    processor.token_tracker.log_summary()
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save chunks with embeddings in JSON format
    chunks_file = args.output_dir / "processed_chunks.json"
    chunks_data = []
    
    for chunk in chunks:
        chunk_data = {
            "chunk_id": chunk.metadata.chunk_id,
            "doc_id": chunk.metadata.doc_id,
            "chunk_type": chunk.metadata.chunk_type,
            "word_count": chunk.metadata.word_count,
            "section_path": chunk.metadata.section_path or [],
            "content": chunk.content,
            "embedding": chunk.embedding if hasattr(chunk, 'embedding') and chunk.embedding else None,
            "embedding_dim": len(chunk.embedding) if hasattr(chunk, 'embedding') and chunk.embedding else 0
        }
        chunks_data.append(chunk_data)
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(chunks),
            "chunks": chunks_data
        }, f, indent=2, ensure_ascii=False)
    
    # Also save a readable summary
    summary_file = args.output_dir / "processing_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Graph-RAG Document Processing Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Total chunks generated: {len(chunks)}\n")
        f.write(f"Chunks with embeddings: {sum(1 for c in chunks if hasattr(c, 'embedding') and c.embedding)}\n")
        f.write(f"Average words per chunk: {sum(c.metadata.word_count for c in chunks) / len(chunks):.1f}\n")
        f.write(f"Total tokens processed: {processor.token_tracker.get_total():,}\n\n")
        
        for chunk in chunks[:5]:  # Show first 5 chunks
            f.write(f"Chunk ID: {chunk.metadata.chunk_id}\n")
            f.write(f"Document: {chunk.metadata.doc_id}\n")
            f.write(f"Word count: {chunk.metadata.word_count}\n")
            f.write(f"Has embedding: {'Yes' if hasattr(chunk, 'embedding') and chunk.embedding else 'No'}\n")
            f.write(f"Section info: {', '.join(chunk.metadata.section_path) if chunk.metadata.section_path else 'None'}\n")
            f.write(f"Content preview: {chunk.content[:200]}...\n")
            f.write("-" * 50 + "\n\n")
        
        if len(chunks) > 5:
            f.write(f"... and {len(chunks)-5} more chunks\n")
    
    logger.info(f"Processing complete!")
    logger.info(f"Chunks saved to: {chunks_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"Generated {len(chunks)} chunks with full pipeline processing")
    logger.info(f"Total non-API tokens: {processor.token_tracker.get_total():,}")
    if processor.token_tracker.get_openai_total() > 0:
        logger.info(f"OpenAI API tokens: {processor.token_tracker.get_openai_total():,}")
        logger.info(f"OpenAI API cost: ${processor.token_tracker.openai_cost:.6f}")
    else:
        logger.info(f"OpenAI API tokens: 0 (no API calls made)")


if __name__ == "__main__":
    main()