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
from src.chunking.processors import PostProcessor, ChunkCleaner, TableProcessor
from src.chunking.processors.entity_extractor import EntityExtractor
from src.chunking.processors.llm_utils import get_openai_api_key, has_openai_api_key, clear_openai_api_key, set_openai_api_key
from src.embeddings import EmbeddingService, clear_milvus_collection, drop_milvus_collection, MilvusVectorStore, get_config

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
    
    def __init__(self, input_dir: Path = None, output_dir: Path = None, use_cache: bool = True, enable_llm: bool = True):
        self.toc_detector = TableOfContentsDetector()
        self.text_chunker = TextChunker(target_words=700, max_words=800, overlap_words=15)
        self.post_processor = PostProcessor()
        self.entity_extractor = EntityExtractor()
        self.embedding_service = EmbeddingService()
        self.token_tracker = TokenTracker()
        self.input_dir = input_dir or Path.cwd() / "input"
        self.output_dir = output_dir or Path.cwd() / "output"
        self.use_cache = use_cache
        self.enable_llm = enable_llm
        self.cache_dir = self.output_dir / ".cache"
        if use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize table processor (will be configured per document)
        self.table_processor = None
        
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
                metadata = {
                    'chunk_id': chunk.metadata.chunk_id,
                    'doc_id': chunk.metadata.doc_id,
                    'chunk_type': chunk.metadata.chunk_type,
                    'word_count': chunk.metadata.word_count,
                    'section_path': chunk.metadata.section_path or []
                }
                
                # Add optional metadata fields if present
                if chunk.metadata.page is not None:
                    metadata['page'] = chunk.metadata.page
                    
                # Add lists if not empty
                if chunk.metadata.outbound_refs:
                    metadata['outbound_refs'] = [ref.model_dump() for ref in chunk.metadata.outbound_refs]
                if chunk.metadata.inbound_refs:
                    metadata['inbound_refs'] = chunk.metadata.inbound_refs
                if chunk.metadata.regions:
                    metadata['regions'] = chunk.metadata.regions
                if chunk.metadata.metrics:
                    metadata['metrics'] = chunk.metadata.metrics
                if chunk.metadata.time_periods:
                    metadata['time_periods'] = chunk.metadata.time_periods
                if chunk.metadata.dates:
                    metadata['dates'] = chunk.metadata.dates
                    
                # Add table-specific metadata for table chunks
                if chunk.metadata.chunk_type == "table":
                    if chunk.metadata.table_id:
                        metadata['table_id'] = chunk.metadata.table_id
                    if chunk.metadata.column_headers:
                        metadata['column_headers'] = chunk.metadata.column_headers
                    if chunk.metadata.table_title:
                        metadata['table_title'] = chunk.metadata.table_title
                    if chunk.metadata.table_caption:
                        metadata['table_caption'] = chunk.metadata.table_caption
                
                chunk_data = {
                    'content': chunk.content,
                    'metadata': metadata
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
            
            # Temporarily disable multiprocessing to debug freezing issue
            logger.info(f"Processing {len(files_to_process)} files sequentially (debugging mode)")
            
            # Sequential processing for debugging
            for file_path in files_to_process:
                try:
                    chunks, token_stats = process_single_document(file_path, self.input_dir, self.output_dir, self.use_cache, self.enable_llm, None)
                    all_chunks.extend(chunks)
                    
                    # Update token tracking
                    self.token_tracker.add_input_tokens(token_stats['input_tokens'])
                    self.token_tracker.add_processing_tokens(token_stats['processing_tokens'])
                    self.token_tracker.add_embedding_tokens(token_stats['embedding_tokens'])
                    
                    logger.info(f"Generated {len(chunks)} chunks from {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to process {file_path.name}: {str(e)}")
                    continue
            
            """
            # Original parallel processing code (temporarily disabled)
            logger.info(f"Processing {len(files_to_process)} files with {max_workers} workers")
            
            # For parallel processing, we need a standalone function
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Get API key from main process to pass to workers
                api_key = None
                if self.enable_llm and has_openai_api_key():
                    try:
                        # Get the raw API key (don't use get_openai_api_key as it might prompt again)
                        from src.chunking.processors.llm_utils import _api_key_manager
                        api_key = _api_key_manager._api_key
                        logger.debug(f"🔑 API key retrieved for worker processes: {'✓' if api_key else '✗'}")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to retrieve API key for workers: {e}")
                        api_key = None
                
                # Submit all files for processing
                future_to_file = {
                    executor.submit(process_single_document, file_path, self.input_dir, self.output_dir, self.use_cache, self.enable_llm, api_key): file_path
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
            """
        
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
        
        # 4. Process tables if JSON file exists
        table_chunks = []
        if file_path:
            json_file = file_path.with_suffix('.json')
            logger.info(f"🔍 Checking for JSON file: {json_file}")
            
            if json_file.exists():
                logger.info(f"✅ Found JSON file - proceeding with table processing")
                table_chunks = self._process_tables(json_file, file_path, metadata)
            else:
                logger.info(f"ℹ️  No JSON file found at {json_file} - skipping table processing")
        else:
            logger.info("ℹ️  No file_path provided - skipping table processing")
        
        # 5. Create initial chunks using TextChunker
        logger.info("🔪 Creating text chunks...")
        raw_chunks = self._create_raw_chunks(cleaned_content, structure, metadata)
        
        # 5. Post-process chunks (clean and merge small chunks)
        logger.info("🧹 Post-processing chunks...")
        processed_chunks = self._post_process_chunks(raw_chunks)
        
        # 6. Combine all chunks (text + table)
        all_chunks = processed_chunks + table_chunks
        logger.info(f"Combined {len(processed_chunks)} text chunks + {len(table_chunks)} table chunks = {len(all_chunks)} total")
        
        # 7. Extract entities and populate metadata
        logger.info("🏷️  Extracting entities...")
        chunks_with_entities = self._extract_entities(all_chunks)
        
        # 8. Generate embeddings
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
    
    def _process_tables(self, json_file: Path, md_file: Path, metadata: DocumentMetadata) -> List[Chunk]:
        """Process tables using JSON detection + MD extraction strategy."""
        try:
            logger.info(f"🏢 Starting table processing for {metadata.doc_id}...")
            logger.info(f"📊 JSON source: {json_file.name}")
            logger.info(f"📝 MD source: {md_file.name}")
            
            # Check file existence and sizes
            if not json_file.exists():
                logger.error(f"❌ JSON file not found: {json_file}")
                return []
            
            if not md_file.exists():
                logger.error(f"❌ MD file not found: {md_file}")
                return []
            
            json_size = json_file.stat().st_size
            md_size = md_file.stat().st_size
            logger.info(f"📏 File sizes: JSON={json_size:,} bytes, MD={md_size:,} bytes")
            
            # Initialize table processor with MD file path and LLM setting
            self.table_processor = TableProcessor(md_file_path=md_file, generate_llm_metadata=self.enable_llm)
            
            # Read JSON content
            logger.debug("📖 Reading JSON content...")
            json_content = json_file.read_text(encoding='utf-8')
            logger.debug(f"JSON content loaded: {len(json_content)} characters")
            
            # Process tables asynchronously
            logger.info("🔄 Processing tables with TableProcessor...")
            import asyncio
            table_chunks = asyncio.run(
                self.table_processor.process(json_content, doc_id=metadata.doc_id)
            )
            
            # Log results and token usage
            logger.info(f"✅ Table processing complete: {len(table_chunks)} chunks generated")
            
            # Extract and log token statistics from table processor
            if (self.table_processor and 
                hasattr(self.table_processor, 'table_chunker') and 
                self.table_processor.table_chunker.llm_processor):
                
                stats = self.table_processor.table_chunker.llm_processor.get_token_stats()
                if stats['global_totals']['api_calls'] > 0:
                    # Log detailed token report
                    self.table_processor.table_chunker.llm_processor.log_final_token_report()
                    
                    # Add to main token tracker
                    self.token_tracker.add_openai_tokens(
                        stats['global_totals']['input_tokens'], 
                        stats['global_totals']['output_tokens'], 
                        stats['global_totals']['total_cost']
                    )
                else:
                    logger.info("ℹ️  No OpenAI API calls made for table processing")
            
            return table_chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to process tables for {metadata.doc_id}: {str(e)}", exc_info=True)
            return []
    
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


def process_single_document(file_path: Path, input_dir: Path, output_dir: Path, use_cache: bool = True, enable_llm: bool = True, api_key: str = None):
    """
    Standalone function for processing a single document.
    This function is designed to be used with ProcessPoolExecutor.
    
    Args:
        api_key: OpenAI API key to set in this worker process
    
    Returns:
        Tuple[List[Chunk], Dict[str, int]]: (chunks, token_stats)
    """
    # Set API key for this worker process if provided
    if api_key and enable_llm:
        try:
            set_openai_api_key(api_key)
            logger.info(f"🔑 API key configured for worker process processing {file_path.name}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to set API key in worker process: {e}")
    
    # Initialize a processor for this worker
    processor = DocumentProcessor(input_dir, output_dir, use_cache, enable_llm)
    
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


def setup_openai_api_key(require_key: bool = False) -> bool:
    """Setup OpenAI API key for table processing with LLM metadata generation."""
    try:
        if has_openai_api_key():
            logger.info("✅ OpenAI API key already available for this session")
            return True
        
        if require_key:
            logger.info("🔑 OpenAI API key required for LLM-powered table metadata generation")
            logger.info("📋 This will generate table titles, summaries, and classifications")
            
            # Prompt for API key using secure method
            api_key = get_openai_api_key()
            logger.info("✅ OpenAI API key configured successfully")
            return True
        else:
            logger.info("🔑 OpenAI API key will be requested for enhanced table metadata")
            logger.info("💡 Press Enter to skip if you don't want to provide an API key")
            try:
                # Try to get API key but don't fail if unavailable
                api_key = get_openai_api_key()
                logger.info("✅ OpenAI API key configured successfully")
                return True
            except Exception as e:
                logger.info("ℹ️  No OpenAI API key provided - table processing will skip LLM metadata generation")
                return False
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  API key setup cancelled by user")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to setup OpenAI API key: {e}")
        return False


def upload_to_milvus(chunks_file: Path, config_type: str = "production") -> bool:
    """
    Upload processed chunks to Milvus vector store.
    
    Args:
        chunks_file: Path to the processed chunks JSON file
        config_type: Milvus configuration type to use
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        import json
        
        # Load configuration
        config = get_config(config_type)
        logger.info(f"📡 Connecting to Milvus at {config.host}:{config.port}")
        logger.info(f"📂 Target collection: {config.collection_name}")
        
        # Load chunks
        logger.info(f"📥 Loading chunks from: {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get("chunks", [])
        chunks_with_embeddings = [c for c in chunks if "embedding" in c and c["embedding"]]
        
        logger.info(f"📊 Total chunks: {len(chunks)}")
        logger.info(f"📊 Chunks with embeddings: {len(chunks_with_embeddings)}")
        
        if not chunks_with_embeddings:
            logger.warning("⚠️ No chunks with embeddings found - nothing to upload")
            return False
        
        # Connect to Milvus
        store = MilvusVectorStore(config)
        
        if not store.connect():
            logger.error("❌ Failed to connect to Milvus")
            return False
        
        # Create collection and index
        logger.info("🏗️ Creating collection...")
        if not store.create_collection():
            logger.error("❌ Failed to create collection")
            return False
        
        logger.info("🔍 Creating index...")
        if not store.create_index():
            logger.error("❌ Failed to create index")
            return False
        
        # Convert and upload chunks
        logger.info("🔄 Converting chunks to Milvus format...")
        milvus_chunks = []
        for chunk in chunks_with_embeddings:
            milvus_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "content": chunk["content"][:65535],  # Milvus string limit
                "word_count": chunk["word_count"],
                "section_path": str(chunk.get("section_path", "")),
                "embedding": chunk["embedding"]
            })
        
        # Upload in batches
        batch_size = 200
        total_uploaded = 0
        
        for i in range(0, len(milvus_chunks), batch_size):
            batch = milvus_chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(milvus_chunks) + batch_size - 1) // batch_size
            
            logger.info(f"📤 Uploading batch {batch_num}/{total_batches}: {len(batch)} chunks")
            
            success = store.insert_chunks(batch)
            if success:
                total_uploaded += len(batch)
            else:
                logger.error(f"❌ Failed to upload batch {batch_num}")
                return False
        
        # Get final collection stats
        entity_count = store.get_entity_count()
        logger.info(f"🎯 Upload complete! {total_uploaded} chunks uploaded")
        logger.info(f"📊 Collection now contains {entity_count:,} entities")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during Milvus upload: {str(e)}")
        return False
    finally:
        # Cleanup connection
        if 'store' in locals() and hasattr(store, 'disconnect'):
            store.disconnect()


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
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip OpenAI API key setup and disable LLM metadata generation"
    )
    parser.add_argument(
        "--require-llm",
        action="store_true", 
        help="Require OpenAI API key for LLM metadata generation (prompt if not available)"
    )
    parser.add_argument(
        "--clear-milvus",
        action="store_true",
        help="Clear all embeddings from Milvus before processing"
    )
    parser.add_argument(
        "--drop-milvus",
        action="store_true", 
        help="Drop entire Milvus collection before processing (more thorough than --clear-milvus)"
    )
    parser.add_argument(
        "--confirm-milvus-cleanup",
        action="store_true",
        help="Skip confirmation prompt for Milvus cleanup operations"
    )
    parser.add_argument(
        "--skip-milvus-upload",
        action="store_true",
        help="Skip automatic upload of embeddings to Milvus vector store"
    )
    parser.add_argument(
        "--milvus-config",
        default="production",
        help="Milvus configuration type to use (default: production)"
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
    
    # Handle Milvus cleanup
    if args.clear_milvus or args.drop_milvus:
        logger.info("🧹 Milvus cleanup requested...")
        
        try:
            if args.drop_milvus:
                logger.info("🗑️  Dropping entire Milvus collection...")
                success = drop_milvus_collection("production", confirm=args.confirm_milvus_cleanup)
                if success:
                    logger.info("✅ Successfully dropped Milvus collection")
                else:
                    logger.error("❌ Failed to drop Milvus collection")
                    if not args.confirm_milvus_cleanup:
                        response = input("Continue processing anyway? (y/N): ").strip().lower()
                        if response != 'y':
                            logger.info("Processing cancelled by user")
                            return
            
            elif args.clear_milvus:
                logger.info("🧽 Clearing Milvus collection data...")
                success = clear_milvus_collection("production", confirm=args.confirm_milvus_cleanup)
                if success:
                    logger.info("✅ Successfully cleared Milvus collection")
                else:
                    logger.error("❌ Failed to clear Milvus collection")
                    if not args.confirm_milvus_cleanup:
                        response = input("Continue processing anyway? (y/N): ").strip().lower()
                        if response != 'y':
                            logger.info("Processing cancelled by user")
                            return
                            
        except Exception as e:
            logger.error(f"❌ Error during Milvus cleanup: {e}")
            if not args.confirm_milvus_cleanup:
                response = input("Continue processing anyway? (y/N): ").strip().lower()
                if response != 'y':
                    logger.info("Processing cancelled due to cleanup error")
                    return
    
    logger.info("Starting Graph-RAG Document Processing Pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Setup OpenAI API key for enhanced metadata generation
    llm_available = False
    if not args.no_llm:
        logger.info("🤖 Setting up LLM integration for enhanced table metadata...")
        logger.info("🏷️  This will generate table titles, summaries, and classifications")
        
        # Try to setup API key, but don't fail if unavailable
        if args.require_llm:
            logger.info("🔑 API key required (--require-llm flag)")
            llm_available = setup_openai_api_key(require_key=True)
            if not llm_available:
                logger.error("❌ LLM integration required but API key setup failed.")
                logger.info("💡 Remove --require-llm flag to continue without LLM features.")
                return
        else:
            llm_available = setup_openai_api_key(require_key=False)
            
        if llm_available:
            logger.info("✅ LLM integration enabled - tables will get enhanced metadata")
        else:
            logger.info("📋 LLM unavailable - tables will use basic metadata only")
    else:
        logger.info("🚫 LLM integration disabled (--no-llm flag)")
    
    # Initialize processor
    processor = DocumentProcessor(input_dir=args.input_dir, use_cache=not args.clear_cache, enable_llm=llm_available)
    
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
        
        # Add optional metadata fields if present
        if chunk.metadata.page is not None:
            chunk_data["page"] = chunk.metadata.page
            
        # Add lists if not empty
        if chunk.metadata.outbound_refs:
            chunk_data["outbound_refs"] = [ref.model_dump() for ref in chunk.metadata.outbound_refs]
        if chunk.metadata.inbound_refs:
            chunk_data["inbound_refs"] = chunk.metadata.inbound_refs
        if chunk.metadata.regions:
            chunk_data["regions"] = chunk.metadata.regions
        if chunk.metadata.metrics:
            chunk_data["metrics"] = chunk.metadata.metrics
        if chunk.metadata.time_periods:
            chunk_data["time_periods"] = chunk.metadata.time_periods
        if chunk.metadata.dates:
            chunk_data["dates"] = chunk.metadata.dates
            
        # Add table-specific metadata for table chunks
        if chunk.metadata.chunk_type == "table":
            if chunk.metadata.table_id:
                chunk_data["table_id"] = chunk.metadata.table_id
            if chunk.metadata.column_headers:
                chunk_data["column_headers"] = chunk.metadata.column_headers
            if chunk.metadata.table_title:
                chunk_data["table_title"] = chunk.metadata.table_title
            if chunk.metadata.table_caption:
                chunk_data["table_caption"] = chunk.metadata.table_caption
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
    
    # Upload to Milvus vector store
    if not args.skip_milvus_upload and len(chunks) > 0:
        logger.info("📤 Uploading embeddings to Milvus vector store...")
        upload_success = upload_to_milvus(chunks_file, args.milvus_config)
        if upload_success:
            logger.info("✅ Successfully uploaded embeddings to Milvus")
        else:
            logger.warning("⚠️ Failed to upload embeddings to Milvus - continuing anyway")
    elif args.skip_milvus_upload:
        logger.info("🚫 Skipping Milvus upload (--skip-milvus-upload flag)")
    elif len(chunks) == 0:
        logger.warning("⚠️ No chunks to upload to Milvus")
    
    logger.info(f"Generated {len(chunks)} chunks with full pipeline processing")
    logger.info(f"Total non-API tokens: {processor.token_tracker.get_total():,}")
    if processor.token_tracker.get_openai_total() > 0:
        logger.info(f"OpenAI API tokens: {processor.token_tracker.get_openai_total():,}")
        logger.info(f"OpenAI API cost: ${processor.token_tracker.openai_cost:.6f}")
    else:
        logger.info(f"OpenAI API tokens: 0 (no API calls made)")


if __name__ == "__main__":
    main()