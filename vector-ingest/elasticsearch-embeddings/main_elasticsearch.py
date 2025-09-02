#!/usr/bin/env python3
"""
Graph-RAG Document Processing Pipeline - Elasticsearch Edition
Main entry point for document chunking with Elasticsearch vector storage.
"""

import logging
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import argparse

# Add src to path (relative to parent directory)
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.chunking.models import Chunk, DocumentMetadata, DocumentStructure, ChunkMetadata, StructuralMetadata
from src.chunking.processors.toc_detector import TableOfContentsDetector
from src.chunking.processors.text_chunker import TextChunker
from src.chunking.processors import PostProcessor, ChunkCleaner, TableProcessor
from src.chunking.processors.entity_extractor import EntityExtractor
from src.chunking.metadata_enrichment.spacy_extractor import create_spacy_extractor
from src.chunking.processors.llm_utils import get_openai_api_key, has_openai_api_key, clear_openai_api_key, set_openai_api_key
from src.embeddings import EmbeddingService
from elasticsearch_client import create_elasticsearch_store

# Elasticsearch Configuration
ELASTICSEARCH_CONFIG = {
    "url": "https://1600c6e333fd4bdb8c8e9b9dec5c5fef.us-west-2.aws.found.io:443",
    "username": "elastic", 
    "password": "XI6rIccvUKLCgVnX11QPI8CV",
    "inference_id": "text-vectorizer"
}

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
            
        # Initialize spaCy extractor with error handling
        try:
            self.spacy_extractor = create_spacy_extractor()
            logger.info("✅ spaCy extractor initialized")
        except Exception as e:
            logger.warning(f"⚠️ spaCy unavailable: {e}")
            self.spacy_extractor = None
        
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
                # Load basic metadata fields
                cached_meta = chunk_data['metadata']
                metadata = ChunkMetadata(
                    chunk_id=cached_meta['chunk_id'],
                    doc_id=cached_meta['doc_id'],
                    chunk_type=cached_meta['chunk_type'],
                    word_count=cached_meta['word_count'],
                    section_path=cached_meta.get('section_path', []),
                    page=cached_meta.get('page'),
                    
                    # Load entity fields
                    regions=cached_meta.get('regions', []),
                    metrics=cached_meta.get('metrics', []),
                    time_periods=cached_meta.get('time_periods', []),
                    dates=cached_meta.get('dates', []),
                    
                    # Load table-specific fields
                    table_id=cached_meta.get('table_id'),
                    column_headers=cached_meta.get('column_headers', []),
                    table_title=cached_meta.get('table_title'),
                    table_caption=cached_meta.get('table_caption'),
                    
                    # Load structural metadata if present
                    structural_metadata=StructuralMetadata(**cached_meta['structural_metadata']) if cached_meta.get('structural_metadata') else None
                )
                
                chunk = Chunk(metadata=metadata, content=chunk_data['content'])
                if 'embedding' in chunk_data:
                    chunk.embedding = chunk_data['embedding']
                # Load spaCy extraction data efficiently  
                spacy_data = chunk_data.get('spacy_extraction')
                if spacy_data:
                    chunk.spacy_extraction = spacy_data
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
                
                # Add structural metadata if present
                if hasattr(chunk.metadata, 'structural_metadata') and chunk.metadata.structural_metadata:
                    metadata['structural_metadata'] = chunk.metadata.structural_metadata.model_dump()
                
                chunk_data = {
                    'content': chunk.content,
                    'metadata': metadata
                }
                
                if hasattr(chunk, 'embedding') and chunk.embedding:
                    chunk_data['embedding'] = chunk.embedding
                
                # Save spaCy extraction data efficiently
                spacy_data = getattr(chunk, 'spacy_extraction', None)
                if spacy_data:
                    chunk_data['spacy_extraction'] = spacy_data
                    
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
            
            # Initialize spaCy results collection
            if not hasattr(self, '_spacy_results'):
                self._spacy_results = {}
            
            # Temporarily disable multiprocessing to debug freezing issue
            logger.info(f"Processing {len(files_to_process)} files sequentially (debugging mode)")
            
            # Sequential processing for debugging
            for file_path in files_to_process:
                try:
                    chunks, token_stats, spacy_results = process_single_document(file_path, self.input_dir, self.output_dir, self.use_cache, self.enable_llm, None)
                    all_chunks.extend(chunks)
                    
                    # Merge spaCy results from worker process
                    self._spacy_results.update(spacy_results)
                    
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
        
        # Extract folder path for chunks
        folder_path = self._extract_folder_path(file_path)
        
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
                table_chunks = self._process_tables(json_file, file_path, metadata, folder_path)
            else:
                logger.info(f"ℹ️  No JSON file found at {json_file} - skipping table processing")
        else:
            logger.info("ℹ️  No file_path provided - skipping table processing")
        
        # 5. Create initial chunks using TextChunker
        logger.info("🔪 Creating text chunks...")
        raw_chunks = self._create_raw_chunks(cleaned_content, structure, metadata, folder_path)
        
        # 5. Post-process chunks (clean and merge small chunks)
        logger.info("🧹 Post-processing chunks...")
        processed_chunks = self._post_process_chunks(raw_chunks)
        
        # 6. Combine all chunks (text + table)
        all_chunks = processed_chunks + table_chunks
        logger.info(f"Combined {len(processed_chunks)} text chunks + {len(table_chunks)} table chunks = {len(all_chunks)} total")
        
        # 7. Extract entities and populate metadata
        logger.info("🏷️  Extracting entities...")
        chunks_with_entities = self._extract_entities(all_chunks, folder_path)
        
        # 7.1. Add spaCy entity extraction
        if self.spacy_extractor:
            logger.info("🧠 Adding spaCy entity extraction...")
            chunks_with_entities = self._add_spacy_extraction(chunks_with_entities)
        
        # 8. Enrich with structural metadata (Phase 1)
        logger.info("🏗️  Enriching with structural metadata...")
        from src.chunking.metadata_enrichment import enrich_chunks_with_structure
        json_file_path = file_path.with_suffix('.json') if file_path else None
        json_content = None
        if json_file_path and json_file_path.exists():
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    json_content = f.read()
            except Exception as e:
                logger.debug(f"Failed to read JSON file: {e}")
        enriched_chunks = enrich_chunks_with_structure(chunks_with_entities, {'doc_id': metadata.doc_id, 'document_structure': structure, 'md_content': cleaned_content, 'json_content': json_content})
        
        # 9. Generate embeddings
        logger.info("🔮 Generating embeddings...")
        final_chunks = self._generate_embeddings(enriched_chunks)
        
        return final_chunks
    
    def _discover_input_files(self, input_dir: Path) -> List[Path]:
        """Discover all supported files recursively in input directory and any nested subdirectories."""
        if not input_dir.exists():
            return []
        
        supported_exts = {'.txt', '.md', '.html', '.htm'}
        files = []
        
        # Recursively find files in all nested subdirectories
        # Uses rglob('*') to handle unlimited nesting depth efficiently
        try:
            for file_path in input_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_exts:
                    files.append(file_path)
        except (OSError, PermissionError) as e:
            logger.warning(f"⚠️ Error accessing some directories: {e}")
        
        logger.info(f"📁 Discovered {len(files)} files across all nested directories")
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
    
    def _extract_folder_path(self, file_path: Path) -> List[str]:
        """Extract folder hierarchy path from file path relative to input directory."""
        try:
            # Get relative path from input directory
            relative_path = file_path.relative_to(self.input_dir)
            # Extract folder parts (excluding the filename)
            folder_parts = list(relative_path.parent.parts)
            # Filter out '.' for current directory
            return [part for part in folder_parts if part != '.']
        except ValueError:
            # If file is not relative to input_dir, use absolute path parts
            return list(file_path.parent.parts)
    
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
    
    def _process_tables(self, json_file: Path, md_file: Path, metadata: DocumentMetadata, folder_path: List[str]) -> List[Chunk]:
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
    
    def _create_raw_chunks(self, content: str, structure: DocumentStructure, doc_metadata: DocumentMetadata, folder_path: List[str]) -> List[Chunk]:
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
                section_path=chunk_dict.get('section_path', []),
                # Add new required fields for text chunks
                product_version="v1",
                folder_path=folder_path
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
    
    def _extract_entities(self, chunks: List[Chunk], folder_path: List[str] = None) -> List[Chunk]:
        """Optimized entity extraction with document-level org extraction."""
        if not chunks:
            return chunks
        
        # Extract document-level organizations once (not per chunk)
        doc_orgs = []
        if chunks:
            doc_id = chunks[0].metadata.doc_id
            sample_content = chunks[0].content
            doc_orgs = self._extract_document_organizations(doc_id, sample_content)
        
        for chunk in chunks:
            # Extract entities from content
            entities = self.entity_extractor.process(chunk.content)
            
            # Pre-extract entity lists once to avoid repeated dict lookups
            people = entities.get('people')
            organizations = entities.get('organizations') 
            dates = entities.get('dates')
            time_periods = entities.get('time_periods')
            regions = entities.get('regions')
            metrics = entities.get('metrics')
            
            # Populate dedicated metadata fields correctly
            if regions:
                chunk.metadata.regions.extend(list(regions)[:5])
            if metrics:
                chunk.metadata.metrics.extend(list(metrics)[:5])
            if dates:
                chunk.metadata.dates.extend(list(dates)[:5])
            if time_periods:
                chunk.metadata.time_periods.extend(list(time_periods)[:5])
            
            # Populate new required fields efficiently
            # Organizations - use entity extraction or document-level fallback
            orgs_to_add = list(organizations)[:5] if organizations else doc_orgs
            if orgs_to_add:
                chunk.metadata.orgs.extend(orgs_to_add)
            
            # Time context - only calculate if dates exist
            if dates:
                chunk.metadata.time_context = self._extract_time_context(list(dates))
            
            # Add folder_path to all chunks (both text and table)
            if folder_path:
                chunk.metadata.folder_path = folder_path
            
            # Create section_path summary only if needed and entities exist
            if people or organizations or dates:
                entity_info = []
                # Build strings efficiently with slice and join in one operation
                if people:
                    entity_info.append(f"People: {', '.join(list(people)[:3])}")
                if organizations:
                    entity_info.append(f"Orgs: {', '.join(list(organizations)[:3])}")
                if dates:
                    entity_info.append(f"Dates: {', '.join(list(dates)[:3])}")
                
                # Set section_path if not already set and we have entity info
                if not chunk.metadata.section_path and entity_info:
                    chunk.metadata.section_path = [entity_info[0]]  # Single element list
        
        logger.info(f"🏷️  Extracted entities for {len(chunks)} chunks")
        return chunks
    
    def _extract_time_context(self, dates: List[str]) -> Optional[Dict[str, str]]:
        """Optimized time context extraction with simple string operations."""
        if not dates:
            return None
        
        # Extract years efficiently using simple string operations
        fiscal_years = []
        for date_str in dates:
            date_lower = date_str.lower()
            
            # Look for 4-digit years
            for i in range(len(date_str) - 3):
                if date_str[i:i+4].isdigit():
                    year = int(date_str[i:i+4])
                    if 1900 <= year <= 2030:  # Reasonable year range
                        fiscal_years.append(year)
                        break
        
        if fiscal_years:
            min_year, max_year = min(fiscal_years), max(fiscal_years)
            return {
                "start": f"{min_year-1}-05-01",
                "end": f"{max_year}-04-30", 
                "granularity": "fiscal_year"
            }
        
        return None
    
    def _extract_document_organizations(self, doc_id: str, content: str) -> List[str]:
        """Simple organization extraction with minimal processing."""
        # Fast document-based organization detection
        if 'elastic' in doc_id.lower():
            return ['Elastic N.V.']
        
        # Simple content-based detection (avoid expensive regex)
        content_lower = content.lower()
        if 'elastic' in content_lower:
            return ['Elastic N.V.']
        elif 'company' in content_lower and 'inc' in content_lower:
            return ['The Company']
        
        return []
    
    def _add_spacy_extraction(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optimized spaCy entity extraction with efficient batch processing."""
        if not chunks or not self.spacy_extractor:
            return chunks
        
        start_time = time.time()
        chunk_count = len(chunks)
        
        # Pre-allocate results dictionary for O(1) access
        if not hasattr(self, '_spacy_results'):
            self._spacy_results = {}
        
        # Pre-allocate empty result for reuse
        empty_result = {"organizations": [], "locations": [], "products": [], "events": []}
        total_entities = 0
        
        # Optimized batch processing with early termination for empty chunks
        for chunk in chunks:
            content = chunk.content
            chunk_id = chunk.metadata.chunk_id
            
            # Skip very short chunks (< 50 chars) for efficiency
            if len(content) < 50:
                self._spacy_results[chunk_id] = empty_result.copy()
                continue
            
            try:
                # Extract 4 clean entity types using optimized spaCy
                spacy_result = self.spacy_extractor.process_chunk_content(content)
                self._spacy_results[chunk_id] = spacy_result
                
                # Fast entity counting using generator expression
                total_entities += sum(len(entities) for entities in spacy_result.values())
                
            except Exception as e:
                logger.warning(f"spaCy extraction failed for {chunk_id}: {e}")
                self._spacy_results[chunk_id] = empty_result.copy()
        
        processing_time = time.time() - start_time
        
        logger.info(f"🧠 spaCy extraction: {total_entities} entities from {chunk_count} chunks in {processing_time:.2f}s")
        logger.info(f"⚡ Performance: {processing_time/chunk_count*1000:.1f}ms per chunk")
        
        return chunks
    
    def _generate_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for all chunks."""
        if not chunks:
            return chunks
        
        # Optimized token estimation (avoid generator expression overhead)
        embedding_tokens = 0
        for chunk in chunks:
            embedding_tokens += self.token_tracker.estimate_tokens(chunk.content)
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
    
    # Return chunks, token statistics, and spaCy results
    token_stats = {
        'input_tokens': processor.token_tracker.input_tokens,
        'processing_tokens': processor.token_tracker.processing_tokens,
        'embedding_tokens': processor.token_tracker.embedding_tokens
    }
    
    # Get spaCy results from processor
    spacy_results = getattr(processor, '_spacy_results', {})
    
    # Save to cache if enabled
    if use_cache and chunks:
        processor._save_to_cache(file_path, chunks)
    
    return chunks, token_stats, spacy_results


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


def upload_to_elasticsearch(chunks_file: Path) -> bool:
    """
    Upload processed chunks to Elasticsearch vector store.
    
    Args:
        chunks_file: Path to the processed chunks JSON file
        
    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
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
        
        # Connect to Elasticsearch
        logger.info(f"📡 Connecting to Elasticsearch at {ELASTICSEARCH_CONFIG['url']}")
        store = create_elasticsearch_store(ELASTICSEARCH_CONFIG)
        
        if not store.connect():
            logger.error("❌ Failed to connect to Elasticsearch")
            return False
        
        # Create index with proper mapping
        logger.info("🏗️ Creating Elasticsearch index...")
        if not store.create_index():
            logger.error("❌ Failed to create Elasticsearch index")
            return False
        
        # Upload chunks
        success = store.upload_chunks(chunks_with_embeddings)
        
        if success:
            doc_count = store.get_document_count()
            logger.info(f"🎯 Upload complete! Elasticsearch index now contains {doc_count:,} documents")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error during Elasticsearch upload: {str(e)}")
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
        "--clear-elasticsearch",
        action="store_true",
        help="Clear all documents from Elasticsearch index before processing"
    )
    parser.add_argument(
        "--drop-elasticsearch",
        action="store_true", 
        help="Drop entire Elasticsearch index before processing (more thorough than --clear-elasticsearch)"
    )
    parser.add_argument(
        "--confirm-elasticsearch-cleanup",
        action="store_true",
        help="Skip confirmation prompt for Elasticsearch cleanup operations"
    )
    parser.add_argument(
        "--skip-elasticsearch-upload",
        action="store_true",
        help="Skip automatic upload of embeddings to Elasticsearch vector store"
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
    
    # Handle Elasticsearch cleanup
    if args.clear_elasticsearch or args.drop_elasticsearch:
        logger.info("🧹 Elasticsearch cleanup requested...")
        
        try:
            store = create_elasticsearch_store(ELASTICSEARCH_CONFIG)
            if not store.connect():
                logger.error("❌ Failed to connect to Elasticsearch for cleanup")
                return
            
            if args.drop_elasticsearch:
                logger.info("🗑️  Dropping entire Elasticsearch index...")
                success = store.delete_index()
                if success:
                    logger.info("✅ Successfully dropped Elasticsearch index")
                else:
                    logger.error("❌ Failed to drop Elasticsearch index")
                    if not args.confirm_elasticsearch_cleanup:
                        response = input("Continue processing anyway? (y/N): ").strip().lower()
                        if response != 'y':
                            logger.info("Processing cancelled by user")
                            return
            
            elif args.clear_elasticsearch:
                logger.info("🧽 Clearing Elasticsearch index data...")
                success = store.clear_index()
                if success:
                    logger.info("✅ Successfully cleared Elasticsearch index")
                else:
                    logger.error("❌ Failed to clear Elasticsearch index")
                    if not args.confirm_elasticsearch_cleanup:
                        response = input("Continue processing anyway? (y/N): ").strip().lower()
                        if response != 'y':
                            logger.info("Processing cancelled by user")
                            return
            
            store.disconnect()
                            
        except Exception as e:
            logger.error(f"❌ Error during Elasticsearch cleanup: {e}")
            if not args.confirm_elasticsearch_cleanup:
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
        
        # Cache metadata reference to avoid repeated attribute access
        metadata = chunk.metadata
        
        # Add optional metadata fields if present
        if metadata.page is not None:
            chunk_data["page"] = metadata.page
            
        # Batch process list fields with single attribute access
        list_fields = [
            ("outbound_refs", [ref.model_dump() for ref in metadata.outbound_refs] if metadata.outbound_refs else None),
            ("inbound_refs", metadata.inbound_refs if metadata.inbound_refs else None),
            ("regions", metadata.regions if metadata.regions else None),
            ("metrics", metadata.metrics if metadata.metrics else None),
            ("time_periods", metadata.time_periods if metadata.time_periods else None),
            ("dates", metadata.dates if metadata.dates else None),
            ("orgs", metadata.orgs if metadata.orgs else None)
        ]
        
        # Add non-empty lists in single loop
        for field_name, field_value in list_fields:
            if field_value:
                chunk_data[field_name] = field_value
        
        # Add universal required fields for all chunks
        if hasattr(metadata, 'time_context') and metadata.time_context:
            chunk_data["time_context"] = metadata.time_context
        
        # Always add product_version
        chunk_data["product_version"] = getattr(metadata, 'product_version', 'v1')
        
        # Add folder_path (always save, even if empty for root directory)
        if hasattr(metadata, 'folder_path'):
            chunk_data["folder_path"] = metadata.folder_path or []
            
        # Add table-specific metadata for table chunks
        if metadata.chunk_type == "table":
            table_fields = [
                ("table_id", metadata.table_id),
                ("column_headers", metadata.column_headers), 
                ("table_title", metadata.table_title),
                ("table_shape", metadata.table_shape)
            ]
            
            for field_name, field_value in table_fields:
                if field_value:
                    chunk_data[field_name] = field_value
        
        # Add Phase 1 structural metadata if present
        if hasattr(chunk.metadata, 'structural_metadata') and chunk.metadata.structural_metadata:
            chunk_data["structural_metadata"] = {
                "element_type": chunk.metadata.structural_metadata.element_type,
                "element_level": chunk.metadata.structural_metadata.element_level,
                "page_number": chunk.metadata.structural_metadata.page_number,
                "bbox_coords": chunk.metadata.structural_metadata.bbox_coords,
                "is_heading": chunk.metadata.structural_metadata.is_heading
            }
        
        # Add spaCy extraction results efficiently
        spacy_results = getattr(processor, '_spacy_results', None)
        if spacy_results:
            spacy_data = spacy_results.get(chunk.metadata.chunk_id)
            if spacy_data:
                chunk_data["spacy_extraction"] = spacy_data
        
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