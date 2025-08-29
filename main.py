#!/usr/bin/env python3
"""
Graph-RAG Document Processing Pipeline
Main entry point for document chunking with TOC detection.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.chunking.models import Chunk, DocumentMetadata, DocumentStructure, ChunkMetadata
from src.chunking.processors import TextPreprocessor
from src.chunking.processors.toc_detector import TableOfContentsDetector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Streamlined document processing with integrated TOC detection."""
    
    def __init__(self, input_dir: Path = None):
        self.preprocessor = TextPreprocessor()
        self.toc_detector = TableOfContentsDetector()
        self.input_dir = input_dir or Path.cwd() / "input"
        
        # File type mapping for efficiency
        self.type_map = {
            '.pdf': 'pdf', '.docx': 'docx', '.doc': 'docx',
            '.html': 'html', '.htm': 'html', '.md': 'md', '.txt': 'md'
        }
    
    def process_all_documents(self) -> List[Chunk]:
        """Process all documents in input directory."""
        # Ensure input directory exists
        if not self.input_dir.exists():
            self.input_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created input directory: {self.input_dir}")
            logger.info(f"Please place your documents in {self.input_dir} and run again.")
            return []
        
        files = self.preprocessor.discover_input_files(self.input_dir)
        
        if not files:
            logger.warning(f"No supported files found in {self.input_dir}")
            logger.info("Supported formats: PDF, DOCX, HTML, MD, TXT")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        
        all_chunks = []
        for file_path in files:
            try:
                logger.info(f"Processing {file_path.name}")
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
                logger.info(f"Generated {len(chunks)} chunks from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                continue
        
        logger.info(f"Total chunks generated: {len(all_chunks)}")
        return all_chunks
    
    def process_document(self, file_path: Path) -> List[Chunk]:
        """Process a single document with TOC detection and chunking."""
        # 1. Preprocess content
        logger.info(f"Preprocessing {file_path.name}...")
        cleaned_content = self.preprocessor.preprocess_file(file_path)
        
        if not cleaned_content.strip():
            logger.warning(f"No content after preprocessing: {file_path.name}")
            return []
        
        logger.info(f"Content length after preprocessing: {len(cleaned_content)} characters")
        
        # 2. Create document metadata
        metadata = self._create_document_metadata(file_path)
        
        # 3. Detect TOC and analyze structure
        logger.info("Analyzing document structure...")
        structure = self._analyze_structure(cleaned_content)
        
        # 4. Create chunks based on structure
        chunks = self._create_chunks(cleaned_content, structure, metadata)
        
        return chunks
    
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
    
    def _analyze_structure(self, content: str) -> DocumentStructure:
        """Analyze document structure using our systematic TOC detector."""
        logger.info("🔍 Detecting Table of Contents...")
        
        # Use our systematic TOC detector
        toc_entries = self.toc_detector.detect_toc(content)
        
        if toc_entries:
            logger.info(f"✅ Found TOC with {len(toc_entries)} entries")
            
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
    
    def _create_chunks(self, content: str, structure: DocumentStructure, doc_metadata: DocumentMetadata) -> List[Chunk]:
        """Create chunks from content and structure."""
        if not content.strip():
            return []
        
        chunks = []
        
        # For now, create simple chunks
        # TODO: Implement intelligent chunking based on TOC structure
        if structure.toc_sections:
            # If we have TOC, we could potentially split by sections
            logger.info("📝 Creating chunks based on document structure")
            # For now, still create one chunk but mark it as structured
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{doc_metadata.doc_id}_chunk_structured",
                doc_id=doc_metadata.doc_id,
                chunk_type="text",
                word_count=len(content.split()),
                section_path=[f"Structured document with {len(structure.toc_sections)} TOC entries"]
            )
        else:
            # No structure detected, create basic chunk
            logger.info("📝 Creating basic text chunk")
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{doc_metadata.doc_id}_chunk_basic",
                doc_id=doc_metadata.doc_id,
                chunk_type="text",
                word_count=len(content.split())
            )
        
        chunks.append(Chunk(metadata=chunk_metadata, content=content))
        
        return chunks


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
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 Starting Graph-RAG Document Processing Pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize processor
    processor = DocumentProcessor(input_dir=args.input_dir)
    
    # Process all documents
    chunks = processor.process_all_documents()
    
    if not chunks:
        logger.warning("No chunks generated. Check your input directory and files.")
        return
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # For now, just save a summary
    summary_file = args.output_dir / "processing_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Document Processing Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Total chunks generated: {len(chunks)}\n\n")
        
        for chunk in chunks:
            f.write(f"Chunk ID: {chunk.metadata.chunk_id}\n")
            f.write(f"Document: {chunk.metadata.doc_id}\n")
            f.write(f"Type: {chunk.metadata.chunk_type}\n")
            f.write(f"Word count: {chunk.metadata.word_count}\n")
            f.write(f"Section path: {', '.join(chunk.metadata.section_path) if chunk.metadata.section_path else 'None'}\n")
            f.write(f"Content preview: {chunk.content[:200]}...\n")
            f.write("-" * 50 + "\n\n")
    
    logger.info(f"✅ Processing complete! Summary saved to {summary_file}")
    logger.info("🎯 TOC detection system is ready for LLM verification when needed.")


if __name__ == "__main__":
    main()