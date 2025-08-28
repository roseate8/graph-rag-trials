from typing import List, Dict, Any
from pathlib import Path
import logging

from .models import Chunk, DocumentMetadata, DocumentStructure, ChunkMetadata
from .preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Core document processing and chunking service."""
    
    def __init__(self, input_dir: Path = None):
        self.preprocessor = TextPreprocessor()
        self.input_dir = input_dir or Path.cwd() / "input"
        
        # File type mapping for efficiency
        self.type_map = {
            '.pdf': 'pdf', '.docx': 'docx', '.doc': 'docx',
            '.html': 'html', '.htm': 'html', '.md': 'md', '.txt': 'md'
        }
    
    def process_all_documents(self) -> List[Chunk]:
        """Process all documents in input directory."""
        files = self.preprocessor.discover_input_files(self.input_dir)
        
        if not files:
            logger.warning(f"No supported files found in {self.input_dir}")
            return []
        
        all_chunks = []
        for file_path in files:
            try:
                logger.info(f"Processing {file_path.name}")
                chunks = self.process_document(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {str(e)}")
                continue
        
        return all_chunks
    
    def process_document(self, file_path: Path) -> List[Chunk]:
        """Process a single document and return chunks with metadata."""
        # 1. Preprocess content
        cleaned_content = self.preprocessor.preprocess_file(file_path)
        
        if not cleaned_content.strip():
            logger.warning(f"No content after preprocessing: {file_path.name}")
            return []
        
        # 2. Create document metadata
        metadata = self._create_document_metadata(file_path)
        
        # 3. Analyze structure (placeholder for now)
        structure = self._analyze_structure(cleaned_content)
        
        # 4. Create chunks (simplified for preprocessing focus)
        chunks = self._chunk_content(cleaned_content, structure, metadata)
        
        return chunks
    
    def _create_document_metadata(self, file_path: Path) -> DocumentMetadata:
        """Create document metadata from file efficiently."""
        return DocumentMetadata(
            doc_id=file_path.stem,
            title=file_path.stem.replace('_', ' ').replace('-', ' ').title(),
            source_type=self.type_map.get(file_path.suffix.lower(), 'md'),
            author=None,
            date=None,
            page_count=None
        )
    
    
    def _analyze_structure(self, content: str) -> DocumentStructure:
        """Analyze document structure including ToC, headings, tables."""
        # Placeholder - return empty structure for now
        return DocumentStructure()
    
    def _chunk_content(self, content: str, structure: DocumentStructure, doc_metadata: DocumentMetadata) -> List[Chunk]:
        """Create chunks from processed content and structure."""
        if not content.strip():
            return []
        
        # Simple chunking - one chunk per document for preprocessing focus
        chunk_metadata = ChunkMetadata(
            chunk_id=f"{doc_metadata.doc_id}_chunk_1",
            doc_id=doc_metadata.doc_id,
            chunk_type="text",
            word_count=len(content.split())
        )
        
        return [Chunk(metadata=chunk_metadata, content=content)]
    
    def _detect_toc(self, content: str) -> List[Dict[str, Any]]:
        """Detect table of contents using multi-faceted analysis."""
        raise NotImplementedError
    
    def _chunk_text(self, text: str, section_path: List[str]) -> List[Chunk]:
        """Chunk text content preserving sentence boundaries."""
        raise NotImplementedError
    
    def _chunk_table(self, table_data: Dict[str, Any]) -> Chunk:
        """Process table as single chunk in markdown format."""
        raise NotImplementedError