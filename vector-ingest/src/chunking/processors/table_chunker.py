"""
Advanced table chunking processor using JSON detection + MD extraction strategy.

This module handles:
- Table detection from DoclingDocument JSON with MD content extraction
- JSON-first approach with MD fallback for boundary identification 
- Intelligent table chunking preserving original MD formatting
- Multi-page table handling with continuation metadata
- Large table division with header preservation
- LLM-powered metadata generation for enhanced searchability
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .base import BaseProcessor
from .llm_utils import get_openai_api_key, has_openai_api_key
from ..models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)


@dataclass
class TableBoundary:
    """Represents table boundary information from JSON."""
    table_id: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    bbox: Optional[Dict] = None
    num_rows: int = 0
    num_cols: int = 0
    content_sample: Optional[str] = None  # First few rows for MD matching
    headers: List[str] = None  # Column headers
    row_headers: List[str] = None  # Row headers


@dataclass
class TableExtractionResult:
    """Result of extracting table from MD using JSON boundaries."""
    table_id: str
    md_content: str  # Extracted markdown table
    boundary_info: TableBoundary
    extraction_method: str  # "json_guided", "md_fallback", "content_match"
    confidence: float = 1.0  # How confident we are in the extraction




class JSONTableDetector:
    """Detects table boundaries from DoclingDocument JSON format."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.JSONTableDetector')
    
    def detect_table_boundaries(self, json_data: Dict[str, Any]) -> List[TableBoundary]:
        """Extract table boundary information from DoclingDocument JSON."""
        boundaries = []
        
        self.logger.info("🔍 Starting JSON table boundary detection...")
        
        if 'tables' not in json_data:
            self.logger.warning("❌ No 'tables' section found in JSON document")
            return boundaries
        
        table_count = len(json_data['tables'])
        self.logger.info(f"📊 Found {table_count} tables in JSON document")
        
        for i, table_data in enumerate(json_data['tables']):
            try:
                self.logger.debug(f"Processing table {i+1}/{table_count}...")
                boundary = self._extract_boundary_info(table_data, i)
                if boundary:
                    boundaries.append(boundary)
                    self.logger.info(f"✅ Table {i+1}: {boundary.table_id} ({boundary.num_rows}x{boundary.num_cols}) - {len(boundary.headers or [])} headers")
                else:
                    self.logger.warning(f"⚠️  Table {i+1}: Failed to extract boundary info")
            except Exception as e:
                self.logger.error(f"❌ Error extracting boundary for table {i+1}: {e}")
        
        self.logger.info(f"🎯 Successfully detected {len(boundaries)} table boundaries from JSON")
        return boundaries
    
    def _extract_boundary_info(self, table_data: Dict, table_index: int) -> Optional[TableBoundary]:
        """Extract boundary information from individual table in JSON."""
        if 'data' not in table_data:
            return None
        
        data = table_data['data']
        if 'grid' not in data:
            return None
        
        grid = data['grid']
        if not grid:
            return None
        
        # Extract basic info
        num_rows = data.get('num_rows', len(grid))
        num_cols = data.get('num_cols', len(grid[0]) if grid else 0)
        
        # Generate table ID
        table_id = f"table_{table_index}"
        if 'self_ref' in table_data:
            table_id = table_data['self_ref'].replace('#/tables/', 'table_')
        
        # Extract column and row headers for matching
        column_headers = []
        row_headers = []
        
        for row_idx, row in enumerate(grid[:3]):  # First 3 rows for headers
            for col_idx, cell_data in enumerate(row):
                if not cell_data or not cell_data.get('text'):
                    continue
                    
                text = cell_data['text'].strip()
                if not text:
                    continue
                
                # Column headers
                if cell_data.get('column_header', False) or (row_idx == 0 and col_idx > 0):
                    column_headers.append(text)
                    
                # Row headers (typically first column)
                if cell_data.get('row_header', False) or (col_idx == 0 and row_idx > 0):
                    row_headers.append(text)
        
        headers = column_headers  # Keep for backward compatibility
        
        # Create content sample for MD matching (first 2 rows)
        content_sample = []
        for row_idx, row in enumerate(grid[:2]):
            row_texts = []
            for cell_data in row:
                if cell_data and 'text' in cell_data:
                    row_texts.append(cell_data['text'].strip())
                else:
                    row_texts.append('')
            if any(row_texts):  # Only add non-empty rows
                content_sample.append(row_texts)
        
        # Extract bbox and page info if available
        bbox = table_data.get('bbox')
        prov = table_data.get('prov', [])
        pages = set()
        for p in prov:
            if isinstance(p, dict) and 'page' in p:
                pages.add(p['page'])
        
        # Store both column and row headers
        boundary = TableBoundary(
            table_id=table_id,
            start_page=min(pages) if pages else None,
            end_page=max(pages) if pages else None,
            bbox=bbox,
            num_rows=num_rows,
            num_cols=num_cols,
            content_sample=content_sample,
            headers=list(dict.fromkeys(column_headers)),  # Remove duplicates
            row_headers=list(dict.fromkeys(row_headers)) if row_headers else None
        )
        
        return boundary


class TableUnitDetector:
    """Fast unit detection for table columns."""
    
    # Static patterns for O(1) lookup
    UNIT_PATTERNS = {
        'thousands': {'thousands', '(in thousands)', '(000)', 'k$'},
        'millions': {'millions', '(in millions)', '(000,000)', 'm$', 'mm$'},
        'billions': {'billions', '(in billions)', 'b$'},
        'percent': {'%', 'percent', 'pct'},
        'dollars': {'$', 'usd', 'dollars'},
        'shares': {'shares', 'share', 'outstanding'},
        'years': {'year', 'years', 'yr', 'annual'},
        'ratio': {'ratio', 'multiple', 'x'},
    }
    
    @classmethod
    def detect_units_per_column(cls, header_cells: List[str]) -> Dict[str, str]:
        """Fast unit detection using precomputed patterns."""
        units = {}
        
        for col_idx, header in enumerate(header_cells):
            if not header:
                continue
                
            header_lower = header.lower()
            
            # Fast lookup using set intersection
            for unit_type, patterns in cls.UNIT_PATTERNS.items():
                if any(pattern in header_lower for pattern in patterns):
                    units[f"col_{col_idx}"] = unit_type
                    break
        
        return units


class MarkdownTableExtractor:
    """Fast table extraction with preprocessing cache."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.MarkdownTableExtractor')
        self._cache = {'hash': None, 'indices': [], 'normalized': [], 'blocks': []}
    
    def _preprocess(self, md_content: str):
        """Single-pass preprocessing for maximum efficiency."""
        content_hash = hash(md_content)
        if self._cache['hash'] == content_hash:
            return
        
        lines = md_content.split('\n')
        indices, normalized, blocks = [], [], []
        current_block = []
        
        # Single pass through content
        for i, line in enumerate(lines):
            if '|' in line:
                indices.append(i)
                normalized.append([c.strip().lower() for c in line.split('|')[1:-1]])
                
                # Build table blocks simultaneously
                if line.strip().startswith('|') and line.strip().endswith('|'):
                    if not re.match(r'^\s*\|[\s\-\|]*\|\s*$', line):  # Skip separators
                        current_block.append(line)
                elif current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                normalized.append([])
                if current_block:
                    blocks.append(current_block)
                    current_block = []
        
        # Add final block
        if current_block:
            blocks.append(current_block)
        
        self._cache = {'hash': content_hash, 'indices': indices, 'normalized': normalized, 'blocks': blocks, 'lines': lines}
    
    def extract_table_from_md(self, md_content: str, boundary: TableBoundary) -> TableExtractionResult:
        """Streamlined table extraction with dual fallback strategy."""
        self._preprocess(md_content)
        
        if not boundary.headers:
            return self._create_result(boundary, "", "no_headers", 0.0)
        
        # Fast header-based matching
        best_idx, confidence = self._find_best_match(boundary.headers)
        if confidence > 0.7:
            content = self._extract_table_content(best_idx, boundary.num_rows)
            self.logger.info(f"✅ Content-based match for {boundary.table_id} (confidence: {confidence:.2f})")
            return self._create_result(boundary, content, "content_match", confidence)
        
        # Pattern-based fallback
        best_block, confidence = self._find_best_block(boundary)
        if confidence > 0.5:
            content = '\n'.join(best_block)
            self.logger.info(f"✅ Pattern-based match for {boundary.table_id} (confidence: {confidence:.2f})")
            return self._create_result(boundary, content, "md_pattern", confidence)
        
        self.logger.warning(f"❌ No match found for {boundary.table_id}")
        return self._create_result(boundary, "", "failed", 0.0)
    
    def _find_best_match(self, headers: List[str]) -> tuple:
        """Fast header matching using set operations."""
        if not headers:
            return -1, 0.0
        
        headers_lower = {h.lower() for h in headers}
        best_idx, best_conf = -1, 0.0
        
        for idx in self._cache['indices']:
            cells = set(self._cache['normalized'][idx])
            if not cells:
                continue
            
            matches = sum(1 for h in headers_lower if any(h in c or c in h for c in cells))
            confidence = matches / len(headers)
            
            if confidence > best_conf:
                best_conf = confidence
                best_idx = idx
        
        return best_idx, best_conf
    
    def _find_best_block(self, boundary: TableBoundary) -> tuple:
        """Fast block matching using size similarity."""
        best_block, best_score = None, 0.0
        
        for block in self._cache['blocks']:
            if not block:
                continue
            
            rows, cols = len(block), len(block[0].split('|')) - 2
            row_score = min(rows, boundary.num_rows) / max(rows, boundary.num_rows, 1)
            col_score = min(cols, boundary.num_cols) / max(cols, boundary.num_cols, 1)
            score = (row_score + col_score) / 2
            
            if score > best_score:
                best_score, best_block = score, block
        
        return best_block or [], best_score
    
    def _extract_table_content(self, start_idx: int, expected_rows: int) -> str:
        """Extract table lines efficiently."""
        lines = self._cache['lines']
        indices = self._cache['indices']
        
        # Find table boundaries
        start = start_idx
        for idx in reversed(indices):
            if idx < start_idx and idx == start_idx - 1:
                start = idx
            else:
                break
        
        # Collect consecutive table lines
        result = []
        for i in range(start, min(len(lines), start + expected_rows + 5)):
            if '|' in lines[i]:
                result.append(lines[i])
            else:
                break
        
        return '\n'.join(result)
    
    def _create_result(self, boundary: TableBoundary, content: str, method: str, confidence: float) -> TableExtractionResult:
        """Helper to create extraction results."""
        return TableExtractionResult(
            table_id=boundary.table_id,
            md_content=content,
            boundary_info=boundary,
            extraction_method=method,
            confidence=confidence
        )
    
    
    
    
    
    


class LLMTableProcessor:
    """Uses LLM to generate table titles and classifications with detailed token tracking."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.LLMTableProcessor')
        
        # Detailed token tracking - per method
        self.method_tokens = {}  # method_name -> {input, output, calls, cost}
        
        # Per-file token tracking
        self.file_tokens = {}  # file_id -> {input, output, calls, cost}
        
        # Global totals
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.api_calls_count = 0
        
        # OpenAI pricing (per 1M tokens)
        self.pricing = {
            'gpt-4.1-nano': {  # Fixed model name
                'input': 0.1 / 1_000_000,   # $0.15 per 1M input tokens
                'output': 0.40 / 1_000_000   # $0.60 per 1M output tokens
            }
        }
    
    def _track_tokens(self, method_name: str, file_id: str, input_tokens: int, output_tokens: int, call_cost: float):
        """Track tokens for method, file, and global totals."""
        
        # Update method tracking
        if method_name not in self.method_tokens:
            self.method_tokens[method_name] = {'input': 0, 'output': 0, 'calls': 0, 'cost': 0.0}
        
        self.method_tokens[method_name]['input'] += input_tokens
        self.method_tokens[method_name]['output'] += output_tokens
        self.method_tokens[method_name]['calls'] += 1
        self.method_tokens[method_name]['cost'] += call_cost
        
        # Update file tracking
        if file_id not in self.file_tokens:
            self.file_tokens[file_id] = {'input': 0, 'output': 0, 'calls': 0, 'cost': 0.0}
        
        self.file_tokens[file_id]['input'] += input_tokens
        self.file_tokens[file_id]['output'] += output_tokens
        self.file_tokens[file_id]['calls'] += 1
        self.file_tokens[file_id]['cost'] += call_cost
        
        # Update global totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += call_cost
        self.api_calls_count += 1
    
    async def generate_table_metadata(self, table_markdown: str, table_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Generate title, summary, and classification for table using LLM."""
        if not has_openai_api_key():
            self.logger.warning("⚠️  No OpenAI API key available - skipping LLM metadata generation")
            return None, None, None
        
        try:
            import requests
            
            # Truncate table content for LLM processing (max ~150-200 tokens)
            truncated_table = table_markdown[:1000]  # Rough token limit
            input_token_estimate = len(truncated_table) // 4  # Rough estimate
            
            self.logger.info(f"🤖 Generating LLM metadata for table {table_id} (~{input_token_estimate} input tokens)")
            
            prompt = f"""Analyze this table and provide:
1. A concise title (3-8 words)
2. A brief summary (max 30 words) 
3. Classification (choose: financial, metrics, ratios, geographic, time_series, reference_data, other)

Table:
{truncated_table}

Response format:
Title: [your title]
Summary: [your summary] 
Classification: [your classification]"""
            
            api_key = get_openai_api_key()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4.1-nano",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 80,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Extract token usage from response
                usage = result.get('usage', {})
                input_tokens = usage.get('prompt_tokens', 0)
                output_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
                
                # Calculate cost for this call
                call_cost = (
                    input_tokens * self.pricing['gpt-4.1-nano']['input'] +
                    output_tokens * self.pricing['gpt-4.1-nano']['output']
                )
                
                # Extract file ID from table_id (format: table_doc_id_N)
                file_id = "_".join(table_id.split("_")[1:-1])  # Remove "table_" prefix and "_N" suffix
                
                # Track tokens with detailed breakdown
                self._track_tokens('generate_table_metadata', file_id, input_tokens, output_tokens, call_cost)
                
                # Concise single-line logging for tracking
                self.logger.info(f"🤖 {table_id}: {input_tokens}+{output_tokens}={total_tokens} tokens (${call_cost:.6f}) | Total: {self.total_input_tokens + self.total_output_tokens:,} tokens (${self.total_cost:.6f})")
                
                # Parse response
                title = None
                summary = None 
                classification = None
                
                for line in content.split('\n'):
                    if line.startswith('Title:'):
                        title = line.replace('Title:', '').strip()
                    elif line.startswith('Summary:'):
                        summary = line.replace('Summary:', '').strip()
                    elif line.startswith('Classification:'):
                        classification = line.replace('Classification:', '').strip()
                
                self.logger.info(f"✅ Generated metadata for {table_id}: '{title}' [{classification}]")
                return title, summary, classification
            else:
                self.logger.error(f"❌ LLM metadata generation failed: HTTP {response.status_code}")
                if response.text:
                    self.logger.debug(f"Response: {response.text}")
                return None, None, None
                
        except Exception as e:
            self.logger.error(f"❌ Error generating table metadata: {e}")
            return None, None, None
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get comprehensive token usage statistics with detailed breakdown."""
        return {
            'global_totals': {
                'api_calls': self.api_calls_count,
                'input_tokens': self.total_input_tokens,
                'output_tokens': self.total_output_tokens,
                'total_tokens': self.total_input_tokens + self.total_output_tokens,
                'total_cost': self.total_cost
            },
            'method_breakdown': self.method_tokens.copy(),
            'file_breakdown': self.file_tokens.copy()
        }
    
    def log_final_token_report(self):
        """Log a comprehensive final token usage report."""
        self.logger.info("=" * 80)
        self.logger.info("🎯 FINAL TOKEN USAGE REPORT")
        self.logger.info("=" * 80)
        
        # Global totals
        self.logger.info(f"📊 GLOBAL TOTALS:")
        self.logger.info(f"   └─ Total API Calls: {self.api_calls_count}")
        self.logger.info(f"   └─ Total Input Tokens: {self.total_input_tokens:,}")
        self.logger.info(f"   └─ Total Output Tokens: {self.total_output_tokens:,}")
        self.logger.info(f"   └─ Total Tokens: {self.total_input_tokens + self.total_output_tokens:,}")
        self.logger.info(f"   └─ Total Cost: ${self.total_cost:.6f}")
        
        # Method breakdown
        if self.method_tokens:
            self.logger.info(f"\n🔧 METHOD BREAKDOWN:")
            for method, stats in self.method_tokens.items():
                total = stats['input'] + stats['output']
                self.logger.info(f"   └─ {method}:")
                self.logger.info(f"      ├─ Calls: {stats['calls']}")
                self.logger.info(f"      ├─ Tokens: {stats['input']:,} input + {stats['output']:,} output = {total:,} total")
                self.logger.info(f"      └─ Cost: ${stats['cost']:.6f}")
        
        # File breakdown
        if self.file_tokens:
            self.logger.info(f"\n📁 FILE BREAKDOWN:")
            for file_id, stats in self.file_tokens.items():
                total = stats['input'] + stats['output']
                self.logger.info(f"   └─ {file_id}:")
                self.logger.info(f"      ├─ Calls: {stats['calls']}")
                self.logger.info(f"      ├─ Tokens: {stats['input']:,} input + {stats['output']:,} output = {total:,} total")
                self.logger.info(f"      └─ Cost: ${stats['cost']:.6f}")
        
        self.logger.info("=" * 80)
    


class TableChunker(BaseProcessor):
    """Advanced table chunker using JSON detection + MD extraction strategy."""
    
    def __init__(self, 
                 max_rows_per_chunk: int = 200,
                 preserve_headers: bool = True,
                 generate_llm_metadata: bool = True,
                 md_file_path: Optional[Path] = None):
        """Initialize table chunker with configuration."""
        self.max_rows_per_chunk = max_rows_per_chunk
        self.preserve_headers = preserve_headers
        self.generate_llm_metadata = generate_llm_metadata
        self.md_file_path = md_file_path
        
        self.json_detector = JSONTableDetector()
        self.md_extractor = MarkdownTableExtractor()
        self.llm_processor = LLMTableProcessor() if generate_llm_metadata else None
        
        self.logger = logging.getLogger(__name__ + '.TableChunker')
    
    async def process(self, json_content: str, doc_id: str, **kwargs) -> List[Chunk]:
        """Process using JSON detection + MD extraction strategy."""
        try:
            self.logger.info(f"🏢 Starting table processing for document: {doc_id}")
            
            # Parse JSON to get table boundaries
            json_data = json.loads(json_content) if isinstance(json_content, str) else json_content
            self.logger.debug(f"Parsed JSON document with {len(json_data.keys())} top-level keys")
            
            boundaries = self.json_detector.detect_table_boundaries(json_data)
            
            if not boundaries:
                self.logger.info("❌ No tables detected in JSON - skipping table processing")
                return []
            
            # Load MD content if available
            md_content = None
            if self.md_file_path and self.md_file_path.exists():
                try:
                    md_content = self.md_file_path.read_text(encoding='utf-8')
                    self.logger.info(f"📂 Loaded MD file: {self.md_file_path} ({len(md_content)} chars)")
                except Exception as e:
                    self.logger.error(f"❌ Failed to load MD file: {e}")
            
            if not md_content:
                self.logger.error("❌ No MD content available - cannot extract tables")
                return []
            
            # Extract and process each table
            chunks = []
            self.logger.info(f"🔄 Processing {len(boundaries)} table boundaries...")
            
            for i, boundary in enumerate(boundaries):
                self.logger.info(f"Processing table {i+1}/{len(boundaries)}: {boundary.table_id}")
                table_chunks = await self._process_table_boundary(boundary, md_content, doc_id)
                chunks.extend(table_chunks)
                self.logger.debug(f"Table {boundary.table_id} generated {len(table_chunks)} chunks")
            
            self.logger.info(f"🎯 Table processing complete: {len(chunks)} chunks from {len(boundaries)} tables")
            
            # Log token statistics if LLM was used
            if self.llm_processor:
                stats = self.llm_processor.get_token_stats()
                if stats['api_calls'] > 0:
                    self.logger.info(f"💰 OpenAI Usage: {stats['api_calls']} calls, {stats['total_tokens']} tokens, ${stats['total_cost']:.6f}")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error processing tables for {doc_id}: {e}", exc_info=True)
            return []
    
    async def _process_table_boundary(self, boundary: TableBoundary, md_content: str, doc_id: str) -> List[Chunk]:
        """Process a table boundary by extracting from MD and chunking."""
        # Extract table from MD using boundary info
        extraction_result = self.md_extractor.extract_table_from_md(md_content, boundary)
        
        if not extraction_result.md_content or extraction_result.confidence < 0.3:
            self.logger.warning(f"Failed to extract table {boundary.table_id} from MD (confidence: {extraction_result.confidence:.2f})")
            return []
        
        self.logger.info(f"Extracted table {boundary.table_id} using {extraction_result.extraction_method} (confidence: {extraction_result.confidence:.2f})")
        
        # Determine chunking strategy based on table size
        table_rows = len([line for line in extraction_result.md_content.split('\n') if '|' in line])
        
        if table_rows <= self.max_rows_per_chunk:
            # Single chunk
            chunk = await self._create_single_chunk(extraction_result, doc_id)
            return [chunk] if chunk else []
        else:
            # Multiple chunks with header preservation
            return await self._create_multi_chunk_table(extraction_result, doc_id)
    
    async def _create_single_chunk(self, extraction_result: TableExtractionResult, doc_id: str) -> Optional[Chunk]:
        """Create a single chunk for a complete table."""
        try:
            boundary = extraction_result.boundary_info
            table_id = boundary.table_id
            markdown_content = extraction_result.md_content
            
            # Generate metadata using LLM if enabled
            title = None
            summary = None
            classification = None
            
            if self.llm_processor:
                title, summary, classification = await self.llm_processor.generate_table_metadata(
                    markdown_content, table_id
                )
            
            # Detect units per column using streamlined static method
            header_cells = boundary.headers or []
            units_per_column = TableUnitDetector.detect_units_per_column(header_cells)
            
            # Create comprehensive table metadata
            metadata = ChunkMetadata(
                chunk_id=f"{doc_id}_{table_id}",
                doc_id=doc_id,
                chunk_type="table",
                word_count=len(markdown_content.split()),
                table_id=table_id,
                column_headers=boundary.headers or [],
                table_title=title,
                table_caption=summary
            )
            
            # Add table-specific metadata to section_path and metrics
            if classification:
                metadata.section_path = [f"Table: {classification}"]
                metadata.metrics.append(f"classification_{classification}")
            
            # Add row headers to metrics if available
            if hasattr(boundary, 'row_headers') and boundary.row_headers:
                metadata.metrics.extend([f"row_header_{i}_{header}" for i, header in enumerate(boundary.row_headers)])
                metadata.metrics.append(f"row_headers_count_{len(boundary.row_headers)}")
            
            # Add table dimensions
            metadata.metrics.extend([
                f"table_rows_{boundary.num_rows}",
                f"table_cols_{boundary.num_cols}",
                f"extraction_method_{extraction_result.extraction_method}",
                f"extraction_confidence_{extraction_result.confidence:.2f}"
            ])
            
            # Add units per column
            if units_per_column:
                for col, unit in units_per_column.items():
                    metadata.metrics.append(f"unit_{col}_{unit}")
            
            # Add page information if available
            if boundary.start_page is not None:
                metadata.page = boundary.start_page
                metadata.metrics.append(f"page_start_{boundary.start_page}")
                if boundary.end_page and boundary.end_page != boundary.start_page:
                    metadata.metrics.append(f"page_end_{boundary.end_page}")
                    metadata.metrics.append(f"multi_page_table")
            
            # Create chunk
            chunk = Chunk(
                metadata=metadata,
                content=markdown_content
            )
            
            self.logger.info(f"Created single chunk for table {table_id}")
            return chunk
            
        except Exception as e:
            self.logger.error(f"Error creating single chunk: {e}")
            return None
    
    async def _create_multi_chunk_table(self, extraction_result: TableExtractionResult, doc_id: str) -> List[Chunk]:
        """Split large MD table into multiple chunks while preserving headers."""
        try:
            boundary = extraction_result.boundary_info
            table_id = boundary.table_id
            markdown_content = extraction_result.md_content
            
            # Split markdown table into lines
            lines = markdown_content.split('\n')
            table_lines = [line for line in lines if '|' in line and line.strip()]
            
            if not table_lines:
                return []
            
            # Identify header lines (first 1-2 lines, before any separator)
            header_lines = []
            data_start_idx = 0
            
            for idx, line in enumerate(table_lines[:3]):  # Check first 3 lines
                if re.match(r'^\s*\|[\s\-\|]*\|\s*$', line):  # Separator line
                    data_start_idx = idx + 1
                    break
                elif idx < 2:  # Potential header lines
                    header_lines.append(line)
                    data_start_idx = idx + 1
            
            # Get data lines
            data_lines = table_lines[data_start_idx:]
            
            # Calculate chunk size accounting for headers
            chunk_size = max(1, self.max_rows_per_chunk - len(header_lines))
            chunks = []
            
            # Generate metadata for first chunk only (to save LLM costs)
            title = None
            summary = None
            classification = None
            
            if self.llm_processor:
                title, summary, classification = await self.llm_processor.generate_table_metadata(
                    markdown_content[:1000], table_id  # Use first part for metadata
                )
            
            # Create chunks
            total_chunks = (len(data_lines) + chunk_size - 1) // chunk_size  # Ceiling division
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(data_lines))
                chunk_data_lines = data_lines[start_idx:end_idx]
                
                # Combine headers + separator + chunk data
                chunk_lines = header_lines[:]
                if header_lines and chunk_data_lines:
                    # Add separator if we have headers
                    sep_cols = len(header_lines[0].split('|')) - 2 if header_lines else 3
                    separator = '|' + '|'.join(['---' for _ in range(sep_cols)]) + '|'
                    chunk_lines.append(separator)
                
                chunk_lines.extend(chunk_data_lines)
                chunk_content = '\n'.join(chunk_lines)
                
                # Create comprehensive chunk metadata for multi-part table
                chunk_id = f"{doc_id}_{table_id}_part_{chunk_idx + 1}"
                
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_type="table",
                    word_count=len(chunk_content.split()),
                    table_id=table_id,
                    column_headers=boundary.headers or [],
                    table_title=title,
                    table_caption=f"{summary} (Part {chunk_idx + 1} of {total_chunks})" if summary else f"Part {chunk_idx + 1} of {total_chunks}"
                )
                
                # Add comprehensive table-specific metadata
                if classification:
                    metadata.section_path = [f"Table: {classification} (Part {chunk_idx + 1})"]
                    metadata.metrics.append(f"classification_{classification}")
                
                # Add row headers to metrics if available
                if hasattr(boundary, 'row_headers') and boundary.row_headers:
                    metadata.metrics.extend([f"row_header_{i}_{header}" for i, header in enumerate(boundary.row_headers)])
                    metadata.metrics.append(f"row_headers_count_{len(boundary.row_headers)}")
                
                # Add table dimensions and chunk-specific info
                metadata.metrics.extend([
                    f"table_rows_{boundary.num_rows}",
                    f"table_cols_{boundary.num_cols}",
                    f"chunk_part_{chunk_idx + 1}_of_{total_chunks}",
                    f"multi_chunk_table",
                    f"chunk_rows_{len(chunk_data_lines)}"
                ])
                
                # Add units per column if available
                units_per_column = TableUnitDetector.detect_units_per_column(boundary.headers or [])
                if units_per_column:
                    for col, unit in units_per_column.items():
                        metadata.metrics.append(f"unit_{col}_{unit}")
                
                # Add page information if available
                if boundary.start_page is not None:
                    metadata.page = boundary.start_page
                    metadata.metrics.append(f"page_start_{boundary.start_page}")
                    if boundary.end_page and boundary.end_page != boundary.start_page:
                        metadata.metrics.append(f"page_end_{boundary.end_page}")
                        metadata.metrics.append(f"multi_page_table")
                
                # Create chunk
                chunk = Chunk(
                    metadata=metadata,
                    content=chunk_content
                )
                
                chunks.append(chunk)
            
            self.logger.info(f"Split table {table_id} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating multi-chunk table: {e}")
            return []


class TableProcessor(BaseProcessor):
    """Main table processing coordinator implementing JSON detection + MD extraction."""
    
    def __init__(self, md_file_path: Optional[Path] = None, **kwargs):
        """Initialize table processor with MD file path for extraction."""
        self.table_chunker = TableChunker(md_file_path=md_file_path, **kwargs)
        self.logger = logging.getLogger(__name__ + '.TableProcessor')
    
    async def process(self, content: str, **kwargs) -> List[Chunk]:
        """Process JSON content and extract table chunks using MD extraction."""
        doc_id = kwargs.get('doc_id', 'unknown_doc')
        
        self.logger.info(f"Starting table processing for document: {doc_id}")
        self.logger.info(f"Strategy: JSON detection + MD extraction")
        
        # Process tables using JSON boundaries + MD content
        table_chunks = await self.table_chunker.process(content, doc_id=doc_id)
        
        self.logger.info(f"Table processing complete: {len(table_chunks)} chunks created")
        return table_chunks
    
    def set_md_file_path(self, md_path: Path):
        """Update the MD file path for extraction."""
        self.table_chunker.md_file_path = md_path
        self.logger.info(f"Updated MD file path: {md_path}")


# Factory functions for easy instantiation
def create_table_processor(md_file_path: Optional[Path] = None, **kwargs) -> TableProcessor:
    """Create configured table processor with JSON detection + MD extraction strategy."""
    return TableProcessor(md_file_path=md_file_path, **kwargs)

def create_table_processor_with_paths(json_path: Path, md_path: Path, **kwargs) -> TableProcessor:
    """Create table processor with explicit JSON and MD file paths."""
    processor = TableProcessor(md_file_path=md_path, **kwargs)
    return processor