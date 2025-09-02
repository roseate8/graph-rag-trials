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
    # Contextual metadata extracted from surrounding text elements
    caption: Optional[str] = None  # Table title/caption from adjacent text
    footnotes: List[str] = None  # Footnote text content


@dataclass
class TableExtractionResult:
    """Result of extracting table from MD using JSON boundaries."""
    table_id: str
    md_content: str  # Extracted markdown table
    boundary_info: TableBoundary
    extraction_method: str  # "json_guided", "md_fallback", "content_match"
    confidence: float = 1.0  # How confident we are in the extraction




class JSONTableDetector:
    """Streamlined table detector with minimal overhead."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.JSONTableDetector')
    
    def detect_table_boundaries(self, json_data: Dict[str, Any]) -> List[TableBoundary]:
        """Extract table boundaries with direct processing - no caching overhead."""
        tables = json_data.get('tables', [])
        if not tables:
            return []
        
        self.logger.info(f"📊 Processing {len(tables)} tables...")
        
        # Simple caption lookup - build only when needed
        text_lookup = {}
        if 'texts' in json_data:
            text_lookup = {
                elem.get('self_ref'): elem.get('text', '').strip()
                for elem in json_data['texts'] 
                if elem.get('self_ref') and elem.get('text', '').strip()
            }
        
        # Process tables efficiently
        boundaries = []
        for i, table_data in enumerate(tables):
            boundary = self._extract_boundary_simple(table_data, i, json_data, text_lookup)
            if boundary:
                boundaries.append(boundary)
        
        self.logger.info(f"🎯 Detected {len(boundaries)} table boundaries")
        return boundaries
    
    def _extract_boundary_simple(self, table_data: Dict, table_index: int, json_data: Dict, text_lookup: Dict) -> Optional[TableBoundary]:
        """Streamlined boundary extraction with minimal processing."""
        grid = table_data.get('data', {}).get('grid', [])
        if not grid:
            return None
        
        # Basic info
        num_rows = len(grid)
        num_cols = len(grid[0]) if grid else 0
        table_id = table_data.get('self_ref', f"#/tables/{table_index}").replace('#/tables/', 'table_')
        
        # Extract headers efficiently - single pass
        headers = []
        if grid and grid[0]:
            for cell_data in grid[0]:
                if cell_data and cell_data.get('text'):
                    headers.append(cell_data['text'].strip())
        
        # Extract page info
        prov = table_data.get('prov', [])
        start_page = None
        if prov and isinstance(prov[0], dict):
            start_page = prov[0].get('page_no') or prov[0].get('page')
        
        # Simple caption extraction - find preceding text with table keywords
        caption = self._find_table_caption_simple(table_data, json_data, text_lookup)
        
        return TableBoundary(
            table_id=table_id,
            start_page=start_page,
            end_page=start_page,  # Simplified
            bbox=table_data.get('bbox'),
            num_rows=num_rows,
            num_cols=num_cols,
            content_sample=None,  # Not needed for direct conversion
            headers=headers,
            row_headers=None,  # Simplified
            caption=caption,
            footnotes=[]  # Simplified
        )
    
    def _find_table_caption_simple(self, table_data: Dict, json_data: Dict, text_lookup: Dict) -> Optional[str]:
        """Simple caption extraction without complex caching."""
        table_ref = table_data.get('self_ref')
        if not table_ref:
            return None
        
        # Find table in body children
        children = json_data.get('body', {}).get('children', [])
        table_idx = None
        for i, child in enumerate(children):
            if child.get('$ref') == table_ref:
                table_idx = i
                break
        
        if table_idx is None:
            return None
        
        # Check 2 preceding elements for captions - be more specific
        for i in range(max(0, table_idx - 2), table_idx):
            if i >= len(children):
                continue
            child_ref = children[i].get('$ref', '')
            text = text_lookup.get(child_ref, '')
            # Only match actual table captions, not "Table of Contents" headers
            if text and 'following table' in text.lower() and 'table of contents' not in text.lower():
                return text
        
        return None


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
1. A specific table name/heading (2-6 words, like "Revenue by Segment" or "Cash Flow Statement")
2. A descriptive summary explaining what the table shows (35-50 words)
3. Classification (choose: financial_statement, summary_kpi, fact_table, dimension_table, timeseries_table, pivot_matrix, list_table, form_like, other)

Table:
{truncated_table}

Response format:
Title: [specific table name]
Summary: [what this table shows and contains]
Classification: [category]"""
            
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
    """Direct JSON-to-markdown table chunker with contextual metadata extraction."""
    
    def __init__(self, 
                 max_rows_per_chunk: int = 200,
                 preserve_headers: bool = True,
                 generate_llm_metadata: bool = True,
                 **kwargs):  # Accept md_file_path for backward compatibility but ignore it
        """Initialize table chunker with configuration."""
        self.max_rows_per_chunk = max_rows_per_chunk
        self.preserve_headers = preserve_headers
        self.generate_llm_metadata = generate_llm_metadata
        
        self.json_detector = JSONTableDetector()
        self.llm_processor = LLMTableProcessor() if generate_llm_metadata else None
        
        self.logger = logging.getLogger(__name__ + '.TableChunker')
    
    def _build_chunk_metadata(self, boundary: TableBoundary, doc_id: str, table_id: str, 
                            word_count: int, final_title: Optional[str], 
                            classification: Optional[str], part_info: Optional[str] = None, 
                            folder_path: Optional[List[str]] = None) -> ChunkMetadata:
        """Optimized shared metadata builder - caption now in content, not metadata."""
        # Build chunk ID efficiently
        chunk_id = f"{doc_id}_{table_id}{part_info}" if part_info else f"{doc_id}_{table_id}"
        
        # Create base metadata (no table_caption - it's now in content)
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk_type="table",
            word_count=word_count,
            table_id=table_id,
            column_headers=boundary.headers or [],
            table_title=final_title,
            # table_caption removed - now in chunk content
            
            # Add new required fields
            table_shape={"rows": boundary.num_rows, "cols": boundary.num_cols},
            product_version="v1",
            folder_path=folder_path or []
        )
        
        # Set classification-based section path
        if classification:
            section_base = f"Table: {classification}"
            metadata.section_path = [f"{section_base}{part_info}" if part_info else section_base]
        
        # Set page information  
        if boundary.start_page is not None:
            metadata.page = boundary.start_page
        
        # Simple structural metadata for tables
        from ..models import StructuralMetadata
        metadata.structural_metadata = StructuralMetadata(
            element_type="table",
            page_number=boundary.start_page,
            bbox_coords=boundary.bbox
        )
        
        return metadata
    
    def _extract_table_heading_simple(self, caption: str) -> Optional[str]:
        """Optimized heading extraction with minimal processing."""
        if not caption:
            return None
        
        caption_lower = caption.lower()
        
        # Fast string-based extraction instead of regex
        if 'following table' in caption_lower:
            # Find the verb and extract what comes after it
            words = caption.split()
            for i, word in enumerate(words):
                if word.lower() in ('summarizes', 'shows', 'presents', 'provides', 'represents', 'sets'):
                    # Take next 2-4 words after the verb
                    start_idx = i + 1
                    if word.lower() == 'sets' and i + 1 < len(words) and words[i + 1].lower() == 'forth':
                        start_idx = i + 2  # Skip "sets forth"
                    heading_words = words[start_idx:start_idx + 4]
                    if heading_words:
                        return ' '.join(w.capitalize() for w in heading_words[:3])
        
        return None
    
    def _create_fallback_summary(self, original_caption: str) -> str:
        """Minimal summary cleanup - single pass."""
        if not original_caption:
            return ""
        
        # Quick cleanup and length limit in one operation
        summary = original_caption.strip()
        if summary.lower().startswith('the following table'):
            summary = summary[20:]  # Remove "the following table" (20 chars)
        
        return summary[:100] if len(summary) <= 100 else summary[:100] + '...'
    
    def _build_table_chunk_content(self, title: Optional[str], caption: Optional[str], summary: Optional[str], table_content: str) -> str:
        """Optimized content building with minimal allocations."""
        parts = []
        
        # Build content parts efficiently - avoid multiple append calls
        if title:
            parts.extend([title, ""])
        if caption:
            parts.extend([caption, ""])
        if summary and summary != caption:  # Avoid duplication
            parts.extend([summary, ""])
        
        parts.append(table_content)
        
        # Single join operation
        return '\n'.join(parts)
    
    def _convert_json_table_to_markdown(self, boundary: TableBoundary, json_data: Dict[str, Any]) -> str:
        """Convert JSON table data directly to markdown - simple and reliable."""
        # Find the table in JSON data
        tables = json_data.get('tables', [])
        
        # Find our table by matching the table_id
        table_index = int(boundary.table_id.replace('table_', '')) if 'table_' in boundary.table_id else 0
        
        if table_index >= len(tables):
            return ""
        
        table_data = tables[table_index]
        data = table_data.get('data', {})
        grid = data.get('grid', [])
        
        if not grid:
            return ""
        
        # Convert grid to markdown table
        markdown_lines = []
        
        for row_idx, row in enumerate(grid):
            row_cells = []
            for cell_data in row:
                if cell_data and 'text' in cell_data:
                    cell_text = cell_data['text'].strip()
                    # Escape pipes in cell content
                    cell_text = cell_text.replace('|', '\\|')
                    row_cells.append(cell_text)
                else:
                    row_cells.append('')
            
            # Create markdown table row
            markdown_line = '| ' + ' | '.join(row_cells) + ' |'
            markdown_lines.append(markdown_line)
            
            # Add separator after header row (first row)
            if row_idx == 0:
                separator = '|' + '|'.join(['---' for _ in row_cells]) + '|'
                markdown_lines.append(separator)
        
        return '\n'.join(markdown_lines)
    
    async def process(self, json_content: str, doc_id: str, **kwargs) -> List[Chunk]:
        """Process using direct JSON-to-markdown conversion strategy."""
        try:
            self.logger.info(f"🏢 Starting table processing for document: {doc_id}")
            
            # Parse JSON to get table boundaries
            json_data = json.loads(json_content) if isinstance(json_content, str) else json_content
            self.logger.debug(f"Parsed JSON document with {len(json_data.keys())} top-level keys")
            
            boundaries = self.json_detector.detect_table_boundaries(json_data)
            
            if not boundaries:
                self.logger.info("❌ No tables detected in JSON - skipping table processing")
                return []
            
            # Process each table directly from JSON data
            chunks = []
            self.logger.info(f"🔄 Processing {len(boundaries)} table boundaries with direct JSON conversion...")
            
            for i, boundary in enumerate(boundaries):
                self.logger.info(f"Processing table {i+1}/{len(boundaries)}: {boundary.table_id}")
                table_chunks = await self._process_table_boundary(boundary, json_data, doc_id)
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
    
    async def _process_table_boundary(self, boundary: TableBoundary, json_data: Dict[str, Any], doc_id: str) -> List[Chunk]:
        """Process a table boundary by converting JSON data directly to markdown."""
        # Convert JSON table data directly to markdown - no complex extraction needed!
        markdown_content = self._convert_json_table_to_markdown(boundary, json_data)
        
        if not markdown_content:
            self.logger.warning(f"Failed to convert JSON table {boundary.table_id} to markdown")
            return []
        
        self.logger.info(f"Converted JSON table {boundary.table_id} to markdown ({len(markdown_content)} chars)")
        
        # Create a simple extraction result
        extraction_result = TableExtractionResult(
            table_id=boundary.table_id,
            md_content=markdown_content,
            boundary_info=boundary,
            extraction_method="json_direct",
            confidence=1.0
        )
        
        # Determine chunking strategy based on table size
        table_rows = len([line for line in markdown_content.split('\n') if '|' in line])
        
        if table_rows <= self.max_rows_per_chunk:
            # Single chunk
            chunk = await self._create_single_chunk(extraction_result, doc_id)
            return [chunk] if chunk else []
        else:
            # Multiple chunks with header preservation
            return await self._create_multi_chunk_table(extraction_result, doc_id)
    
    async def _create_single_chunk(self, extraction_result: TableExtractionResult, doc_id: str) -> Optional[Chunk]:
        """Optimized single chunk creation using shared metadata builder."""
        try:
            boundary = extraction_result.boundary_info
            table_id = boundary.table_id
            markdown_content = extraction_result.md_content
            
            # Extract heading from original JSON caption (if available)
            extracted_heading = self._extract_table_heading_simple(boundary.caption) if boundary.caption else None
            
            # LLM metadata generation (if enabled)
            llm_title, llm_summary, classification = None, None, None
            if self.llm_processor:
                llm_title, llm_summary, classification = await self.llm_processor.generate_table_metadata(
                    markdown_content, table_id
                )
            
            # Proper mapping:
            # table_title (heading) = extracted heading OR LLM title
            # table_caption (summary) = LLM summary OR fallback description
            final_title = extracted_heading or llm_title  # Short specific heading
            
            # For summary: prefer LLM, fallback to a clean version of original caption
            if llm_summary:
                final_caption = llm_summary  # High-level LLM summary
            elif boundary.caption:
                # Create a clean summary from original caption (without redundant phrases)
                final_caption = self._create_fallback_summary(boundary.caption)
            else:
                final_caption = None
            
            # Build enhanced chunk content with title + caption + summary + table
            enhanced_content = self._build_table_chunk_content(
                final_title, boundary.caption, final_caption, markdown_content
            )
            
            # Use shared metadata builder (no table_caption in metadata!)
            metadata = self._build_chunk_metadata(
                boundary, doc_id, table_id, len(enhanced_content.split()),
                final_title, classification  # No caption parameter
            )
            
            # Create chunk with enhanced content
            chunk = Chunk(metadata=metadata, content=enhanced_content)
            self.logger.info(f"Created single chunk for table {table_id}")
            return chunk
            
        except Exception as e:
            self.logger.error(f"Error creating single chunk: {e}")
            return None
    
    async def _create_multi_chunk_table(self, extraction_result: TableExtractionResult, doc_id: str) -> List[Chunk]:
        """Optimized multi-chunk table creation using shared metadata builder."""
        try:
            boundary = extraction_result.boundary_info
            table_id = boundary.table_id
            markdown_content = extraction_result.md_content
            
            # Optimized table line extraction and header detection
            table_lines = [line for line in markdown_content.split('\n') if '|' in line and line.strip()]
            if not table_lines:
                return []
            
            # Find header lines and data start index efficiently
            header_lines, data_start_idx = [], 0
            separator_pattern = re.compile(r'^\s*\|[\s\-\|]*\|\s*$')
            
            for idx, line in enumerate(table_lines[:3]):
                if separator_pattern.match(line):
                    data_start_idx = idx + 1
                    break
                elif idx < 2:
                    header_lines.append(line)
                    data_start_idx = idx + 1
            
            data_lines = table_lines[data_start_idx:]
            chunk_size = max(1, self.max_rows_per_chunk - len(header_lines))
            total_chunks = (len(data_lines) + chunk_size - 1) // chunk_size
            
            # Extract heading from original JSON caption
            extracted_heading = self._extract_table_heading_simple(boundary.caption) if boundary.caption else None
            
            # Generate LLM metadata once (not per chunk)
            llm_title, llm_summary, classification = None, None, None
            if self.llm_processor:
                llm_title, llm_summary, classification = await self.llm_processor.generate_table_metadata(
                    markdown_content[:1000], table_id
                )
            
            # Proper mapping for multi-chunk tables
            final_title = extracted_heading or llm_title  # Short specific heading
            
            # For summary: prefer LLM, fallback to clean original caption
            if llm_summary:
                final_caption = llm_summary  # High-level LLM summary
            elif boundary.caption:
                final_caption = self._create_fallback_summary(boundary.caption)
            else:
                final_caption = None
            
            # Pre-calculate separator for efficiency
            separator = None
            if header_lines:
                sep_cols = len(header_lines[0].split('|')) - 2
                separator = '|' + '|'.join(['---'] * sep_cols) + '|'
            
            # Create chunks efficiently
            chunks = []
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(data_lines))
                chunk_data_lines = data_lines[start_idx:end_idx]
                
                # Build chunk content
                chunk_lines = header_lines[:]
                if separator and chunk_data_lines:
                    chunk_lines.append(separator)
                chunk_lines.extend(chunk_data_lines)
                chunk_content = '\n'.join(chunk_lines)
                
                # Create part-specific caption for content
                part_suffix = f" (Part {chunk_idx + 1} of {total_chunks})"
                part_caption = f"{final_caption}{part_suffix}" if final_caption else f"Part {chunk_idx + 1} of {total_chunks}"
                
                # Build enhanced chunk content with title + caption + summary + table
                enhanced_content = self._build_table_chunk_content(
                    final_title, boundary.caption, part_caption, chunk_content
                )
                
                # Use shared metadata builder (no table_caption in metadata!)
                metadata = self._build_chunk_metadata(
                    boundary, doc_id, table_id, len(enhanced_content.split()),
                    final_title, classification, f"_part_{chunk_idx + 1}"  # No caption parameter
                )
                
                chunks.append(Chunk(metadata=metadata, content=enhanced_content))
            
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
        """Process JSON content and convert tables directly to markdown chunks."""
        doc_id = kwargs.get('doc_id', 'unknown_doc')
        
        self.logger.info(f"Starting table processing for document: {doc_id}")
        self.logger.info(f"Strategy: Direct JSON-to-markdown conversion")
        
        # Process tables using direct JSON-to-markdown conversion
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