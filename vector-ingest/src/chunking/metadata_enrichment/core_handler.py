import logging
from typing import List, Dict, Any, Optional

from .models import EnrichmentContext
from .structural_matcher import StructuralMatcher

logger = logging.getLogger(__name__)


def enrich_chunks_with_structure(chunks: List[Any], context_data: Dict[str, Any]) -> List[Any]:
    """Main entry point for Phase 1 structural metadata enrichment."""
    if not chunks:
        return chunks
    
    try:
        context = EnrichmentContext(
            doc_id=context_data.get('doc_id', 'unknown_doc'),
            document_structure=context_data.get('document_structure'),
            json_elements=_parse_json_content(context_data.get('json_content')),
            md_content=context_data.get('md_content'),
            total_chunks=len(chunks)
        )
        
        # Debug logging
        logger.info(f"🔍 Debug: doc_structure={context.document_structure is not None}")
        logger.info(f"🔍 Debug: json_elements={len(context.json_elements) if context.json_elements else 0}")
        logger.info(f"🔍 Debug: md_content={len(context.md_content) if context.md_content else 0} chars")
        
        if not context.has_structure_data():
            logger.info("🔍 Debug: No structure data available")
            return chunks
        
        matcher = StructuralMatcher()
        matching_results = matcher.match_chunks_to_structure(chunks, context)
        
        enrichment_count = 0
        for chunk, result in zip(chunks, matching_results):
            if result.has_match:
                structural_metadata = matcher.create_structural_metadata(result)
                if structural_metadata:
                    chunk.metadata.structural_metadata = structural_metadata
                    enrichment_count += 1
                    
                    # Structural metadata is now stored separately - don't pollute metrics field
        
        logger.info(f"Enriched {enrichment_count}/{len(chunks)} chunks with structural metadata")
        return chunks
        
    except Exception as e:
        logger.error(f"Structural enrichment error: {e}")
        return chunks


def _parse_json_content(json_content: Any) -> Optional[List[Dict[str, Any]]]:
    """Fast JSON content parsing."""
    if not json_content:
        return None
    
    try:
        import json
        data = json.loads(json_content) if isinstance(json_content, str) else json_content
        
        elements = []
        # Fast extraction from common sections
        for section in ['texts', 'tables', 'figures']:
            if section in data and isinstance(data[section], list):
                elements.extend(data[section])
        
        # Handle body references (simplified)
        if 'body' in data and 'children' in data['body']:
            for child_ref in data['body']['children']:
                if isinstance(child_ref, dict) and '$ref' in child_ref:
                    ref_parts = child_ref['$ref'].strip('#/').split('/')
                    if len(ref_parts) >= 2:
                        section, idx_str = ref_parts[0], ref_parts[1]
                        try:
                            idx = int(idx_str)
                            if section in data and isinstance(data[section], list) and idx < len(data[section]):
                                elements.append(data[section][idx])
                        except (ValueError, IndexError):
                            pass
        
        return elements if elements else None
        
    except Exception:
        return None


def _add_structural_metrics(metrics: List[str], structural_metadata) -> None:
    """Add key structural info to metrics."""
    if structural_metadata.element_type:
        metrics.append(f"element_type_{structural_metadata.element_type}")
    if structural_metadata.is_heading:
        metrics.append("is_heading")
    if structural_metadata.page_number:
        metrics.append(f"page_{structural_metadata.page_number}")
    if structural_metadata.bbox_coords:
        metrics.append("has_bbox_coords")