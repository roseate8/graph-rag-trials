"""
Output formatter for BEIR-compatible files.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from query_generator import Query

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Writes output in BEIR format:
    - queries.jsonl
    - qrels.tsv
    - corpus.jsonl
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize output formatter.
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evals/synthetic-eval/output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized OutputFormatter with output dir: {self.output_dir}")
    
    def write_queries(self, queries: List[Query], output_path: str = None) -> str:
        """
        Write queries to JSONL format.
        
        Format: {"_id": "q1", "text": "What is...", "metadata": {...}}
        
        Args:
            queries: List of Query objects
            output_path: Output file path (default: output_dir/queries.jsonl)
            
        Returns:
            Path to written file
        """
        if output_path is None:
            output_path = self.output_dir / "queries.jsonl"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Writing {len(queries)} queries to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for query in queries:
                query_entry = {
                    "_id": query.query_id,
                    "text": query.query_text,
                    "metadata": {
                        "answer": query.answer,
                        "gold_chunk_ids": query.gold_chunk_ids,
                        "query_type": query.query_type,
                        "question_style": query.question_style,
                        **query.metadata
                    }
                }
                f.write(json.dumps(query_entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully wrote queries to {output_path}")
        return str(output_path)
    
    def write_qrels(
        self,
        qrels: Dict[str, Dict[str, int]],
        output_path: str = None
    ) -> str:
        """
        Write qrels (query relevance labels) to TSV format.
        
        Format: query-id \t corpus-id \t score
        
        Args:
            qrels: Dictionary of query_id -> {chunk_id: relevance_score}
            output_path: Output file path (default: output_dir/qrels.tsv)
            
        Returns:
            Path to written file
        """
        if output_path is None:
            output_path = self.output_dir / "qrels.tsv"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Writing qrels for {len(qrels)} queries to {output_path}")
        
        total_entries = 0
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header
            f.write("query-id\tcorpus-id\tscore\n")
            
            # Write qrels
            for query_id, chunk_labels in sorted(qrels.items()):
                for chunk_id, score in sorted(chunk_labels.items()):
                    # Only write relevant chunks (score > 0) for efficiency
                    if score > 0:
                        f.write(f"{query_id}\t{chunk_id}\t{score}\n")
                        total_entries += 1
        
        logger.info(f"Successfully wrote {total_entries} qrel entries to {output_path}")
        return str(output_path)
    
    def write_corpus(
        self,
        chunks: List[Dict[str, Any]],
        output_path: str = None
    ) -> str:
        """
        Write corpus (chunks) to JSONL format.
        
        Format: {"_id": "chunk1", "text": "...", "metadata": {...}}
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Output file path (default: output_dir/corpus.jsonl)
            
        Returns:
            Path to written file
        """
        if chunks is None:
            logger.info("Skipping corpus.jsonl writing (chunks not provided)")
            return None

        if output_path is None:
            output_path = self.output_dir / "corpus.jsonl"
        else:
            output_path = Path(output_path)

        logger.info(f"Writing {len(chunks)} chunks to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                corpus_entry = {
                    "_id": chunk.get('chunk_id', ''),
                    "title": chunk.get('section_path', ''),
                    "text": chunk.get('content', ''),
                    "metadata": {
                        "doc_id": chunk.get('doc_id', ''),
                        "word_count": chunk.get('word_count', 0),
                        "chunk_type": chunk.get('chunk_type'),
                        "regions": chunk.get('regions'),
                        "product_version": chunk.get('product_version'),
                        "folder_path": chunk.get('folder_path'),
                    }
                }
                f.write(json.dumps(corpus_entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully wrote corpus to {output_path}")
        return str(output_path)
    
    def write_generation_stats(
        self,
        stats: Dict[str, Any],
        output_path: str = None
    ) -> str:
        """
        Write generation statistics to JSON format.
        
        Args:
            stats: Statistics dictionary
            output_path: Output file path (default: output_dir/generation_stats.json)
            
        Returns:
            Path to written file
        """
        if output_path is None:
            output_path = self.output_dir / "generation_stats.json"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Writing generation statistics to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully wrote statistics to {output_path}")
        return str(output_path)
    
    def write_generation_report(
        self,
        stats: Dict[str, Any],
        output_path: str = None
    ) -> str:
        """
        Write human-readable generation report.
        
        Args:
            stats: Statistics dictionary
            output_path: Output file path (default: output_dir/generation_report.txt)
            
        Returns:
            Path to written file
        """
        if output_path is None:
            output_path = self.output_dir / "generation_report.txt"
        else:
            output_path = Path(output_path)
        
        logger.info(f"Writing generation report to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SYNTHETIC EVALUATION DATASET GENERATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Sampling stats
            if 'sampling' in stats:
                f.write("SAMPLING STATISTICS\n")
                f.write("-" * 40 + "\n")
                sampling = stats['sampling']
                f.write(f"Total chunks in collection: {sampling.get('total_chunks', 'N/A')}\n")
                f.write(f"Number of clusters: {sampling.get('num_clusters', 'N/A')}\n")
                f.write(f"Target samples: {sampling.get('target_samples', 'N/A')}\n")
                f.write(f"Actual samples: {sampling.get('actual_samples', 'N/A')}\n\n")
            
            # Fact extraction stats
            if 'fact_extraction' in stats:
                f.write("FACT EXTRACTION STATISTICS\n")
                f.write("-" * 40 + "\n")
                facts = stats['fact_extraction']
                f.write(f"Total facts extracted: {facts.get('total_facts', 'N/A')}\n")
                f.write(f"Facts per chunk (avg): {facts.get('avg_facts_per_chunk', 0):.1f}\n")
                f.write(f"Fact types:\n")
                for fact_type, count in facts.get('fact_types', {}).items():
                    f.write(f"  - {fact_type}: {count}\n")
                f.write("\n")
            
            # Query generation stats
            if 'query_generation' in stats:
                f.write("QUERY GENERATION STATISTICS\n")
                f.write("-" * 40 + "\n")
                queries = stats['query_generation']
                f.write(f"Total queries: {queries.get('total_queries', 'N/A')}\n")
                f.write(f"Single-hop queries: {queries.get('single_hop', 'N/A')}\n")
                f.write(f"Multi-hop queries: {queries.get('multi_hop', 'N/A')}\n")
                f.write(f"Question styles:\n")
                for style, count in queries.get('query_styles', {}).items():
                    f.write(f"  - {style}: {count}\n")
                f.write("\n")
            
            # Silver labeling stats
            if 'silver_labeling' in stats:
                f.write("SILVER LABELING STATISTICS\n")
                f.write("-" * 40 + "\n")
                labels = stats['silver_labeling']
                f.write(f"Total labels assigned: {labels.get('total_labels', 'N/A')}\n")
                f.write(f"Label distribution:\n")
                for label, count in sorted(labels.get('label_distribution', {}).items()):
                    pct = labels.get('label_percentages', {}).get(label, 0)
                    f.write(f"  - Relevance {label}: {count} ({pct:.1f}%)\n")
                f.write(f"\nAverage relevant chunks per query: {labels.get('avg_relevant_per_query', 0):.1f}\n")
                f.write(f"Queries with no relevant chunks: {labels.get('queries_with_no_relevant', 0)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Generation completed successfully!\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Successfully wrote report to {output_path}")
        return str(output_path)
    
    def write_all(
        self,
        queries: List[Query],
        qrels: Dict[str, Dict[str, int]],
        chunks: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Write all output files at once.
        
        Args:
            queries: List of queries
            qrels: Qrels dictionary
            chunks: List of chunks
            stats: Statistics dictionary
            
        Returns:
            Dictionary mapping file type -> file path
        """
        logger.info("Writing all output files...")
        
        output_files = {
            'queries': self.write_queries(queries),
            'qrels': self.write_qrels(qrels),
            'corpus': self.write_corpus(chunks),
            'stats': self.write_generation_stats(stats),
            'report': self.write_generation_report(stats)
        }
        
        logger.info(f"Successfully wrote all output files to {self.output_dir}")
        
        return output_files

