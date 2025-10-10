"""
Run file generator for BEIR-like evaluation - connects test queries to naive-rag system.

This script takes queries from the generated test data and runs them through the 
naive-rag pipeline to produce ranked retrieval results compatible with ranx.
"""

import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from tqdm import tqdm

# Add paths to import naive-rag components
CURRENT_FILE = Path(__file__).absolute()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent.parent  # evals/custom/calculate-metrics -> project root
NAIVE_RAG_PATH = PROJECT_ROOT / "naive-rag"
VECTOR_INGEST_PATH = PROJECT_ROOT / "vector-ingest" / "src"

sys.path.insert(0, str(NAIVE_RAG_PATH))
sys.path.insert(0, str(VECTOR_INGEST_PATH))

# Import RAG system components
from core import RAGSystem, create_rag_system
from retrieval import RetrievedChunk
from chunking.processors.llm_utils import set_openai_api_key, has_openai_api_key

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RunFileGenerator:
    """Generates ranx-compatible run files from naive-rag system outputs."""
    
    def __init__(
        self,
        rag_system: Optional[RAGSystem] = None,
        retrieval_multiplier: float = 12.8,  # 12.8 * 10 = 128 (under ef limit)
        max_retrieval_size: int = 10
    ):
        """
        Initialize run generator with RAG system.
        
        Args:
            rag_system: Pre-initialized RAG system (if None, will create one)
            retrieval_multiplier: Multiplier for initial retrieval (before reranking)
            max_retrieval_size: Maximum number of documents to retrieve per query
        """
        self.retrieval_multiplier = retrieval_multiplier
        self.max_retrieval_size = max_retrieval_size
        
        # Initialize RAG system if not provided - always with reranking enabled
        if rag_system is None:
            logger.info("Initializing naive-rag system with reranking enabled...")
            self.rag_system = create_rag_system(
                llm_type="mock",  # We only need retrieval, not generation
                enable_reranking=True,  # Always use reranking
                retrieval_multiplier=retrieval_multiplier
            )
        else:
            self.rag_system = rag_system
        
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to the RAG system (Milvus)."""
        logger.info("Connecting to Milvus vector database...")
        
        if not self.rag_system.connect():
            logger.error("Failed to connect to Milvus")
            return False
        
        # Verify data availability
        stats = self.rag_system.get_system_stats()
        num_entities = stats.get('retriever_stats', {}).get('num_entities', 0)
        
        logger.info(f"Connected to collection with {num_entities} document chunks")
        
        if num_entities == 0:
            logger.warning("Entity count shows 0, verifying data availability...")
            try:
                # Test search to verify data exists
                test_chunks = self.rag_system.retriever.retrieve("test query", top_k=1)
                if test_chunks:
                    logger.info(f"Data verified via test search: {len(test_chunks)} chunks available")
                else:
                    logger.error("No documents found. Please ensure data is loaded in Milvus.")
                    return False
            except Exception as e:
                logger.error(f"Could not verify data availability: {e}")
                return False
        
        self.connected = True
        return True
    
    def disconnect(self):
        """Disconnect from RAG system."""
        if self.connected:
            self.rag_system.disconnect()
            self.connected = False
            logger.info("Disconnected from RAG system")
    
    def load_queries(self, queries_file: Path) -> List[Dict[str, Any]]:
        """Load queries from JSONL file."""
        logger.info(f"Loading queries from {queries_file}")
        
        queries = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    query_data = json.loads(line)
                    queries.append(query_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        
        logger.info(f"Loaded {len(queries)} queries")
        return queries
    
    def retrieve_for_query(self, query_text: str, query_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a single query using naive-rag system.
        
        Args:
            query_text: The query text to search for
            query_id: Query identifier for logging
        
        Returns:
            List of retrieved documents with scores
        """
        try:
            logger.info(f"Calling retriever for query {query_id} with top_k={self.max_retrieval_size}")
            # Use retriever directly to get only retrieval results (no LLM generation)
            chunks = self.rag_system.retriever.retrieve(
                query=query_text,
                top_k=self.max_retrieval_size,
                min_similarity=0.0  # Don't filter by similarity
            )
            logger.info(f"Retriever returned {len(chunks)} chunks for query {query_id}")
            
            # Convert chunks to ranx format
            results = []
            for chunk in chunks:
                # Always use rerank_score since reranking is always enabled
                if chunk.rerank_score is None:
                    logger.error(f"Missing rerank_score for chunk {chunk.chunk_id} - reranking should always be enabled!")
                    continue
                
                results.append({
                    "doc_id": chunk.doc_id,
                    "score": float(chunk.rerank_score),  # Always use rerank score
                    "similarity_score": float(chunk.similarity_score),
                    "rerank_score": float(chunk.rerank_score),
                    "chunk_id": chunk.chunk_id
                })
            
            logger.debug(f"Query {query_id}: Retrieved {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"Error retrieving documents for query {query_id}: {e}")
            return []
    
    def generate_run_file(
        self,
        queries_file: Path,
        output_file: Path,
        max_queries: Optional[int] = None,
        query_buckets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete run file from queries.
        
        Args:
            queries_file: Path to queries.jsonl file
            output_file: Path to output run.jsonl file
            max_queries: Maximum number of queries to process (for testing)
            query_buckets: Only process queries from these buckets (None = all)
        
        Returns:
            Generation statistics
        """
        if not self.connected:
            raise ConnectionError("RAG system not connected. Call connect() first.")
        
        # Load queries
        all_queries = self.load_queries(queries_file)
        
        # Filter by buckets if specified
        if query_buckets:
            queries = [q for q in all_queries if q.get("bucket") in query_buckets]
            logger.info(f"Filtered to {len(queries)} queries from buckets: {query_buckets}")
        else:
            queries = all_queries
        
        # Limit number of queries for testing
        if max_queries and max_queries < len(queries):
            queries = queries[:max_queries]
            logger.info(f"Limited to first {max_queries} queries for testing")
        
        # Generate run data
        start_time = time.time()
        run_data = {}
        query_stats = {
            "total_queries": len(queries),
            "processed_queries": 0,
            "failed_queries": 0,
            "total_documents_retrieved": 0,
            "queries_by_bucket": {},
            "average_docs_per_query": 0.0,
            "processing_time": 0.0
        }
        
        logger.info(f"Processing {len(queries)} queries through naive-rag...")
        
        # Process queries with progress bar
        with tqdm(total=len(queries), desc="Generating run") as pbar:
            for query_data in queries:
                query_id = query_data["qid"]
                query_text = query_data["text"]
                bucket = query_data.get("bucket", "unknown")
                
                # Track bucket statistics - use more efficient bucket lookup
                bucket_stats = query_stats["queries_by_bucket"]
                if bucket not in bucket_stats:
                    bucket_stats[bucket] = {"processed": 0, "failed": 0}
                
                try:
                    logger.info(f"Processing query {query_id}: {query_text[:50]}...")

                    # Update RAG system configuration for this query (same as RAG UI)
                    self.rag_system.retrieval_multiplier = self.retrieval_multiplier

                    # Retrieve documents for this query
                    logger.info(f"Starting retrieval for query {query_id}")
                    results = self.retrieve_for_query(query_text, query_id)
                    logger.info(f"Completed retrieval for query {query_id}, got {len(results)} results")
                    
                    if results:
                        # Store results for this query
                        for result in results:
                            if query_id not in run_data:
                                run_data[query_id] = {}
                            run_data[query_id][result["chunk_id"]] = result["score"]
                        
                        query_stats["processed_queries"] += 1
                        query_stats["total_documents_retrieved"] += len(results)
                        bucket_stats[bucket]["processed"] += 1
                    else:
                        logger.warning(f"No results for query {query_id}: {query_text[:50]}...")
                        query_stats["failed_queries"] += 1
                        bucket_stats[bucket]["failed"] += 1
                
                except Exception as e:
                    logger.error(f"Failed to process query {query_id}: {e}")
                    query_stats["failed_queries"] += 1
                    bucket_stats[bucket]["failed"] += 1
                
                pbar.update(1)
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        query_stats["processing_time"] = processing_time
        if query_stats["processed_queries"] > 0:
            query_stats["average_docs_per_query"] = query_stats["total_documents_retrieved"] / query_stats["processed_queries"]
        
        # Save run file in JSONL format (one doc per line)
        logger.info(f"Saving run file to {output_file}")
        logger.info(f"run_data contains {len(run_data)} queries")
        for qid, docs in list(run_data.items())[:2]:  # Log first 2 for debugging
            logger.info(f"Query {qid} has {len(docs)} documents")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for qid, doc_scores in run_data.items():
                for doc_id, score in doc_scores.items():
                    run_entry = {
                        "qid": qid,
                        "doc_id": doc_id,
                        "score": score
                    }
                    f.write(json.dumps(run_entry) + '\n')
        
        # Save generation statistics with buffered I/O
        stats_file = output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8', buffering=8192) as f:
            json.dump(query_stats, f, indent=2)
        
        logger.info(f"Run generation completed in {processing_time:.2f}s")
        logger.info(f"Processed: {query_stats['processed_queries']}/{query_stats['total_queries']} queries")
        logger.info(f"Total documents: {query_stats['total_documents_retrieved']}")
        logger.info(f"Average docs/query: {query_stats['average_docs_per_query']:.1f}")
        logger.info(f"Run file saved: {output_file}")
        logger.info(f"Statistics saved: {stats_file}")
        
        return query_stats
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def main():
    """Main CLI interface for run generation."""
    parser = argparse.ArgumentParser(
        description="Generate run files from test queries using naive-rag system"
    )
    
    # Input/output paths
    parser.add_argument(
        "--queries", 
        type=Path, 
        default=Path("../test-data/queries.jsonl"),
        help="Path to queries.jsonl file"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("../test-data/run.jsonl"),
        help="Path to output run.jsonl file (saved in test-data folder)"
    )
    
    # Processing options
    parser.add_argument(
        "--max-queries", 
        type=int, 
        help="Maximum number of queries to process (for testing)"
    )
    parser.add_argument(
        "--buckets", 
        type=str, 
        nargs='+',
        help="Only process queries from these buckets"
    )
    
    # Retrieval configuration
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=100,
        help="Maximum number of documents to retrieve per query"
    )
    parser.add_argument(
        "--retrieval-multiplier", 
        type=int, 
        default=10,
        help="Multiplier for initial retrieval before reranking"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.queries.exists():
        logger.error(f"Queries file not found: {args.queries}")
        return 1
    
    try:
        # Create run generator with specified configuration
        logger.info("=== NAIVE-RAG RUN FILE GENERATOR ===")
        
        generator = RunFileGenerator(
            retrieval_multiplier=args.retrieval_multiplier,
            max_retrieval_size=args.top_k
        )
        
        # Generate run file using context manager
        with generator:
            stats = generator.generate_run_file(
                queries_file=args.queries,
                output_file=args.output,
                max_queries=args.max_queries,
                query_buckets=args.buckets
            )
        
        logger.info("✅ Run file generation completed successfully!")
        return 0
    
    except KeyboardInterrupt:
        logger.info("🛑 Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error generating run file: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
