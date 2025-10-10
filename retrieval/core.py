"""
Core RAG system orchestration and main interface - combines rag_system.py + main.py functionality.
"""

import sys
import logging
import warnings
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Lazy imports for better startup performance
_psutil = None
def get_psutil():
    global _psutil
    if _psutil is None:
        import psutil
        _psutil = psutil
    return _psutil

try:
    from .retrieval import MilvusRetriever, RetrievedChunk
    from .formatting import ContextFormatter, RAGPrompt
    from .llm import SecureOpenAIClient, MockLLMClient, RAGResponse, create_llm_client
except ImportError:
    from retrieval import MilvusRetriever, RetrievedChunk
    from formatting import ContextFormatter, RAGPrompt
    from llm import SecureOpenAIClient, MockLLMClient, RAGResponse, create_llm_client


@dataclass
class RAGResult:
    """Complete RAG result with full pipeline metadata."""
    query: str
    response: str
    retrieved_chunks: List[RetrievedChunk]
    context_token_count: int
    response_tokens: Optional[int] = None
    model_used: Optional[str] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None


class RAGSystem:
    """Optimized RAG system that orchestrates retrieval and generation."""
    
    def __init__(
        self,
        retriever: Optional[MilvusRetriever] = None,
        formatter: Optional[ContextFormatter] = None,
        llm_client: Optional[Any] = None,
        # Retriever parameters
        embedding_model: str = "BAAI/bge-m3",
        collection_name: str = "elastic_embeddings_m3",
        # Re-ranking parameters
        enable_reranking: bool = False,
        reranker_config: Optional[Dict[str, Any]] = None,
        retrieval_multiplier: int = 10,
        # Query decomposition parameters
        enable_query_decomposition: bool = False,
        max_sub_queries: int = 5,
        fusion_k_constant: int = 60,
        # Context parameters
        max_context_tokens: int = 4000,
        include_scores: bool = False,
        # LLM parameters
        llm_type: str = "mock",
        llm_model: str = "gpt-4o-mini",
        # History parameters
        enable_history: bool = True,
        history_file: str = "conversation_history.json"
    ):
        """Initialize complete RAG system."""
        # Store re-ranking configuration
        self.enable_reranking = enable_reranking
        self.retrieval_multiplier = retrieval_multiplier
        
        # Store query decomposition configuration
        self.enable_query_decomposition = enable_query_decomposition
        self.max_sub_queries = max_sub_queries
        self.fusion_k_constant = fusion_k_constant
        
        # Initialize query decomposer and fusion reranker if enabled
        self.query_decomposer = None
        self.fusion_reranker = None
        if enable_query_decomposition:
            try:
                # Try relative imports first, then absolute
                try:
                    from .decomposer import QueryDecomposer
                    from .re_rankers.fusion_reranker import FusionReranker
                except ImportError:
                    from decomposer import QueryDecomposer
                    from re_rankers.fusion_reranker import FusionReranker

                self.query_decomposer = QueryDecomposer(max_sub_queries=max_sub_queries)
                self.fusion_reranker = FusionReranker(k_constant=fusion_k_constant)
                logger.info(f"Query decomposition enabled (max_sub_queries={max_sub_queries}, fusion_k={fusion_k_constant})")
            except ImportError as e:
                logger.error(f"Failed to import query decomposition modules: {e}")
                self.enable_query_decomposition = False
        
        # Initialize components
        self.retriever = retriever or MilvusRetriever(
            embedding_model=embedding_model,
            collection_name=collection_name,
            enable_reranking=enable_reranking,
            reranker_config=reranker_config
        )
        
        self.formatter = formatter or ContextFormatter(
            max_context_tokens=max_context_tokens,
            include_scores=include_scores
        )
        
        self.llm_client = llm_client or create_llm_client(
            client_type=llm_type,
            model=llm_model
        )
        
        # History configuration
        self.enable_history = enable_history
        # Save conversation history to rag-ui folder
        rag_ui_path = Path(__file__).parent.parent / "rag-ui"
        self.history_file = rag_ui_path / history_file
        
        self.connected = False
        logger.info("Initialized RAG system")
    
    def connect(self) -> bool:
        """Connect to required services (Milvus)."""
        if not self.retriever.connect():
            logger.error("Failed to connect retriever")
            return False
        
        self.connected = True
        logger.info("RAG system connected")
        return True
    
    def disconnect(self):
        """Disconnect from services."""
        if self.connected:
            self.retriever.disconnect()
            self.connected = False
            logger.info("RAG system disconnected")
    
    def _get_resource_info(self) -> Dict[str, Any]:
        """Get current resource usage information - optimized with lazy imports."""
        try:
            # Optimized GPU info with lazy import
            gpu_info = {"gpu_available": False, "gpuutil_not_installed": True}
            
            # Try to get GPU info if available (lazy import)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    gpu_info = {
                        "gpu_available": True,
                        "gpuutil_not_installed": False,
                        "gpu_count": len(gpus),
                        "gpu_memory_used": gpu.memoryUsed,
                        "gpu_memory_total": gpu.memoryTotal
                    }
            except ImportError:
                pass  # Keep default GPU info
            
            # Get system memory and CPU with lazy import
            psutil = get_psutil()
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.05)  # Faster interval for better performance
            
            # Pre-calculate values to avoid repeated division
            mb_divisor = 1024 * 1024
            return {
                "memory_used_mb": round(memory.used / mb_divisor, 1),
                "memory_total_mb": round(memory.total / mb_divisor, 1),
                "memory_percent": round(memory.percent, 1),
                "cpu_percent": round(cpu_percent, 1),
                "gpu": gpu_info
            }
        except Exception as e:
            logger.error(f"Error getting resource info: {e}")
            return {
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "memory_percent": 0.0,
                "cpu_percent": 0.0,
                "gpu": {"gpu_available": False, "gpuutil_not_installed": True}
            }

    def _load_conversation_history(self) -> List[Dict]:
        """Load conversation history from JSON file."""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            return []
    
    def _save_conversation_entry(self, query: str, result: 'RAGResult', initial_resources: Dict, final_resources: Dict, timing_data: Dict):
        """Save a conversation entry to history in the exact format specified - optimized."""
        if not self.enable_history:
            return
        
        try:
            # Ensure rag-ui directory exists
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            
            history = self._load_conversation_history()
            
            # Optimized peak resource calculation
            peak_memory_mb = max(
                initial_resources.get("memory_used_mb", 0), 
                final_resources.get("memory_used_mb", 0)
            )
            peak_cpu_percent = max(
                initial_resources.get("cpu_percent", 0), 
                final_resources.get("cpu_percent", 0)
            )
            
            # Optimized chunk data formatting using list comprehension
            retrieved_chunks_data = [
                {
                    "content": chunk.content,
                    "score": chunk.similarity_score,
                    "metadata": {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "word_count": chunk.word_count,
                        "section_path": chunk.section_path,
                        "chunk_type": getattr(chunk, 'chunk_type', 'unknown')
                    }
                }
                for chunk in result.retrieved_chunks
            ]
            
            # Pre-calculate timestamp to avoid multiple calls
            now = datetime.now()
            timestamp_main = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z"
            timestamp_llm = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
            
            # Pre-calculate timing values to avoid repeated dict lookups
            retrieval_time = timing_data.get("retrieval_time", 0)
            rerank_time = retrieval_time if self.enable_reranking else 0.0
            context_time = timing_data.get("context_time", 0)
            llm_time = timing_data.get("generation_time", 0)
            total_time = timing_data.get("total_time", 0)
            response_tokens = result.response_tokens or 0
            
            # Build timing dict once and reuse
            timing_dict = {
                "vector_search_time": retrieval_time,
                "rerank_time": rerank_time,
                "context_time": context_time,
                "llm_time": llm_time,
                "total_time": total_time
            }
            
            # Build resources dict once and reuse
            resources_dict = {
                "initial": initial_resources,
                "final": final_resources,
                "peak_memory_mb": peak_memory_mb,
                "peak_cpu_percent": peak_cpu_percent
            }
            
            # Create the entry in the exact format specified
            entry = {
                "timestamp": timestamp_main,
                "query": query,
                "result": {
                    "retrieved_chunks": retrieved_chunks_data,
                    "context_length": result.context_token_count,
                    "llm_response": {
                        "query": query,
                        "method": "layout_aware_chunking",
                        "context": "",  # Context would be the formatted chunks text
                        "answer": result.response,
                        "timestamp": timestamp_llm,
                        "tokens_used": response_tokens
                    },
                    "timing": timing_dict,
                    "resources": resources_dict
                },
                "tokens_used": response_tokens,
                "timing": timing_dict,
                "resources": resources_dict
            }
            
            history.append(entry)
            
            # Optimized history size management
            if len(history) > 100:
                history = history[-100:]
            
            # Write with optimized JSON settings
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False, separators=(',', ': '))
            
            logger.info(f"Saved conversation entry to {self.history_file}")
        
        except Exception as e:
            logger.error(f"Error saving conversation history: {e}")
    
    def query(
        self,
        user_query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        system_prompt: Optional[str] = None
    ) -> RAGResult:
        """Execute complete RAG pipeline for a user query."""
        # Capture initial resources
        initial_resources = self._get_resource_info()
        start_total_time = time.time()
        
        try:
            self._validate_connection(user_query)
            # Choose retrieval strategy based on configuration
            if self.enable_query_decomposition and self.query_decomposer:
                logger.info("=" * 80)
                logger.info("QUERY DECOMPOSITION RETRIEVAL STRATEGY")
                logger.info("=" * 80)
                chunks, retrieval_time = self._retrieve_with_decomposition(
                    user_query, top_k, min_similarity
                )
            else:
                logger.info("Using standard single-query retrieval")
                chunks, retrieval_time = self._retrieve_chunks(
                    user_query, top_k, min_similarity
                )
            
            # Time context formatting
            context_start = time.time()
            rag_prompt = self._format_context(user_query, chunks, system_prompt)
            context_time = time.time() - context_start
            
            llm_response, generation_time = self._generate_response(rag_prompt)
            result = self._build_result(user_query, chunks, llm_response, retrieval_time, generation_time)
            
            # Capture final resources and timing
            total_time = time.time() - start_total_time
            final_resources = self._get_resource_info()
            
            timing_data = {
                "retrieval_time": retrieval_time,
                "context_time": context_time,
                "generation_time": generation_time,
                "total_time": total_time
            }
            
            # Save to conversation history with full metrics
            self._save_conversation_entry(user_query, result, initial_resources, final_resources, timing_data)
            
            return result
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            error_result = self._create_error_result(user_query, str(e))
            
            # Capture final resources for error case
            total_time = time.time() - start_total_time
            final_resources = self._get_resource_info()
            
            timing_data = {
                "retrieval_time": 0,
                "context_time": 0,
                "generation_time": 0,
                "total_time": total_time
            }
            
            # Save error to conversation history as well
            self._save_conversation_entry(user_query, error_result, initial_resources, final_resources, timing_data)
            
            return error_result
    
    def _retrieve_with_decomposition(
        self, 
        user_query: str, 
        top_k: int, 
        min_similarity: float
    ) -> Tuple[List, float]:
        """Retrieve using query decomposition and fusion re-ranking with detailed logging."""
        import time
        start_time = time.time()
        
        logger.info(f"Original query: {user_query}")
        logger.info(f"Parameters: top_k={top_k}, min_similarity={min_similarity}")
        
        # Step 1: Decompose query
        logger.info("-" * 80)
        logger.info("STEP 1: QUERY DECOMPOSITION")
        decompose_start = time.time()
        decomposed = self.query_decomposer.decompose_query(user_query)
        decompose_time = time.time() - decompose_start
        
        logger.info(f"Decomposition completed in {decompose_time:.3f}s")
        logger.info(f"Strategy: {decomposed.decomposition_type}")
        logger.info(f"Generated {decomposed.query_count} sub-queries:")
        for idx, sq in enumerate(decomposed.sub_queries, 1):
            logger.info(f"  [{idx}] {sq}")
        logger.info(f"Reasoning: {decomposed.decomposition_reasoning}")
        
        # Step 2: Retrieve for each sub-query
        logger.info("-" * 80)
        logger.info("STEP 2: MULTI-QUERY RETRIEVAL")
        retrieve_start = time.time()
        chunk_results = self.retriever.retrieve_multi_query(
            queries=decomposed.sub_queries,
            top_k_per_query=top_k,
            min_similarity=min_similarity
        )
        retrieve_time = time.time() - retrieve_start
        
        logger.info(f"Multi-query retrieval completed in {retrieve_time:.3f}s")
        total_retrieved = sum(len(chunks) for chunks in chunk_results.values())
        logger.info(f"Total chunks retrieved: {total_retrieved}")
        for idx, (query, chunks) in enumerate(chunk_results.items(), 1):
            logger.info(f"  Sub-query [{idx}] retrieved {len(chunks)} chunks (avg score: {sum(c.similarity_score for c in chunks)/len(chunks) if chunks else 0:.3f})")
        
        # Step 3: Fusion re-ranking
        logger.info("-" * 80)
        logger.info("STEP 3: FUSION RE-RANKING")
        fusion_start = time.time()
        fused_chunks = self.fusion_reranker.fuse_and_rerank(
            sub_queries=decomposed.sub_queries,
            chunk_results=chunk_results,
            top_k=top_k,
            reranker=self.retriever.reranker if self.enable_reranking else None
        )
        fusion_time = time.time() - fusion_start
        
        logger.info(f"Fusion re-ranking completed in {fusion_time:.3f}s")
        logger.info(f"Final top-K chunks: {len(fused_chunks)}")
        if fused_chunks:
            logger.info(f"Top chunk score: {fused_chunks[0].similarity_score:.3f}")
            logger.info(f"Lowest chunk score: {fused_chunks[-1].similarity_score:.3f}")
        
        retrieval_time = time.time() - start_time
        logger.info("-" * 80)
        logger.info(f"TOTAL DECOMPOSITION PIPELINE: {retrieval_time:.3f}s")
        logger.info(f"  - Decomposition: {decompose_time:.3f}s ({decompose_time/retrieval_time*100:.1f}%)")
        logger.info(f"  - Multi-retrieval: {retrieve_time:.3f}s ({retrieve_time/retrieval_time*100:.1f}%)")
        logger.info(f"  - Fusion: {fusion_time:.3f}s ({fusion_time/retrieval_time*100:.1f}%)")
        logger.info("=" * 80)
        
        return fused_chunks, retrieval_time
    
    def _validate_connection(self, user_query: str) -> None:
        """Validate RAG system connection."""
        if not self.connected:
            logger.error("RAG system not connected. Call connect() first.")
            raise ConnectionError("System not connected")
    
    def _retrieve_chunks(self, user_query: str, top_k: int, min_similarity: float) -> Tuple[List, float]:
        """Retrieve relevant chunks with timing (includes re-ranking if enabled) - optimized."""
        logger.debug(f"Retrieving chunks for query: {user_query[:50]}...")
        
        start_time = time.time()
        chunks = self.retriever.retrieve(
            query=user_query,
            top_k=top_k,
            min_similarity=min_similarity,
            retrieval_multiplier=self.retrieval_multiplier
        )
        retrieval_time = time.time() - start_time
        
        # Optimized logging with fewer string operations
        chunk_count = len(chunks)
        method = "Retrieved and re-ranked" if self.enable_reranking else "Retrieved"
        logger.info(f"{method} {chunk_count} chunks in {retrieval_time:.2f}s")
        return chunks, retrieval_time
    
    def _format_context(self, user_query: str, chunks: List, system_prompt: Optional[str]) -> Any:
        """Format chunks into LLM context."""
        logger.debug("Formatting context...")
        return self.formatter.format_context(
            query=user_query,
            chunks=chunks,
            system_prompt=system_prompt
        )
    
    def _generate_response(self, rag_prompt: Any) -> Tuple[Any, float]:
        """Generate LLM response with timing."""
        import time
        logger.debug("Generating LLM response...")
        
        start_time = time.time()
        llm_response = self.llm_client.generate_response(rag_prompt)
        generation_time = time.time() - start_time
        
        logger.info(f"Generated response in {generation_time:.2f}s")
        return llm_response, generation_time
    
    def _build_result(self, user_query: str, chunks: List, llm_response: Any, 
                     retrieval_time: float, generation_time: float) -> RAGResult:
        """Build final RAG result."""
        return RAGResult(
            query=user_query,
            response=llm_response.response,
            retrieved_chunks=chunks,
            context_token_count=llm_response.context_token_count,
            response_tokens=llm_response.response_tokens,
            model_used=llm_response.model_used,
            retrieval_time=retrieval_time,
            generation_time=generation_time
        )
    
    def _create_error_result(self, query: str, error: str) -> RAGResult:
        """Create error result."""
        return RAGResult(
            query=query,
            response=f"Error: {error}",
            retrieved_chunks=[],
            context_token_count=0,
            response_tokens=0,
            model_used="error"
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        stats = {
            "connected": self.connected,
            "retriever_stats": {},
            "reranking_config": {
                "enabled": self.enable_reranking,
                "retrieval_multiplier": self.retrieval_multiplier,
                "reranker_info": self.retriever.reranker.get_model_info() if self.enable_reranking and self.retriever.reranker else None
            },
            "formatter_config": {
                "max_context_tokens": self.formatter.max_context_tokens,
                "include_metadata": self.formatter.include_metadata,
                "include_scores": self.formatter.include_scores
            },
            "llm_config": {
                "type": type(self.llm_client).__name__,
                "model": getattr(self.llm_client, 'model', 'unknown')
            }
        }
        
        if self.connected:
            stats["retriever_stats"] = self.retriever.get_collection_stats()
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


def create_rag_system(
    llm_type: str = "mock",
    collection_name: str = "elastic_embeddings_m3",
    enable_reranking: bool = False,
    **kwargs
) -> RAGSystem:
    """Factory function to create configured RAG system."""
    return RAGSystem(
        llm_type=llm_type,
        collection_name=collection_name,
        enable_reranking=enable_reranking,
        **kwargs
    )


def ask_rag(
    query: str,
    top_k: int = 10,
    llm_type: str = "mock",
    enable_reranking: bool = False,
    **kwargs
) -> str:
    """Simple function to ask RAG system without managing lifecycle."""
    with create_rag_system(llm_type=llm_type, enable_reranking=enable_reranking, **kwargs) as rag:
        result = rag.query(query, top_k=top_k)
        return result.response


def test_reranking_comparison(test_query: str = "What are the key financial metrics and performance indicators?", top_k: int = 10):
    """
    Compare retrieval with and without re-ranking for debugging and performance analysis.
    
    Args:
        test_query: Query to test with
        top_k: Number of final chunks to retrieve
    
    Returns:
        Dict with comparison results
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    results = {
        "query": test_query,
        "top_k": top_k,
        "without_reranking": {},
        "with_reranking": {},
        "comparison": {}
    }
    
    print("=== RE-RANKING COMPARISON TEST ===\n")
    
    # Test 1: Without re-ranking (traditional approach)
    print("1. Testing WITHOUT re-ranking (traditional approach):")
    try:
        with create_rag_system(llm_type="mock", enable_reranking=False) as rag_no_rerank:
            result_no_rerank = rag_no_rerank.query(test_query, top_k=top_k)
            
            results["without_reranking"] = {
                "chunks_retrieved": len(result_no_rerank.retrieved_chunks),
                "retrieval_time": result_no_rerank.retrieval_time,
                "total_time": result_no_rerank.retrieval_time + (result_no_rerank.generation_time or 0),
                "top_chunks": [
                    {"id": chunk.chunk_id, "score": chunk.similarity_score}
                    for chunk in result_no_rerank.retrieved_chunks[:3]
                ]
            }
            
            print(f"   ✓ Retrieved {len(result_no_rerank.retrieved_chunks)} chunks")
            print(f"   ✓ Retrieval time: {result_no_rerank.retrieval_time:.2f}s")
            
    except Exception as e:
        print(f"   ✗ Error testing without re-ranking: {e}")
        results["without_reranking"]["error"] = str(e)
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: With re-ranking (enhanced approach)
    print("2. Testing WITH re-ranking (enhanced approach):")
    try:
        with create_rag_system(llm_type="mock", enable_reranking=True, retrieval_multiplier=10) as rag_rerank:
            result_rerank = rag_rerank.query(test_query, top_k=top_k)
            
            # Get system stats for detailed info
            stats = rag_rerank.get_system_stats()
            rerank_config = stats.get('reranking_config', {})
            
            results["with_reranking"] = {
                "chunks_retrieved": len(result_rerank.retrieved_chunks),
                "retrieval_time": result_rerank.retrieval_time,
                "total_time": result_rerank.retrieval_time + (result_rerank.generation_time or 0),
                "initial_retrieval": top_k * 10,  # 10x multiplier
                "reranker_model": rerank_config.get('reranker_info', {}).get('model_name', 'N/A'),
                "top_chunks": [
                    {"id": chunk.chunk_id, "score": chunk.similarity_score}
                    for chunk in result_rerank.retrieved_chunks[:3]
                ]
            }
            
            print(f"   ✓ Retrieved and re-ranked to {len(result_rerank.retrieved_chunks)} chunks")
            print(f"   ✓ Total time (retrieval + re-ranking): {result_rerank.retrieval_time:.2f}s")
            print(f"   ✓ Initial retrieval: {top_k * 10} chunks")
            print(f"   ✓ Re-ranker model: {rerank_config.get('reranker_info', {}).get('model_name', 'N/A')}")
            
    except Exception as e:
        print(f"   ✗ Error testing with re-ranking: {e}")
        results["with_reranking"]["error"] = str(e)
    
    # Generate comparison
    if "error" not in results["without_reranking"] and "error" not in results["with_reranking"]:
        time_diff = results["with_reranking"]["total_time"] - results["without_reranking"]["total_time"]
        results["comparison"] = {
            "time_difference": time_diff,
            "reranking_overhead": f"{time_diff:.2f}s" if time_diff > 0 else "0s",
            "performance_impact": "slower" if time_diff > 0 else "faster",
            "initial_vs_final": f"{results['with_reranking']['initial_retrieval']} → {results['with_reranking']['chunks_retrieved']}"
        }
        
        print("\n" + "="*50 + "\n")
        print("3. Performance Comparison:")
        print(f"   • Traditional: {results['without_reranking']['total_time']:.2f}s")
        print(f"   • Re-ranking: {results['with_reranking']['total_time']:.2f}s")
        print(f"   • Overhead: {results['comparison']['reranking_overhead']}")
        print(f"   • Retrieval pattern: {results['comparison']['initial_vs_final']}")
    
    print("\n✓ Re-ranking comparison test completed!")
    return results


def benchmark_optimization_performance(num_runs: int = 5, chunk_sizes: List[int] = [10, 50, 100]) -> Dict[str, Any]:
    """
    Benchmark the performance improvements from optimizations.
    
    Args:
        num_runs: Number of test runs for averaging
        chunk_sizes: Different chunk sizes to test with
    
    Returns:
        Dict with benchmark results
    """
    import time
    import statistics
    
    print("=== OPTIMIZATION PERFORMANCE BENCHMARK ===\n")
    
    benchmark_results = {
        "test_config": {
            "num_runs": num_runs,
            "chunk_sizes": chunk_sizes,
            "timestamp": datetime.now().isoformat()
        },
        "results": {}
    }
    
    test_query = "What are the key financial metrics and revenue trends?"
    
    for chunk_size in chunk_sizes:
        print(f"Testing with {chunk_size} chunks:")
        
        # Test without re-ranking (baseline)
        no_rerank_times = []
        for run in range(num_runs):
            try:
                with create_rag_system(llm_type="mock", enable_reranking=False) as rag:
                    start_time = time.time()
                    rag.query(test_query, top_k=chunk_size)
                    end_time = time.time()
                    no_rerank_times.append(end_time - start_time)
            except Exception as e:
                print(f"   ⚠️ Run {run+1} failed (no re-ranking): {e}")
        
        # Test with re-ranking (optimized)
        rerank_times = []
        for run in range(num_runs):
            try:
                with create_rag_system(llm_type="mock", enable_reranking=True, retrieval_multiplier=10) as rag:
                    start_time = time.time()
                    rag.query(test_query, top_k=chunk_size)
                    end_time = time.time()
                    rerank_times.append(end_time - start_time)
            except Exception as e:
                print(f"   ⚠️ Run {run+1} failed (with re-ranking): {e}")
        
        if no_rerank_times and rerank_times:
            # Calculate statistics
            no_rerank_avg = statistics.mean(no_rerank_times)
            no_rerank_std = statistics.stdev(no_rerank_times) if len(no_rerank_times) > 1 else 0
            
            rerank_avg = statistics.mean(rerank_times)
            rerank_std = statistics.stdev(rerank_times) if len(rerank_times) > 1 else 0
            
            overhead = rerank_avg - no_rerank_avg
            overhead_pct = (overhead / no_rerank_avg) * 100 if no_rerank_avg > 0 else 0
            
            benchmark_results["results"][f"{chunk_size}_chunks"] = {
                "no_reranking": {
                    "avg_time": no_rerank_avg,
                    "std_dev": no_rerank_std,
                    "min_time": min(no_rerank_times),
                    "max_time": max(no_rerank_times)
                },
                "with_reranking": {
                    "avg_time": rerank_avg,
                    "std_dev": rerank_std,
                    "min_time": min(rerank_times),
                    "max_time": max(rerank_times)
                },
                "performance": {
                    "overhead_seconds": overhead,
                    "overhead_percentage": overhead_pct,
                    "initial_chunks": chunk_size * 10,
                    "final_chunks": chunk_size
                }
            }
            
            print(f"   • No re-ranking: {no_rerank_avg:.3f}s ± {no_rerank_std:.3f}s")
            print(f"   • With re-ranking: {rerank_avg:.3f}s ± {rerank_std:.3f}s")
            print(f"   • Overhead: {overhead:.3f}s ({overhead_pct:.1f}%)")
            print(f"   • Quality gain: {chunk_size * 10} → {chunk_size} chunks\n")
    
    print("✓ Optimization benchmark completed!")
    return benchmark_results


# Main CLI interface functionality
logger = logging.getLogger(__name__)


def setup_clean_logging():
    """Setup detailed logging for CLI with clean format."""
    # Configure root logger with clean format
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',  # Clean format without timestamps for better readability
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress noisy third-party loggers
    logging.getLogger('pymilvus').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    # Suppress protobuf warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')


def main():
    """Main RAG interface with detailed logging."""
    setup_clean_logging()
    
    print("=" * 80)
    print("ADVANCED RAG SYSTEM - Query Decomposition & Fusion Re-ranking")
    print("=" * 80)
    print()
    
    # Step 1: Initialize system components
    print("1. Initializing RAG components...")
    print("   - Collection: elastic_embeddings_m3 (ECS documentation)")
    print("   - Embedding model: BAAI/bge-m3 (1024 dim)")
    print("   - Re-ranking: ENABLED (retrieval_multiplier=50)")
    print("   - Query decomposition: ENABLED (max_sub_queries=5)")
    print()
    rag = create_rag_system(
        llm_type="openai",
        collection_name="elastic_embeddings_m3",
        embedding_model="BAAI/bge-m3",
        enable_reranking=True,
        retrieval_multiplier=50,
        enable_query_decomposition=True,
        max_sub_queries=5
    )
    
    # Step 2: Connect to Milvus
    print("2. Connecting to Milvus vector database...")
    if not rag.connect():
        print("ERROR: Could not connect to Milvus.")
        print("Make sure Milvus is running: docker-compose up -d")
        return 1
    
    # Step 3: Check data availability (with fallback test search)
    stats = rag.get_system_stats()
    num_entities = stats.get('retriever_stats', {}).get('num_entities', 0)
    print(f"3. Connected to collection with {num_entities} document chunks")
    
    if num_entities == 0:
        print("   Entity count shows 0, trying test search to verify data availability...")
        try:
            # Try a test search to see if data is actually there (Milvus consistency issue)
            test_chunks = rag.retriever.retrieve("test query", top_k=1)
            if test_chunks:
                print(f"   Found data via search! {len(test_chunks)} chunks available")
                print("   (Entity count may show 0 due to Milvus consistency delay)")
                num_entities = 1  # Set to non-zero to continue
            else:
                print("ERROR: No documents found via search either. Upload data first.")
                return 1
        except Exception as e:
            print(f"ERROR: Could not verify data availability: {e}")
            return 1
    
    # Step 4: Initialize LLM (this will prompt for API key)
    print("4. Initializing OpenAI LLM client...")
    print("   (API key will be requested securely)")
    
    # Step 5: Ready for query
    print("5. RAG system ready!")
    print()
    
    try:
        # Get user query
        query = input("Enter your query: ").strip()
        
        if not query:
            print("No query provided. Exiting.")
            return 1
        
        # Process query through RAG pipeline
        print()
        print("=" * 80)
        print("PROCESSING QUERY")
        print("=" * 80)
        result = rag.query(query, top_k=10)
        print()
        
        # Display response
        print(f"\nResponse:")
        print("-" * 50)
        print(result.response)
        print("-" * 50)
        print(f"Sources: {len(result.retrieved_chunks)} chunks | Model: {result.model_used}")
        print()
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        rag.disconnect()
        print("RAG system disconnected.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())