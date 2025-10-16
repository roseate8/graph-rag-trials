"""
Evaluator orchestrates the entire evaluation pipeline.

Optimized for memory efficiency and minimal I/O overhead.
Loads queries/qrels, performs batch retrieval, calculates metrics, and saves results.
"""

import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

from config import EvalConfig
from retriever_for_evals import EvalRetriever, RetrievalResult
from metrics import IRMetrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Main evaluator that orchestrates the evaluation pipeline.
    
    Pipeline:
    1. Load queries and qrels
    2. Batch async retrieval
    3. Calculate metrics per query
    4. Aggregate metrics overall and by query type
    5. Save all results
    """
    
    def __init__(self, config: EvalConfig):
        """
        Initialize evaluator.
        
        Args:
            config: EvalConfig instance
        """
        self.config = config
        self.retriever = EvalRetriever(config)
        
        # Data storage
        self.queries = []
        self.qrels = defaultdict(dict)  # query_id -> {doc_id: relevance}
        self.corpus = {}  # doc_id -> content
        
        # Results storage
        self.retrieval_results = []
        self.per_query_metrics = []
        self.failed_queries = []
    
    def load_queries(self) -> bool:
        """Load queries from JSONL file. Optimized: single-pass type counting."""
        queries_path = Path(self.config.queries_file)

        if not queries_path.exists():
            logger.error(f"Queries file not found: {queries_path}")
            return False

        try:
            type_counts = defaultdict(int)

            # Single-pass loading and counting
            with open(queries_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        query = json.loads(line)
                        self.queries.append(query)

                        # Count types during load (single pass)
                        qtype = query.get('metadata', {}).get('query_type', 'unknown')
                        type_counts[qtype] += 1

            logger.info(f"✓ Loaded {len(self.queries)} queries from {queries_path}")
            logger.info(f"  Query types: {dict(type_counts)}")

            return True
        except Exception as e:
            logger.error(f"Error loading queries: {e}")
            return False
    
    def load_qrels(self) -> bool:
        """Load qrels (relevance judgments) from TSV file."""
        qrels_path = Path(self.config.qrels_file)
        
        if not qrels_path.exists():
            logger.error(f"Qrels file not found: {qrels_path}")
            return False
        
        try:
            with open(qrels_path, 'r', encoding='utf-8') as f:
                # Skip header line
                next(f)
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue
                    
                    query_id, doc_id, score = parts
                    self.qrels[query_id][doc_id] = int(score)
            
            logger.info(f"✓ Loaded qrels for {len(self.qrels)} queries from {qrels_path}")
            
            # Calculate statistics
            total_judgments = sum(len(docs) for docs in self.qrels.values())
            avg_judgments = total_judgments / len(self.qrels) if self.qrels else 0
            
            logger.info(f"  Total judgments: {total_judgments}")
            logger.info(f"  Avg judgments per query: {avg_judgments:.1f}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading qrels: {e}")
            return False
    
    def load_corpus(self) -> bool:
        """Load corpus (optional - for analysis)."""
        corpus_path = Path(self.config.corpus_file)
        
        if not corpus_path.exists():
            logger.warning(f"Corpus file not found: {corpus_path} (optional)")
            return True  # Not critical
        
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    doc_id = doc.get('_id', '')
                    if doc_id:
                        self.corpus[doc_id] = doc
            
            logger.info(f"✓ Loaded {len(self.corpus)} documents from corpus")
            return True
        except Exception as e:
            logger.warning(f"Error loading corpus (non-critical): {e}")
            return True  # Not critical
    
    async def run_retrieval(self) -> bool:
        """
        Run batch async retrieval for all queries.
        
        Returns:
            True if successful
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING BATCH RETRIEVAL")
        logger.info(f"{'='*60}")
        
        if not self.retriever.connect():
            logger.error("Failed to connect to retriever")
            return False
        
        try:
            # Get max K value for retrieval
            max_k = self.config.get_max_k()
            
            # Run batch async retrieval
            self.retrieval_results = await self.retriever.retrieve_batch(
                queries=self.queries,
                top_k=max_k,
                show_progress=True
            )
            
            # Separate successful and failed
            self.failed_queries = [r for r in self.retrieval_results if not r.success]
            
            if self.failed_queries:
                logger.warning(f"⚠ {len(self.failed_queries)} queries failed retrieval")
            
            return True
        
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return False
        finally:
            self.retriever.disconnect()
    
    def calculate_metrics(self):
        """Calculate metrics for all queries."""
        logger.info(f"\n{'='*60}")
        logger.info(f"CALCULATING METRICS")
        logger.info(f"{'='*60}")
        
        for result in self.retrieval_results:
            if not result.success:
                continue
            
            query_id = result.query_id
            query_type = result.query_type
            
            # Get retrieved doc IDs
            retrieved_ids = [doc['chunk_id'] for doc in result.retrieved_docs]
            
            # Get relevance scores from qrels
            relevance_scores = self.qrels.get(query_id, {})
            
            if not relevance_scores:
                logger.warning(f"No qrels found for query {query_id}")
                continue
            
            # Calculate all metrics for this query
            query_metrics = IRMetrics.calculate_all_metrics(
                query_id=query_id,
                retrieved_ids=retrieved_ids,
                relevance_scores=relevance_scores,
                k_values=self.config.k_values
            )
            
            # Add query type for breakdown
            query_metrics['query_type'] = query_type
            
            self.per_query_metrics.append(query_metrics)
        
        logger.info(f"✓ Calculated metrics for {len(self.per_query_metrics)} queries")
    
    def aggregate_results(self) -> Tuple[Dict, Dict, Dict]:
        """
        Aggregate results overall and by query type.
        
        Returns:
            Tuple of (overall_metrics, by_type_metrics, by_k_metrics)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"AGGREGATING RESULTS")
        logger.info(f"{'='*60}")
        
        # 1. Overall metrics
        overall_metrics = IRMetrics.aggregate_metrics(
            self.per_query_metrics,
            self.config.k_values
        )
        overall_metrics['evaluation_type'] = 'overall'
        
        logger.info(f"✓ Aggregated overall metrics")
        
        # 2. By query type (single-hop vs multi-hop)
        by_type_metrics = {}
        
        for query_type in ['single_hop', 'multi_hop', 'unknown']:
            type_queries = [m for m in self.per_query_metrics if m['query_type'] == query_type]
            
            if type_queries:
                by_type_metrics[query_type] = IRMetrics.aggregate_metrics(
                    type_queries,
                    self.config.k_values
                )
                by_type_metrics[query_type]['query_type'] = query_type
                
                logger.info(f"✓ Aggregated metrics for {query_type}: {len(type_queries)} queries")
        
        # 3. By K value (for visualization)
        by_k_metrics = {}
        
        for k in self.config.k_values:
            by_k_metrics[f"k={k}"] = {
                'k': k,
                'recall': overall_metrics.get(f'recall@{k}', 0.0),
                'precision': overall_metrics.get(f'precision@{k}', 0.0),
                'ndcg': overall_metrics.get(f'ndcg@{k}', 0.0),
                'hits': overall_metrics.get(f'hits@{k}', 0.0)
            }
        
        # Add MAP and MRR
        by_k_metrics['map'] = {'metric': 'MAP', 'value': overall_metrics.get('MAP', 0.0)}
        by_k_metrics['mrr'] = {'metric': 'MRR', 'value': overall_metrics.get('MRR', 0.0)}
        
        return overall_metrics, by_type_metrics, by_k_metrics
    
    def save_results(self, overall_metrics: Dict, by_type_metrics: Dict, by_k_metrics: Dict):
        """Save all results to files."""
        logger.info(f"\n{'='*60}")
        logger.info(f"SAVING RESULTS")
        logger.info(f"{'='*60}")
        
        # Ensure results directory exists
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save retrieval results (raw)
        retrieval_path = Path(self.config.retrieval_results_file)
        with open(retrieval_path, 'w', encoding='utf-8') as f:
            for result in self.retrieval_results:
                result_dict = {
                    'query_id': result.query_id,
                    'query_text': result.query_text,
                    'query_type': result.query_type,
                    'retrieved_docs': result.retrieved_docs,
                    'success': result.success,
                    'error': result.error
                }
                f.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
        logger.info(f"✓ Saved retrieval results: {retrieval_path}")
        
        # 2. Save overall metrics
        overall_path = Path(self.config.metrics_overall_file)
        with open(overall_path, 'w', encoding='utf-8') as f:
            json.dump(overall_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved overall metrics: {overall_path}")
        
        # 3. Save by-type metrics
        by_type_path = Path(self.config.metrics_by_type_file)
        with open(by_type_path, 'w', encoding='utf-8') as f:
            json.dump(by_type_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved by-type metrics: {by_type_path}")
        
        # 4. Save by-k metrics
        by_k_path = Path(self.config.metrics_by_k_file)
        with open(by_k_path, 'w', encoding='utf-8') as f:
            json.dump(by_k_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved by-k metrics: {by_k_path}")
        
        # 5. Save failed queries
        if self.failed_queries:
            failed_path = Path(self.config.failed_queries_file)
            with open(failed_path, 'w', encoding='utf-8') as f:
                for result in self.failed_queries:
                    failed_dict = {
                        'query_id': result.query_id,
                        'query_text': result.query_text,
                        'error': result.error
                    }
                    f.write(json.dumps(failed_dict, ensure_ascii=False) + '\n')
            logger.info(f"⚠ Saved failed queries: {failed_path}")
    
    async def run(self) -> bool:
        """
        Run complete evaluation pipeline.
        
        Returns:
            True if successful
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"SYNTHETIC EVALUATION - METRICS CALCULATION")
        logger.info(f"{'='*80}")
        
        # Step 1: Load data
        logger.info(f"\n[Step 1/5] Loading data...")
        if not self.load_queries():
            return False
        if not self.load_qrels():
            return False
        self.load_corpus()  # Optional
        
        # Step 2: Run retrieval
        logger.info(f"\n[Step 2/5] Running batch retrieval...")
        if not await self.run_retrieval():
            return False
        
        # Step 3: Calculate metrics
        logger.info(f"\n[Step 3/5] Calculating metrics...")
        self.calculate_metrics()
        
        # Step 4: Aggregate results
        logger.info(f"\n[Step 4/5] Aggregating results...")
        overall_metrics, by_type_metrics, by_k_metrics = self.aggregate_results()
        
        # Step 5: Save results
        logger.info(f"\n[Step 5/5] Saving results...")
        self.save_results(overall_metrics, by_type_metrics, by_k_metrics)
        
        # Generate report (next step)
        from reporter import Reporter
        reporter = Reporter(self.config)
        reporter.generate_report(
            overall_metrics=overall_metrics,
            by_type_metrics=by_type_metrics,
            by_k_metrics=by_k_metrics,
            per_query_metrics=self.per_query_metrics,
            failed_queries=self.failed_queries
        )
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EVALUATION COMPLETE!")
        logger.info(f"{'='*80}")
        logger.info(f"\nResults saved to: {self.config.results_dir}/")
        
        return True

