#!/usr/bin/env python3
"""
Optimized HotpotQA BEIR Evaluation for Naive RAG System.

This script efficiently evaluates the naive RAG system on the HotpotQA dataset
with optimized performance and minimal dependencies.
"""

import sys
import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "naive-rag"))
sys.path.insert(0, str(project_root / "vector-ingest" / "src"))

try:
    from retrieval import MilvusRetriever
    from config import get_config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure naive-rag module is available.")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)


class OptimizedHotpotQAEvaluator:
    """Optimized HotpotQA evaluator with efficient batch processing."""
    
    def __init__(self, collection_name: str = "bier_hotpotqa_chunks"):
        self.collection_name = collection_name
        self.data_dir = Path(__file__).parent / "data"
        self.results_dir = Path(__file__).parent / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize retriever
        self.retriever = None
        self._init_retriever()
    
    def _init_retriever(self):
        """Initialize retriever with optimized settings."""
        print("🔌 Initializing retriever...")
        self.retriever = MilvusRetriever(
            embedding_model="BAAI/bge-small-en-v1.5",
            milvus_profile="production",
            collection_name=self.collection_name,
            enable_reranking=True,
            reranker_config={
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "batch_size": 64  # Increased batch size for efficiency
            }
        )
        
        # Connect with fallback to default collection
        if not self.retriever.connect():
            print(f"⚠️  Could not connect to {self.collection_name}. Trying default collection...")
            self.retriever = MilvusRetriever(
                embedding_model="BAAI/bge-small-en-v1.5",
                milvus_profile="production",
                collection_name="document_chunks",
                enable_reranking=True,
                reranker_config={
                    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "batch_size": 64
                }
            )
            if not self.retriever.connect():
                raise ConnectionError("Could not connect to Milvus")
            self.collection_name = "document_chunks"
        
        print(f"✅ Connected to collection: {self.collection_name}")
    
    def download_hotpotqa_dataset(self) -> None:
        """Download HotpotQA dataset if not available."""
        import requests
        
        zip_path = self.data_dir / "hotpotqa" / "hotpotqa.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        
        if zip_path.exists():
            # Verify it's a valid zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.testzip()
                return  # File is valid
            except zipfile.BadZipFile:
                print("⚠️  Existing zip file is corrupted, re-downloading...")
                zip_path.unlink()
        
        print("📥 Downloading HotpotQA dataset...")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print("✅ Download completed")
    
    def load_hotpotqa_dataset(self) -> Tuple[Dict, Dict, Dict]:
        """Load HotpotQA dataset efficiently from zip file."""
        print("📥 Loading HotpotQA dataset...")
        
        # Ensure dataset is downloaded and valid
        self.download_hotpotqa_dataset()
        
        zip_path = self.data_dir / "hotpotqa" / "hotpotqa.zip"
        corpus, queries, qrels = {}, {}, defaultdict(dict)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Load corpus efficiently
            try:
                with zip_ref.open('hotpotqa/corpus.jsonl') as f:
                    for line in f:
                        doc = json.loads(line.decode('utf-8'))
                        corpus[doc['_id']] = {
                            'title': doc.get('title', ''),
                            'text': doc.get('text', '')
                        }
            except KeyError:
                raise FileNotFoundError("corpus.jsonl not found in dataset")
            
            # Load queries efficiently
            try:
                with zip_ref.open('hotpotqa/queries.jsonl') as f:
                    for line in f:
                        query = json.loads(line.decode('utf-8'))
                        queries[query['_id']] = query['text']
            except KeyError:
                raise FileNotFoundError("queries.jsonl not found in dataset")
            
            # Load relevance judgments efficiently
            try:
                with zip_ref.open('hotpotqa/qrels/test.tsv') as f:
                    for line in f:
                        parts = line.decode('utf-8').strip().split('\t')
                        if len(parts) >= 4:
                            query_id, _, doc_id, relevance = parts[:4]
                            qrels[query_id][doc_id] = int(relevance)
            except KeyError:
                raise FileNotFoundError("qrels/test.tsv not found in dataset")
        
        print(f"📊 Dataset loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
        return corpus, queries, dict(qrels)
    
    def evaluate_batch(self, query_batch: List[Tuple[str, str]], 
                      top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """Evaluate a batch of queries efficiently."""
        results = {}
        
        for query_id, query_text in query_batch:
            try:
                # Retrieve with optimized settings
                chunks = self.retriever.retrieve(
                    query=query_text,
                    top_k=top_k,
                    min_similarity=0.0  # Get all results for proper evaluation
                )
                
                # Convert to evaluation format
                results[query_id] = [
                    (chunk.chunk_id, chunk.rerank_score if chunk.rerank_score else chunk.similarity_score)
                    for chunk in chunks
                ]
                
            except Exception as e:
                logger.warning(f"Error processing query {query_id}: {e}")
                results[query_id] = []
        
        return results
    
    def calculate_metrics(self, results: Dict[str, List[Tuple[str, float]]], 
                         qrels: Dict[str, Dict[str, int]], 
                         k_values: List[int] = [1, 3, 5, 10, 20]) -> Dict[str, float]:
        """Calculate BEIR evaluation metrics efficiently."""
        metrics = {}
        
        # Pre-compute relevant docs for efficiency
        relevant_docs_cache = {}
        for query_id in qrels:
            relevant_docs_cache[query_id] = set(
                doc_id for doc_id, rel in qrels[query_id].items() if rel > 0
            )
        
        for k in k_values:
            precisions, recalls, ndcgs, maps = [], [], [], []
            
            for query_id, retrieved_docs in results.items():
                if query_id not in relevant_docs_cache:
                    continue
                
                relevant_docs = relevant_docs_cache[query_id]
                if not relevant_docs:
                    continue
                
                # Get top-k results
                top_k_docs = [doc_id for doc_id, _ in retrieved_docs[:k]]
                relevant_retrieved = set(top_k_docs) & relevant_docs
                
                # Precision@k
                precision = len(relevant_retrieved) / k if k > 0 else 0
                precisions.append(precision)
                
                # Recall@k
                recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
                recalls.append(recall)
                
                # NDCG@k (optimized calculation)
                dcg = sum(
                    qrels[query_id].get(doc_id, 0) / np.log2(i + 2)
                    for i, doc_id in enumerate(top_k_docs)
                    if qrels[query_id].get(doc_id, 0) > 0
                )
                
                ideal_relevances = sorted(
                    [rel for rel in qrels[query_id].values() if rel > 0], 
                    reverse=True
                )
                idcg = sum(
                    rel / np.log2(i + 2) 
                    for i, rel in enumerate(ideal_relevances[:k])
                )
                
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcgs.append(ndcg)
                
                # MAP calculation (for k=all)
                if k == max(k_values):
                    ap = 0
                    relevant_found = 0
                    for i, doc_id in enumerate(top_k_docs):
                        if doc_id in relevant_docs:
                            relevant_found += 1
                            ap += relevant_found / (i + 1)
                    ap = ap / len(relevant_docs) if relevant_docs else 0
                    maps.append(ap)
            
            # Store average metrics
            if precisions:
                metrics[f'precision@{k}'] = np.mean(precisions)
            if recalls:
                metrics[f'recall@{k}'] = np.mean(recalls)
            if ndcgs:
                metrics[f'ndcg@{k}'] = np.mean(ndcgs)
            if maps and k == max(k_values):
                metrics['map'] = np.mean(maps)
        
        return metrics
    
    def run_evaluation(self, max_queries: Optional[int] = None, 
                      batch_size: int = 50) -> Dict:
        """Run complete HotpotQA evaluation with optimizations."""
        start_time = time.time()
        
        # Load dataset
        corpus, queries, qrels = self.load_hotpotqa_dataset()
        
        # Limit queries if specified
        if max_queries:
            queries = dict(list(queries.items())[:max_queries])
            print(f"🔬 Evaluating on {len(queries)} queries (limited for testing)")
        
        print(f"🚀 Starting evaluation of {len(queries)} queries...")
        
        # Process queries in batches for efficiency
        all_results = {}
        query_items = list(queries.items())
        
        for i in range(0, len(query_items), batch_size):
            batch = query_items[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(query_items) + batch_size - 1) // batch_size
            
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} queries)...")
            
            batch_results = self.evaluate_batch(batch)
            all_results.update(batch_results)
            
            # Progress update
            processed = min(i + batch_size, len(query_items))
            print(f"  Progress: {processed}/{len(query_items)} queries ({processed/len(query_items)*100:.1f}%)")
        
        # Calculate metrics
        print("📊 Calculating evaluation metrics...")
        metrics = self.calculate_metrics(all_results, qrels)
        
        # Prepare results
        evaluation_time = time.time() - start_time
        result_data = {
            'dataset': 'hotpotqa',
            'collection': self.collection_name,
            'evaluation_time_seconds': evaluation_time,
            'num_queries_evaluated': len(all_results),
            'num_queries_with_results': len([r for r in all_results.values() if r]),
            'success_rate': len([r for r in all_results.values() if r]) / len(all_results),
            'metrics': metrics,
            'retriever_config': {
                'embedding_model': 'BAAI/bge-small-en-v1.5',
                'enable_reranking': True,
                'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            }
        }
        
        # Save results
        result_file = self.results_dir / "hotpotqa_evaluation_results.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"💾 Results saved to: {result_file}")
        return result_data
    
    def close(self):
        """Clean up resources."""
        if self.retriever and hasattr(self.retriever, 'close'):
            self.retriever.close()


def main():
    """Main evaluation function."""
    print("🎯 HotpotQA BEIR Evaluation")
    print("===========================")
    
    evaluator = None
    try:
        # Initialize evaluator
        evaluator = OptimizedHotpotQAEvaluator()
        
        # Run evaluation (limit to 100 queries for initial test)
        print("Running evaluation with first 100 queries for testing...")
        results = evaluator.run_evaluation(max_queries=100)
        
        # Display results
        print("\n✅ HotpotQA Evaluation Results:")
        print("=" * 50)
        print(f"Collection: {results['collection']}")
        print(f"Queries Evaluated: {results['num_queries_evaluated']}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Evaluation Time: {results['evaluation_time_seconds']:.1f}s")
        
        print("\n📈 Key Metrics:")
        metrics = results['metrics']
        for metric in ['ndcg@10', 'map', 'recall@10', 'precision@10']:
            if metric in metrics:
                print(f"  {metric:15}: {metrics[metric]:.4f}")
        
        print("\n🎉 Evaluation completed successfully!")
        print("\nTo run full evaluation on all queries, modify max_queries=None in the script.")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if evaluator:
            evaluator.close()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
