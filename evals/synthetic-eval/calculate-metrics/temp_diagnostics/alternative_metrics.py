"""
Alternative Metrics Calculator

Calculates metrics with different matching strategies to see how they differ:
1. Strict matching (exact chunk ID must match) - CURRENT
2. Lenient matching (same document, any chunk)
3. Top-1 only matching (only care if #1 result is good)
4. Pruned qrels (only top-K most relevant per query)

This helps identify if the issue is with evaluation criteria vs actual retrieval.

Usage:
    python alternative_metrics.py
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import statistics


class AlternativeMetricsCalculator:
    """Calculate metrics with different matching strategies."""
    
    def __init__(self,
                 queries_file: str = "../output/queries.jsonl",
                 qrels_file: str = "../output/qrels.tsv",
                 retrieval_results_file: str = "../results/retrieval_results.jsonl"):
        self.queries_file = queries_file
        self.qrels_file = qrels_file
        self.retrieval_results_file = retrieval_results_file
        
        self.queries = {}
        self.qrels = defaultdict(dict)
        self.retrieval_results = {}
    
    def load_data(self):
        """Load data files."""
        print("Loading data...")
        
        with open(self.queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    self.queries[q['_id']] = q
        
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query_id, doc_id, score = parts
                    self.qrels[query_id][doc_id] = int(score)
        
        with open(self.retrieval_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    self.retrieval_results[result['query_id']] = result
        
        print(f"✓ Loaded {len(self.queries)} queries\n")
    
    def get_doc_base(self, chunk_id: str) -> str:
        """Extract base document ID."""
        if '_chunk_' in chunk_id:
            return chunk_id.split('_chunk_')[0]
        return chunk_id
    
    def calculate_hits_at_k(self, k: int, matching_strategy: str = "strict") -> float:
        """
        Calculate hits@k with different matching strategies.
        
        Args:
            k: Top-k to consider
            matching_strategy: "strict" (exact chunk) or "lenient" (same doc)
        
        Returns:
            Hits@k score (fraction of queries with at least one hit in top-k)
        """
        hits = 0
        total = 0
        
        for query_id, result in self.retrieval_results.items():
            if not result.get('success', False):
                continue
            
            total += 1
            qrels = self.qrels.get(query_id, {})
            
            if not qrels:
                continue
            
            retrieved_docs = result.get('retrieved_docs', [])[:k]
            retrieved_ids = [doc['chunk_id'] for doc in retrieved_docs]
            
            if matching_strategy == "strict":
                # Exact chunk ID must match
                if any(rid in qrels for rid in retrieved_ids):
                    hits += 1
            
            elif matching_strategy == "lenient":
                # Same document (any chunk) counts as hit
                qrel_doc_bases = {self.get_doc_base(qid) for qid in qrels.keys()}
                retrieved_doc_bases = [self.get_doc_base(rid) for rid in retrieved_ids]
                
                if any(rdb in qrel_doc_bases for rdb in retrieved_doc_bases):
                    hits += 1
        
        return hits / total if total > 0 else 0.0
    
    def calculate_recall_at_k(self, k: int, matching_strategy: str = "strict") -> float:
        """Calculate recall@k."""
        recall_scores = []
        
        for query_id, result in self.retrieval_results.items():
            if not result.get('success', False):
                continue
            
            qrels = self.qrels.get(query_id, {})
            if not qrels:
                continue
            
            retrieved_docs = result.get('retrieved_docs', [])[:k]
            retrieved_ids = [doc['chunk_id'] for doc in retrieved_docs]
            
            if matching_strategy == "strict":
                relevant_retrieved = sum(1 for rid in retrieved_ids if rid in qrels)
                total_relevant = len(qrels)
            
            elif matching_strategy == "lenient":
                qrel_doc_bases = {self.get_doc_base(qid) for qid in qrels.keys()}
                retrieved_doc_bases = [self.get_doc_base(rid) for rid in retrieved_ids]
                relevant_retrieved = len(set(retrieved_doc_bases) & qrel_doc_bases)
                total_relevant = len(qrel_doc_bases)
            
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
            recall_scores.append(recall)
        
        return statistics.mean(recall_scores) if recall_scores else 0.0
    
    def calculate_mrr(self, matching_strategy: str = "strict") -> float:
        """Calculate Mean Reciprocal Rank."""
        reciprocal_ranks = []
        
        for query_id, result in self.retrieval_results.items():
            if not result.get('success', False):
                continue
            
            qrels = self.qrels.get(query_id, {})
            if not qrels:
                continue
            
            retrieved_docs = result.get('retrieved_docs', [])
            
            if matching_strategy == "strict":
                for rank, doc in enumerate(retrieved_docs, 1):
                    if doc['chunk_id'] in qrels:
                        reciprocal_ranks.append(1.0 / rank)
                        break
                else:
                    reciprocal_ranks.append(0.0)
            
            elif matching_strategy == "lenient":
                qrel_doc_bases = {self.get_doc_base(qid) for qid in qrels.keys()}
                for rank, doc in enumerate(retrieved_docs, 1):
                    if self.get_doc_base(doc['chunk_id']) in qrel_doc_bases:
                        reciprocal_ranks.append(1.0 / rank)
                        break
                else:
                    reciprocal_ranks.append(0.0)
        
        return statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_pruned_qrels_metrics(self, max_qrels: int = 5) -> Dict:
        """
        Calculate metrics with pruned qrels (only keep top-N most relevant per query).
        This simulates what metrics would be if we only marked the MOST important chunks.
        """
        print(f"\nCalculating metrics with qrels pruned to top-{max_qrels} most relevant...")
        
        hits_scores = []
        recall_scores = []
        
        for query_id, result in self.retrieval_results.items():
            if not result.get('success', False):
                continue
            
            qrels = self.qrels.get(query_id, {})
            if not qrels:
                continue
            
            # Prune to top-N by relevance score
            sorted_qrels = sorted(qrels.items(), key=lambda x: x[1], reverse=True)
            pruned_qrels = dict(sorted_qrels[:max_qrels])
            
            retrieved_docs = result.get('retrieved_docs', [])[:10]
            retrieved_ids = [doc['chunk_id'] for doc in retrieved_docs]
            
            # Calculate hits@10
            has_hit = any(rid in pruned_qrels for rid in retrieved_ids)
            hits_scores.append(1.0 if has_hit else 0.0)
            
            # Calculate recall@10
            relevant_retrieved = sum(1 for rid in retrieved_ids if rid in pruned_qrels)
            recall = relevant_retrieved / len(pruned_qrels) if pruned_qrels else 0.0
            recall_scores.append(recall)
        
        return {
            'hits@10': statistics.mean(hits_scores) if hits_scores else 0.0,
            'recall@10': statistics.mean(recall_scores) if recall_scores else 0.0
        }
    
    def run_comparison(self):
        """Run full comparison of different metrics."""
        print("=" * 80)
        print("ALTERNATIVE METRICS COMPARISON")
        print("=" * 80)
        
        self.load_data()
        
        # Strategy 1: Current (Strict) metrics
        print("\n📊 STRATEGY 1: STRICT MATCHING (Current)")
        print("─" * 80)
        print("Requirement: Exact chunk ID must match qrels")
        print()
        
        strict_results = {}
        for k in [1, 3, 5, 10]:
            hits = self.calculate_hits_at_k(k, "strict")
            recall = self.calculate_recall_at_k(k, "strict")
            strict_results[k] = {'hits': hits, 'recall': recall}
            print(f"  hits@{k:2d}:   {hits:.3f} ({hits*100:.1f}%)")
            print(f"  recall@{k:2d}: {recall:.3f}")
        
        mrr_strict = self.calculate_mrr("strict")
        print(f"  MRR:      {mrr_strict:.3f}")
        
        # Strategy 2: Lenient (document-level) metrics
        print("\n📊 STRATEGY 2: LENIENT MATCHING (Document-level)")
        print("─" * 80)
        print("Requirement: Any chunk from the same document counts as correct")
        print()
        
        lenient_results = {}
        for k in [1, 3, 5, 10]:
            hits = self.calculate_hits_at_k(k, "lenient")
            recall = self.calculate_recall_at_k(k, "lenient")
            lenient_results[k] = {'hits': hits, 'recall': recall}
            print(f"  hits@{k:2d}:   {hits:.3f} ({hits*100:.1f}%)")
            print(f"  recall@{k:2d}: {recall:.3f}")
        
        mrr_lenient = self.calculate_mrr("lenient")
        print(f"  MRR:      {mrr_lenient:.3f}")
        
        # Strategy 3: Pruned qrels
        print("\n📊 STRATEGY 3: PRUNED QRELS")
        print("─" * 80)
        print("Requirement: Only top-5 most relevant chunks per query count")
        print()
        
        pruned_results = self.calculate_pruned_qrels_metrics(max_qrels=5)
        print(f"  hits@10:   {pruned_results['hits@10']:.3f} ({pruned_results['hits@10']*100:.1f}%)")
        print(f"  recall@10: {pruned_results['recall@10']:.3f}")
        
        # Comparison summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        print("\nHits@10 comparison:")
        print(f"  Strict (current):     {strict_results[10]['hits']:.3f}")
        print(f"  Lenient (doc-level):  {lenient_results[10]['hits']:.3f}")
        print(f"  Pruned qrels (top-5): {pruned_results['hits@10']:.3f}")
        
        improvement_lenient = (lenient_results[10]['hits'] / strict_results[10]['hits'] - 1) * 100 if strict_results[10]['hits'] > 0 else 0
        improvement_pruned = (pruned_results['hits@10'] / strict_results[10]['hits'] - 1) * 100 if strict_results[10]['hits'] > 0 else 0
        
        print(f"\n  Lenient is {improvement_lenient:.1f}% higher than strict")
        print(f"  Pruned is {improvement_pruned:.1f}% higher than strict")
        
        print("\nRecall@10 comparison:")
        print(f"  Strict (current):     {strict_results[10]['recall']:.3f}")
        print(f"  Lenient (doc-level):  {lenient_results[10]['recall']:.3f}")
        print(f"  Pruned qrels (top-5): {pruned_results['recall@10']:.3f}")
        
        # Diagnosis
        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)
        
        if lenient_results[10]['hits'] > strict_results[10]['hits'] * 1.5:
            print("\n⚠️ FINDING: Lenient matching shows MUCH better results!")
            print(f"   This means your RAG is finding the RIGHT documents but 'wrong' chunks.")
            print(f"   The low metrics are due to overly strict evaluation, not bad retrieval.")
            print(f"\n   RECOMMENDATION: Your RAG system is likely performing better than metrics suggest.")
        
        if pruned_results['hits@10'] > strict_results[10]['hits'] * 2:
            print("\n⚠️ FINDING: Pruned qrels show MUCH better results!")
            print(f"   This means qrel inflation is killing your scores.")
            print(f"   With only 5 relevant docs per query, your system performs much better.")
            print(f"\n   RECOMMENDATION: Regenerate synthetic data with stricter relevance criteria.")
        
        if strict_results[10]['hits'] < 0.7:
            print(f"\n❌ CONCERN: Even with lenient matching, hits@10 = {lenient_results[10]['hits']:.3f}")
            if lenient_results[10]['hits'] < 0.7:
                print(f"   This suggests genuine retrieval issues beyond evaluation methodology.")
            else:
                print(f"   But lenient hits@10 = {lenient_results[10]['hits']:.3f} is reasonable!")
                print(f"   The issue is primarily with evaluation criteria, not retrieval quality.")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        
        if lenient_results[10]['hits'] > 0.8:
            print("\n✅ Your RAG system is performing well!")
            print("   The low official metrics are due to overly strict evaluation criteria.")
            print("\n   Options:")
            print("   1. Use lenient (document-level) matching for monitoring")
            print("   2. Regenerate synthetic data with fewer relevant chunks per query")
            print("   3. Create a hand-curated validation set with realistic expectations")
        elif lenient_results[10]['hits'] > 0.6:
            print("\n⚠️ Your RAG system is performing reasonably but has room for improvement.")
            print("   Consider both improving retrieval AND adjusting evaluation criteria.")
        else:
            print("\n❌ Your RAG system may have genuine retrieval issues.")
            print("   Focus on improving embedding quality, chunk size, and query processing.")
        
        print("\n")


def main():
    """Main entry point."""
    calculator = AlternativeMetricsCalculator()
    calculator.run_comparison()


if __name__ == "__main__":
    main()

