"""
Comprehensive Diagnostic Tool for Evaluation Metrics

This tool helps verify if the evaluation metrics are correct or if there are issues with:
1. Qrel inflation (too many relevant documents per query)
2. ID mismatches between retrieved chunks and qrels
3. Different data distributions (synthetic vs production)
4. Multi-hop evaluation issues

Usage:
    python diagnostic_analyzer.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics


class EvaluationDiagnostics:
    """Comprehensive diagnostics for evaluation metrics."""
    
    def __init__(self, 
                 queries_file: str = "../../output/queries.jsonl",
                 qrels_file: str = "../../output/qrels.tsv",
                 retrieval_results_file: str = "../results/retrieval_results.jsonl",
                 corpus_file: str = "../../output/corpus.jsonl"):
        self.queries_file = queries_file
        self.qrels_file = qrels_file
        self.retrieval_results_file = retrieval_results_file
        self.corpus_file = corpus_file
        
        self.queries = {}
        self.qrels = defaultdict(dict)
        self.retrieval_results = {}
        self.corpus = {}
        
    def load_data(self):
        """Load all data files."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Load queries
        print(f"\n[1/4] Loading queries from {self.queries_file}...")
        with open(self.queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    self.queries[q['_id']] = q
        print(f"  ✓ Loaded {len(self.queries)} queries")
        
        # Load qrels
        print(f"\n[2/4] Loading qrels from {self.qrels_file}...")
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query_id, doc_id, score = parts
                    self.qrels[query_id][doc_id] = int(score)
        print(f"  ✓ Loaded qrels for {len(self.qrels)} queries")
        
        # Load retrieval results
        print(f"\n[3/4] Loading retrieval results from {self.retrieval_results_file}...")
        with open(self.retrieval_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    self.retrieval_results[result['query_id']] = result
        print(f"  ✓ Loaded {len(self.retrieval_results)} retrieval results")
        
        # Load corpus (optional)
        print(f"\n[4/4] Loading corpus from {self.corpus_file}...")
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        self.corpus[doc['_id']] = doc
            print(f"  ✓ Loaded {len(self.corpus)} corpus documents")
        except Exception as e:
            print(f"  ⚠ Could not load corpus (non-critical): {e}")
    
    def analyze_qrel_inflation(self):
        """Analyze if qrels have too many relevant documents per query."""
        print("\n" + "=" * 80)
        print("QREL INFLATION ANALYSIS")
        print("=" * 80)
        
        qrel_counts = []
        qrel_scores = defaultdict(list)
        
        for query_id, relevance_dict in self.qrels.items():
            count = len(relevance_dict)
            qrel_counts.append(count)
            
            # Count by score
            for doc_id, score in relevance_dict.items():
                qrel_scores[score].append(query_id)
        
        print(f"\n📊 Qrel Statistics:")
        print(f"  Total queries with qrels: {len(self.qrels)}")
        print(f"  Total relevance judgments: {sum(qrel_counts)}")
        print(f"  Mean relevant docs per query: {statistics.mean(qrel_counts):.2f}")
        print(f"  Median relevant docs per query: {statistics.median(qrel_counts):.2f}")
        print(f"  Min relevant docs per query: {min(qrel_counts)}")
        print(f"  Max relevant docs per query: {max(qrel_counts)}")
        print(f"  Std dev: {statistics.stdev(qrel_counts):.2f}")
        
        print(f"\n📊 Distribution by Relevance Score:")
        for score in sorted(qrel_scores.keys()):
            count = len(qrel_scores[score])
            print(f"  Score {score}: {count} queries")
        
        # Show percentiles
        sorted_counts = sorted(qrel_counts)
        p25 = sorted_counts[len(sorted_counts) // 4]
        p50 = sorted_counts[len(sorted_counts) // 2]
        p75 = sorted_counts[3 * len(sorted_counts) // 4]
        p90 = sorted_counts[9 * len(sorted_counts) // 10]
        p95 = sorted_counts[19 * len(sorted_counts) // 20]
        
        print(f"\n📊 Percentiles:")
        print(f"  25th percentile: {p25} docs")
        print(f"  50th percentile: {p50} docs")
        print(f"  75th percentile: {p75} docs")
        print(f"  90th percentile: {p90} docs")
        print(f"  95th percentile: {p95} docs")
        
        # Diagnosis
        print(f"\n🔍 DIAGNOSIS:")
        avg_count = statistics.mean(qrel_counts)
        if avg_count > 20:
            print(f"  ⚠️ CRITICAL: Average {avg_count:.1f} relevant docs per query is VERY HIGH!")
            print(f"     This is likely causing the low recall scores.")
            print(f"     Even if you retrieve the perfect chunk, recall = 1/{avg_count:.0f} = {1/avg_count:.3f}")
            print(f"     This suggests qrel inflation - too many docs marked as relevant.")
        elif avg_count > 10:
            print(f"  ⚠️ WARNING: Average {avg_count:.1f} relevant docs per query is HIGH.")
            print(f"     This makes it hard to achieve high recall.")
            print(f"     Consider reviewing if all marked docs are truly essential.")
        else:
            print(f"  ✅ OK: Average {avg_count:.1f} relevant docs per query is reasonable.")
        
        return qrel_counts
    
    def analyze_id_alignment(self, sample_size: int = 10):
        """Check if retrieved IDs match qrel IDs format."""
        print("\n" + "=" * 80)
        print("ID ALIGNMENT ANALYSIS")
        print("=" * 80)
        
        print(f"\n🔍 Sampling {sample_size} queries to check ID format consistency...")
        
        sample_queries = list(self.retrieval_results.keys())[:sample_size]
        
        mismatches = []
        partial_matches = []
        exact_matches = []
        
        for query_id in sample_queries:
            result = self.retrieval_results[query_id]
            qrel_ids = set(self.qrels.get(query_id, {}).keys())
            retrieved_ids = [doc['chunk_id'] for doc in result['retrieved_docs'][:10]]
            
            print(f"\n{'─' * 80}")
            print(f"Query: {query_id}")
            print(f"Text: {result['query_text'][:100]}...")
            print(f"\nQrels ({len(qrel_ids)} relevant docs):")
            print(f"  Sample: {list(qrel_ids)[:5]}")
            print(f"\nRetrieved (top 10):")
            for i, doc_id in enumerate(retrieved_ids[:5], 1):
                is_match = doc_id in qrel_ids
                marker = "✅" if is_match else "❌"
                print(f"  {i}. {marker} {doc_id}")
            
            # Check for exact matches
            exact_match_count = sum(1 for rid in retrieved_ids if rid in qrel_ids)
            
            # Check for partial matches (same doc, different chunk)
            def get_doc_base(chunk_id):
                # Extract base document ID (before _chunk_)
                if '_chunk_' in chunk_id:
                    return chunk_id.split('_chunk_')[0]
                return chunk_id
            
            qrel_doc_bases = {get_doc_base(qid) for qid in qrel_ids}
            retrieved_doc_bases = [get_doc_base(rid) for rid in retrieved_ids]
            partial_match_count = sum(1 for rdb in retrieved_doc_bases if rdb in qrel_doc_bases)
            
            print(f"\nMatches:")
            print(f"  Exact chunk matches: {exact_match_count}/10")
            print(f"  Same document (any chunk): {partial_match_count}/10")
            
            if exact_match_count == 0:
                if partial_match_count > 0:
                    partial_matches.append((query_id, partial_match_count))
                    print(f"  ⚠️ ISSUE: No exact matches, but {partial_match_count} same-doc matches")
                    print(f"     This suggests the RIGHT doc but WRONG chunk!")
                else:
                    mismatches.append(query_id)
                    print(f"  ❌ PROBLEM: No matches at all!")
            else:
                exact_matches.append((query_id, exact_match_count))
        
        # Summary
        print(f"\n{'=' * 80}")
        print(f"SUMMARY:")
        print(f"  Queries with exact matches: {len(exact_matches)}/{sample_size}")
        print(f"  Queries with partial matches (right doc, wrong chunk): {len(partial_matches)}/{sample_size}")
        print(f"  Queries with no matches: {len(mismatches)}/{sample_size}")
        
        print(f"\n🔍 DIAGNOSIS:")
        if len(partial_matches) > len(exact_matches):
            print(f"  ⚠️ CRITICAL: Most queries retrieve the RIGHT document but WRONG chunk!")
            print(f"     Your RAG system is working correctly, but evaluation is too strict.")
            print(f"     The qrels expect a specific chunk, but any chunk from the doc may be valid.")
            print(f"     This explains why your RAG works in practice but scores poorly.")
        elif len(mismatches) > sample_size // 2:
            print(f"  ❌ PROBLEM: Many queries retrieve completely wrong documents.")
            print(f"     This suggests a genuine retrieval issue.")
        else:
            print(f"  ✅ REASONABLE: Most queries have some matches.")
    
    def analyze_query_difficulty(self):
        """Analyze query characteristics that might affect performance."""
        print("\n" + "=" * 80)
        print("QUERY DIFFICULTY ANALYSIS")
        print("=" * 80)
        
        single_hop = []
        multi_hop = []
        
        for query_id, query in self.queries.items():
            qtype = query['metadata'].get('query_type', 'unknown')
            if qtype == 'single_hop':
                single_hop.append(query_id)
            elif qtype == 'multi_hop':
                multi_hop.append(query_id)
        
        print(f"\n📊 Query Type Distribution:")
        print(f"  Single-hop queries: {len(single_hop)}")
        print(f"  Multi-hop queries: {len(multi_hop)}")
        print(f"  Other: {len(self.queries) - len(single_hop) - len(multi_hop)}")
        
        # Analyze hits by type
        print(f"\n📊 Success Rate by Type (top-10 contains any qrel):")
        
        for qtype, query_list in [('single_hop', single_hop), ('multi_hop', multi_hop)]:
            if not query_list:
                continue
                
            success_count = 0
            for query_id in query_list:
                if query_id not in self.retrieval_results:
                    continue
                    
                result = self.retrieval_results[query_id]
                qrel_ids = set(self.qrels.get(query_id, {}).keys())
                retrieved_ids = [doc['chunk_id'] for doc in result['retrieved_docs'][:10]]
                
                if any(rid in qrel_ids for rid in retrieved_ids):
                    success_count += 1
            
            success_rate = success_count / len(query_list) if query_list else 0
            print(f"  {qtype}: {success_count}/{len(query_list)} ({success_rate:.1%})")
    
    def deep_dive_failures(self, num_failures: int = 5):
        """Deep dive into specific failed queries."""
        print("\n" + "=" * 80)
        print(f"DEEP DIVE: {num_failures} FAILED QUERIES")
        print("=" * 80)
        
        print(f"\nAnalyzing queries where top-10 contains NO qrel matches...")
        
        failures = []
        for query_id, result in self.retrieval_results.items():
            qrel_ids = set(self.qrels.get(query_id, {}).keys())
            retrieved_ids = [doc['chunk_id'] for doc in result['retrieved_docs'][:10]]
            
            # Check if any exact match
            if not any(rid in qrel_ids for rid in retrieved_ids):
                failures.append(query_id)
        
        print(f"\nFound {len(failures)} queries with no top-10 matches")
        
        for i, query_id in enumerate(failures[:num_failures], 1):
            query = self.queries[query_id]
            result = self.retrieval_results[query_id]
            qrel_ids = list(self.qrels.get(query_id, {}).keys())
            
            print(f"\n{'─' * 80}")
            print(f"[FAILURE {i}] Query: {query_id}")
            print(f"Type: {query['metadata'].get('query_type')}")
            print(f"Question: {query['text']}")
            print(f"\nExpected Answer: {query['metadata'].get('answer', 'N/A')}")
            print(f"\nExpected Chunks ({len(qrel_ids)} total):")
            for j, chunk_id in enumerate(qrel_ids[:3], 1):
                print(f"  {j}. {chunk_id}")
                if self.corpus and chunk_id in self.corpus:
                    chunk = self.corpus[chunk_id]
                    print(f"     Text preview: {chunk.get('text', '')[:150]}...")
            
            print(f"\nActually Retrieved (top 5):")
            for j, doc in enumerate(result['retrieved_docs'][:5], 1):
                chunk_id = doc['chunk_id']
                score = doc['score']
                print(f"  {j}. {chunk_id} (score: {score:.3f})")
                if self.corpus and chunk_id in self.corpus:
                    chunk = self.corpus[chunk_id]
                    print(f"     Text preview: {chunk.get('text', '')[:150]}...")
    
    def generate_recommendations(self):
        """Generate actionable recommendations."""
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        
        # Calculate key metrics
        avg_qrel_count = statistics.mean([len(docs) for docs in self.qrels.values()])
        
        # Check partial match rate
        partial_match_count = 0
        exact_match_count = 0
        total_queries = 0
        
        for query_id, result in list(self.retrieval_results.items())[:50]:
            total_queries += 1
            qrel_ids = set(self.qrels.get(query_id, {}).keys())
            retrieved_ids = [doc['chunk_id'] for doc in result['retrieved_docs'][:10]]
            
            def get_doc_base(chunk_id):
                if '_chunk_' in chunk_id:
                    return chunk_id.split('_chunk_')[0]
                return chunk_id
            
            qrel_doc_bases = {get_doc_base(qid) for qid in qrel_ids}
            retrieved_doc_bases = [get_doc_base(rid) for rid in retrieved_ids]
            
            has_exact = any(rid in qrel_ids for rid in retrieved_ids)
            has_partial = any(rdb in qrel_doc_bases for rdb in retrieved_doc_bases)
            
            if has_exact:
                exact_match_count += 1
            elif has_partial:
                partial_match_count += 1
        
        partial_match_rate = partial_match_count / total_queries if total_queries > 0 else 0
        exact_match_rate = exact_match_count / total_queries if total_queries > 0 else 0
        
        print(f"\n🎯 ACTIONABLE RECOMMENDATIONS:\n")
        
        if avg_qrel_count > 20:
            print(f"1. REDUCE QREL INFLATION (CRITICAL)")
            print(f"   Problem: Each query has {avg_qrel_count:.0f} relevant docs on average")
            print(f"   Action: Review synthetic data generation - only mark truly essential chunks")
            print(f"   Impact: This alone could 5-10x your recall scores\n")
        
        if partial_match_rate > 0.3:
            print(f"2. IMPLEMENT LENIENT MATCHING (RECOMMENDED)")
            print(f"   Problem: {partial_match_rate:.0%} of queries get right doc, wrong chunk")
            print(f"   Action: Consider doc-level matching instead of chunk-level")
            print(f"   Impact: Would significantly improve metrics without changing retrieval\n")
        
        if exact_match_rate < 0.3:
            print(f"3. REVIEW RETRIEVAL QUALITY")
            print(f"   Problem: Only {exact_match_rate:.0%} of queries find exact matches")
            print(f"   Action: Investigate embedding quality, chunk size, or query preprocessing")
            print(f"   Impact: Core retrieval improvement needed\n")
        
        print(f"4. VALIDATE SYNTHETIC DATA QUALITY")
        print(f"   Action: Manually review 10-20 queries and their qrels")
        print(f"   Check: Are the 'relevant' chunks actually necessary to answer?")
        print(f"   Impact: Ensures eval reflects real user experience\n")
        
        print(f"5. CREATE PRODUCTION VALIDATION SET")
        print(f"   Action: Hand-curate 20-50 real user questions with single gold answers")
        print(f"   Use: This becomes your true north star metric")
        print(f"   Impact: Confidence that metrics reflect actual performance\n")
    
    def run_full_analysis(self):
        """Run complete diagnostic analysis."""
        print("\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "EVALUATION METRICS DIAGNOSTICS" + " " * 28 + "║")
        print("╚" + "═" * 78 + "╝")
        
        self.load_data()
        self.analyze_qrel_inflation()
        self.analyze_id_alignment(sample_size=10)
        self.analyze_query_difficulty()
        self.deep_dive_failures(num_failures=5)
        self.generate_recommendations()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print("\nNext steps:")
        print("1. Review the recommendations above")
        print("2. Manually inspect a few queries and their qrels")
        print("3. Consider re-generating synthetic data with stricter relevance criteria")
        print("4. Create a small hand-curated validation set")
        print("\n")


def main():
    """Main entry point."""
    diagnostics = EvaluationDiagnostics()
    diagnostics.run_full_analysis()


if __name__ == "__main__":
    main()

