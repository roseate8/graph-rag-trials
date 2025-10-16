"""
Quick Sanity Check

Fast diagnostic that runs in seconds to give you immediate feedback.
Use this first before running the comprehensive analysis.

Usage:
    python quick_check.py
"""

import json
from collections import defaultdict
import statistics


def quick_check():
    """Run a quick sanity check on evaluation metrics."""
    
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 25 + "QUICK SANITY CHECK" + " " * 35 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Load data
    print("Loading data...", end=" ", flush=True)
    
    qrels = defaultdict(dict)
    with open("../output/qrels.tsv", 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                query_id, doc_id, score = parts
                qrels[query_id][doc_id] = int(score)
    
    retrieval_results = {}
    with open("../results/retrieval_results.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                retrieval_results[result['query_id']] = result
    
    print("✓")
    print()
    
    # Check 1: Qrel inflation
    print("=" * 80)
    print("CHECK 1: Qrel Inflation")
    print("=" * 80)
    
    qrel_counts = [len(docs) for docs in qrels.values()]
    avg_qrels = statistics.mean(qrel_counts)
    median_qrels = statistics.median(qrel_counts)
    max_qrels = max(qrel_counts)
    
    print(f"\nAverage relevant docs per query: {avg_qrels:.1f}")
    print(f"Median relevant docs per query:  {median_qrels:.1f}")
    print(f"Maximum relevant docs per query: {max_qrels}")
    
    if avg_qrels > 20:
        print(f"\n❌ PROBLEM: {avg_qrels:.0f} is VERY HIGH!")
        print(f"   This is likely the main cause of low recall scores.")
        print(f"   Even perfect retrieval would only get recall = 1/{avg_qrels:.0f} = {1/avg_qrels:.3f}")
        problem_severity = "CRITICAL"
    elif avg_qrels > 10:
        print(f"\n⚠️ WARNING: {avg_qrels:.0f} is HIGH")
        print(f"   This makes it hard to achieve good recall.")
        problem_severity = "MODERATE"
    else:
        print(f"\n✅ OK: {avg_qrels:.0f} is reasonable")
        problem_severity = "NONE"
    
    # Check 2: Actual vs Expected matches
    print("\n" + "=" * 80)
    print("CHECK 2: Match Analysis (first 50 queries)")
    print("=" * 80)
    
    exact_matches = 0
    partial_matches = 0
    no_matches = 0
    
    sample_size = min(50, len(retrieval_results))
    
    for i, (query_id, result) in enumerate(list(retrieval_results.items())[:sample_size]):
        qrel_ids = set(qrels.get(query_id, {}).keys())
        retrieved_docs = result.get('retrieved_docs', [])[:10]
        retrieved_ids = [doc['chunk_id'] for doc in retrieved_docs]
        
        # Check exact matches
        has_exact = any(rid in qrel_ids for rid in retrieved_ids)
        
        # Check partial matches (same doc, different chunk)
        def get_doc_base(chunk_id):
            if '_chunk_' in chunk_id:
                return chunk_id.split('_chunk_')[0]
            return chunk_id
        
        qrel_doc_bases = {get_doc_base(qid) for qid in qrel_ids}
        retrieved_doc_bases = [get_doc_base(rid) for rid in retrieved_ids]
        has_partial = any(rdb in qrel_doc_bases for rdb in retrieved_doc_bases)
        
        if has_exact:
            exact_matches += 1
        elif has_partial:
            partial_matches += 1
        else:
            no_matches += 1
    
    print(f"\nOut of {sample_size} queries:")
    print(f"  Exact matches:  {exact_matches} ({exact_matches/sample_size*100:.1f}%)")
    print(f"  Partial matches: {partial_matches} ({partial_matches/sample_size*100:.1f}%)")
    print(f"  No matches:     {no_matches} ({no_matches/sample_size*100:.1f}%)")
    
    if partial_matches > exact_matches:
        print(f"\n⚠️ ISSUE: More partial than exact matches!")
        print(f"   Your RAG finds the RIGHT document but 'wrong' chunk.")
        print(f"   This is a false negative - evaluation is too strict.")
        matching_issue = True
    else:
        print(f"\n✅ OK: Matching behavior looks reasonable")
        matching_issue = False
    
    # Check 3: Quick metrics comparison
    print("\n" + "=" * 80)
    print("CHECK 3: Quick Metrics Comparison")
    print("=" * 80)
    
    strict_hits = exact_matches / sample_size
    lenient_hits = (exact_matches + partial_matches) / sample_size
    
    print(f"\nHits@10 (first {sample_size} queries):")
    print(f"  Strict (exact chunk):  {strict_hits:.3f} ({strict_hits*100:.1f}%)")
    print(f"  Lenient (any chunk):   {lenient_hits:.3f} ({lenient_hits*100:.1f}%)")
    
    if lenient_hits > strict_hits * 1.5:
        print(f"\n⚠️ FINDING: Lenient is {lenient_hits/strict_hits:.1f}x better!")
        print(f"   Your RAG is performing better than metrics suggest.")
    
    # Overall diagnosis
    print("\n" + "=" * 80)
    print("OVERALL DIAGNOSIS")
    print("=" * 80)
    print()
    
    if problem_severity == "CRITICAL" and matching_issue:
        print("🔴 CRITICAL: Multiple issues found!")
        print()
        print("   1. Qrel inflation is killing your scores")
        print(f"      → {avg_qrels:.0f} relevant docs per query is too many")
        print()
        print("   2. Evaluation is too strict (chunk-level vs doc-level)")
        print(f"      → {partial_matches} queries find right doc, wrong chunk")
        print()
        print("   Your RAG is likely working MUCH better than the metrics suggest!")
        print()
        print("   RECOMMENDED ACTION:")
        print("   - Regenerate synthetic data with max 5-10 relevant docs per query")
        print("   - Consider using document-level matching instead of chunk-level")
        print()
    
    elif problem_severity == "CRITICAL":
        print("🔴 CRITICAL: Qrel inflation is the main issue")
        print()
        print(f"   With {avg_qrels:.0f} relevant docs per query, even perfect retrieval")
        print(f"   would only achieve recall@10 = {10/avg_qrels:.3f}")
        print()
        print("   RECOMMENDED ACTION:")
        print("   - Review synthetic data generation process")
        print("   - Reduce relevant docs per query to 5-10 maximum")
        print("   - Focus on marking only the MOST essential chunks")
        print()
    
    elif matching_issue:
        print("⚠️ WARNING: Evaluation criteria may be too strict")
        print()
        print("   Your RAG finds the right documents but 'wrong' chunks.")
        print("   This suggests the evaluation is more strict than necessary.")
        print()
        print("   RECOMMENDED ACTION:")
        print("   - Use lenient (document-level) matching for monitoring")
        print("   - Or review if all marked chunks are truly essential")
        print()
    
    elif lenient_hits > 0.8:
        print("✅ GOOD: Your RAG is performing well!")
        print()
        print(f"   Lenient hits@10 = {lenient_hits:.1%} is solid.")
        print("   The lower strict metrics are due to evaluation criteria.")
        print()
        print("   RECOMMENDED ACTION:")
        print("   - Use lenient matching as your primary metric")
        print("   - Or tighten qrels to only most essential chunks")
        print()
    
    elif lenient_hits > 0.6:
        print("⚠️ MODERATE: Room for improvement")
        print()
        print(f"   Lenient hits@10 = {lenient_hits:.1%} is reasonable but not great.")
        print()
        print("   RECOMMENDED ACTION:")
        print("   - Consider improving retrieval quality")
        print("   - Also review evaluation criteria")
        print()
    
    else:
        print("❌ CONCERN: Potential retrieval issues")
        print()
        print(f"   Even with lenient matching, hits@10 = {lenient_hits:.1%}")
        print("   This suggests genuine retrieval quality issues.")
        print()
        print("   RECOMMENDED ACTION:")
        print("   - Run full diagnostic_analyzer.py for detailed analysis")
        print("   - Review embedding quality and chunk size")
        print("   - Check query preprocessing")
        print()
    
    # Next steps
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("For detailed analysis, run:")
    print("  python diagnostic_analyzer.py")
    print()
    print("To inspect specific queries, run:")
    print("  python query_inspector.py --failed 5")
    print()
    print("To calculate alternative metrics, run:")
    print("  python alternative_metrics.py")
    print()


if __name__ == "__main__":
    try:
        quick_check()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find required files")
        print(f"   Make sure you're in the temp_diagnostics folder")
        print(f"   and that evaluation has been run")
        print(f"\n   Missing: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

