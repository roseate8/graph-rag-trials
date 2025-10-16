"""
Interactive Query Inspector

Allows detailed inspection of individual queries to understand:
- What the query is asking
- What chunks are marked as relevant (and why)
- What chunks were actually retrieved
- Whether the retrieved chunks could answer the question

Usage:
    python query_inspector.py q0001
    python query_inspector.py --random 5
    python query_inspector.py --failed 5
"""

import json
import sys
import random
from pathlib import Path
from typing import Dict, List


class QueryInspector:
    """Detailed inspector for individual queries."""
    
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
        self.qrels = {}
        self.retrieval_results = {}
        self.corpus = {}
        
        self.load_data()
    
    def load_data(self):
        """Load all data files."""
        print("Loading data...")
        
        # Load queries
        with open(self.queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    self.queries[q['_id']] = q
        
        # Load qrels
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query_id, doc_id, score = parts
                    if query_id not in self.qrels:
                        self.qrels[query_id] = {}
                    self.qrels[query_id][doc_id] = int(score)
        
        # Load retrieval results
        with open(self.retrieval_results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    self.retrieval_results[result['query_id']] = result
        
        # Load corpus
        try:
            with open(self.corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        self.corpus[doc['_id']] = doc
        except:
            print("Warning: Could not load corpus")
        
        print(f"✓ Loaded {len(self.queries)} queries, {len(self.corpus)} corpus docs")
    
    def inspect_query(self, query_id: str):
        """Inspect a single query in detail."""
        if query_id not in self.queries:
            print(f"❌ Query {query_id} not found!")
            return
        
        query = self.queries[query_id]
        qrels = self.qrels.get(query_id, {})
        result = self.retrieval_results.get(query_id, {})
        
        print("\n" + "=" * 100)
        print(f"QUERY INSPECTION: {query_id}")
        print("=" * 100)
        
        # Query details
        print(f"\n📋 QUERY DETAILS:")
        print(f"  ID: {query_id}")
        print(f"  Type: {query['metadata'].get('query_type', 'unknown')}")
        print(f"  Style: {query['metadata'].get('question_style', 'unknown')}")
        print(f"\n  Question:")
        print(f"  {query['text']}")
        print(f"\n  Expected Answer:")
        print(f"  {query['metadata'].get('answer', 'N/A')}")
        
        # Qrels analysis
        print(f"\n" + "─" * 100)
        print(f"📚 EXPECTED RELEVANT CHUNKS ({len(qrels)} total):")
        
        if len(qrels) > 20:
            print(f"  ⚠️ WARNING: {len(qrels)} relevant chunks is very high!")
            print(f"     This makes it nearly impossible to achieve good recall.")
        
        # Group by document
        doc_groups = {}
        for chunk_id, score in qrels.items():
            doc_base = chunk_id.split('_chunk_')[0] if '_chunk_' in chunk_id else chunk_id
            if doc_base not in doc_groups:
                doc_groups[doc_base] = []
            doc_groups[doc_base].append((chunk_id, score))
        
        print(f"\n  Relevant chunks span {len(doc_groups)} different documents:")
        for doc_base, chunks in list(doc_groups.items())[:5]:
            print(f"\n  Document: {doc_base}")
            print(f"    Chunks: {len(chunks)}")
            for chunk_id, score in sorted(chunks, key=lambda x: x[1], reverse=True)[:3]:
                print(f"      - {chunk_id} (score: {score})")
                if chunk_id in self.corpus:
                    text = self.corpus[chunk_id].get('text', '')[:200]
                    print(f"        Preview: {text}...")
        
        if len(doc_groups) > 5:
            print(f"\n  ... and {len(doc_groups) - 5} more documents")
        
        # Retrieval results analysis
        if not result:
            print(f"\n❌ No retrieval results found for this query!")
            return
        
        print(f"\n" + "─" * 100)
        print(f"🔍 RETRIEVED CHUNKS (top 10):")
        
        retrieved_docs = result.get('retrieved_docs', [])[:10]
        
        exact_matches = 0
        partial_matches = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            chunk_id = doc['chunk_id']
            score = doc.get('score', 0)
            
            # Check if it's an exact match
            is_exact_match = chunk_id in qrels
            
            # Check if it's a partial match (same doc, different chunk)
            doc_base = chunk_id.split('_chunk_')[0] if '_chunk_' in chunk_id else chunk_id
            is_partial_match = doc_base in doc_groups and not is_exact_match
            
            if is_exact_match:
                marker = "✅ EXACT MATCH"
                exact_matches += 1
            elif is_partial_match:
                marker = "⚠️ SAME DOC, DIFF CHUNK"
                partial_matches += 1
            else:
                marker = "❌ NO MATCH"
            
            print(f"\n  Rank {i}: {marker}")
            print(f"    Chunk ID: {chunk_id}")
            print(f"    Score: {score:.4f}")
            
            if chunk_id in self.corpus:
                text = self.corpus[chunk_id].get('text', '')
                print(f"    Text preview: {text[:300]}...")
                print(f"    Full length: {len(text)} chars")
            else:
                print(f"    (Chunk content not in corpus)")
        
        # Summary
        print(f"\n" + "─" * 100)
        print(f"📊 MATCH SUMMARY:")
        print(f"  Exact matches in top-10: {exact_matches}/10")
        print(f"  Same doc, diff chunk: {partial_matches}/10")
        print(f"  No matches: {10 - exact_matches - partial_matches}/10")
        
        # Diagnosis
        print(f"\n🔍 DIAGNOSIS:")
        if exact_matches > 0:
            print(f"  ✅ GOOD: Found {exact_matches} exact matches")
            print(f"     The retrieval system is working correctly for this query.")
        elif partial_matches > 0:
            print(f"  ⚠️ PARTIAL SUCCESS: Found {partial_matches} chunks from the right document(s)")
            print(f"     The retrieval found the correct document but a different chunk.")
            print(f"     This might still answer the question correctly!")
            print(f"     Consider: Does the retrieved chunk contain the answer?")
        else:
            print(f"  ❌ FAILURE: No matches found in top-10")
            print(f"     The retrieval system failed to find relevant content.")
        
        # Could the retrieved chunks answer the question?
        print(f"\n💡 MANUAL VALIDATION:")
        print(f"  Question: {query['text']}")
        print(f"  Expected answer: {query['metadata'].get('answer', 'N/A')}")
        print(f"\n  Top retrieved chunk content:")
        if retrieved_docs and retrieved_docs[0]['chunk_id'] in self.corpus:
            top_chunk = self.corpus[retrieved_docs[0]['chunk_id']]
            print(f"  {top_chunk.get('text', '')[:500]}...")
            print(f"\n  ❓ Could this chunk answer the question? (Manual review needed)")
        
        print("\n" + "=" * 100 + "\n")
    
    def inspect_random(self, count: int = 5):
        """Inspect random queries."""
        query_ids = random.sample(list(self.queries.keys()), min(count, len(self.queries)))
        for query_id in query_ids:
            self.inspect_query(query_id)
            input("\nPress Enter to see next query...")
    
    def inspect_failed(self, count: int = 5):
        """Inspect queries that had no matches in top-10."""
        print("Finding failed queries...")
        
        failed = []
        for query_id, result in self.retrieval_results.items():
            qrels = set(self.qrels.get(query_id, {}).keys())
            retrieved_ids = [doc['chunk_id'] for doc in result.get('retrieved_docs', [])[:10]]
            
            if not any(rid in qrels for rid in retrieved_ids):
                failed.append(query_id)
        
        print(f"Found {len(failed)} failed queries")
        
        if not failed:
            print("No failed queries found!")
            return
        
        selected = random.sample(failed, min(count, len(failed)))
        for query_id in selected:
            self.inspect_query(query_id)
            input("\nPress Enter to see next query...")
    
    def inspect_successful(self, count: int = 5):
        """Inspect queries that had matches in top-1."""
        print("Finding successful queries...")
        
        successful = []
        for query_id, result in self.retrieval_results.items():
            qrels = set(self.qrels.get(query_id, {}).keys())
            retrieved_docs = result.get('retrieved_docs', [])
            
            if retrieved_docs and retrieved_docs[0]['chunk_id'] in qrels:
                successful.append(query_id)
        
        print(f"Found {len(successful)} successful queries")
        
        if not successful:
            print("No successful queries found!")
            return
        
        selected = random.sample(successful, min(count, len(successful)))
        for query_id in selected:
            self.inspect_query(query_id)
            input("\nPress Enter to see next query...")


def main():
    """Main entry point."""
    inspector = QueryInspector()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python query_inspector.py <query_id>          # Inspect specific query")
        print("  python query_inspector.py --random 5          # Inspect 5 random queries")
        print("  python query_inspector.py --failed 5          # Inspect 5 failed queries")
        print("  python query_inspector.py --successful 5      # Inspect 5 successful queries")
        return
    
    arg = sys.argv[1]
    
    if arg == '--random':
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        inspector.inspect_random(count)
    elif arg == '--failed':
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        inspector.inspect_failed(count)
    elif arg == '--successful':
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        inspector.inspect_successful(count)
    else:
        # Assume it's a query ID
        inspector.inspect_query(arg)


if __name__ == "__main__":
    main()

