# Evaluation Metrics Diagnostics

This folder contains diagnostic tools to help you understand whether your low evaluation metrics reflect genuine retrieval problems or issues with the evaluation methodology itself.

## The Problem

You've observed that:
- Your RAG pipeline answers questions well in practice
- But evaluation metrics are surprisingly low (recall@10 = 0.032, hits@10 = 0.64)
- This mismatch suggests the evaluation may not reflect reality

## What These Tools Do

### 1. `diagnostic_analyzer.py` - Comprehensive Analysis

The main diagnostic tool that analyzes:
- **Qrel inflation**: Are too many documents marked as relevant per query?
- **ID alignment**: Do retrieved chunks match qrel expectations?
- **Query difficulty**: How do different query types perform?
- **Deep dive failures**: What's happening with failed queries?

**Usage:**
```bash
cd evals/synthetic-eval/calculate-metrics/temp_diagnostics
python diagnostic_analyzer.py
```

**What to look for:**
- If average qrels per query > 20: Major qrel inflation problem
- If "same doc, different chunk" rate is high: Evaluation is too strict
- If exact match rate < 30%: Genuine retrieval issues

---

### 2. `query_inspector.py` - Detailed Query Inspection

Interactive tool to inspect individual queries in detail.

**Usage:**
```bash
# Inspect a specific query
python query_inspector.py q0001

# Inspect 5 random queries
python query_inspector.py --random 5

# Inspect 5 failed queries (no top-10 matches)
python query_inspector.py --failed 5

# Inspect 5 successful queries (exact top-1 match)
python query_inspector.py --successful 5
```

**What to look for:**
- Does the retrieved chunk actually contain the answer?
- Are the "relevant" chunks truly necessary to answer the question?
- Are retrieved chunks from the same document as expected chunks?

---

### 3. `alternative_metrics.py` - Metric Comparison

Calculates metrics using different evaluation strategies to isolate the problem.

**Usage:**
```bash
python alternative_metrics.py
```

**Strategies compared:**
1. **Strict** (current): Exact chunk ID must match
2. **Lenient**: Any chunk from the correct document counts
3. **Pruned**: Only top-5 most relevant chunks per query count

**What to look for:**
- If lenient >> strict: Your RAG finds right docs but "wrong" chunks (false negative)
- If pruned >> strict: Qrel inflation is the main problem
- If all strategies are low: Genuine retrieval issues

---

## Diagnostic Workflow

### Step 1: Run Comprehensive Analysis
```bash
python diagnostic_analyzer.py > analysis_report.txt
```

Read the report carefully. Look for:
- ⚠️ **CRITICAL** or ⚠️ **WARNING** markers
- The DIAGNOSIS sections
- The RECOMMENDATIONS section

### Step 2: Inspect Sample Queries
```bash
# Look at some failed queries
python query_inspector.py --failed 3

# Look at some successful queries for comparison
python query_inspector.py --successful 3
```

For each query, ask yourself:
- Could the retrieved chunk answer the question?
- Are all the "relevant" chunks truly necessary?
- Is the evaluation being too strict?

### Step 3: Calculate Alternative Metrics
```bash
python alternative_metrics.py > alternative_metrics_report.txt
```

This shows what your metrics WOULD be with different criteria.

### Step 4: Take Action

Based on the findings:

#### If qrel inflation is the problem (avg > 20 relevant docs):
- Review synthetic data generation code
- Reduce the number of chunks marked as relevant
- Focus on marking only the MOST essential chunks

#### If "right doc, wrong chunk" is common:
- Consider using document-level matching
- Or implement a "passage window" approach
- Or accept that chunk-level matching is too strict

#### If genuine retrieval issues exist:
- Review embedding model quality
- Check chunk size and overlap settings
- Validate query preprocessing
- Test different similarity thresholds

---

## Expected Outcomes

### Good RAG System Signs:
- Lenient hits@10 > 0.8
- "Same doc, different chunk" rate > 30%
- Manual inspection shows retrieved chunks CAN answer questions

### Evaluation Issues Signs:
- Average qrels per query > 20
- Huge gap between strict and lenient metrics
- Retrieved chunks contain answers but don't match exact qrel IDs

### Genuine Retrieval Problems:
- Lenient hits@10 < 0.6
- Retrieved chunks don't contain relevant information
- Even with pruned qrels, metrics are low

---

## Quick Diagnostic Checklist

Run through this checklist:

- [ ] Run `diagnostic_analyzer.py` and check average qrels per query
- [ ] If > 20: Major qrel inflation problem
- [ ] If 10-20: Moderate qrel inflation
- [ ] If < 10: Qrels are reasonable

- [ ] Run `alternative_metrics.py` and compare strict vs lenient
- [ ] If lenient is 50%+ higher: Evaluation is too strict
- [ ] If similar: Evaluation criteria is reasonable

- [ ] Run `query_inspector.py --failed 5`
- [ ] Manually read retrieved chunks for 3-5 failed queries
- [ ] Could they answer the question? If yes → evaluation issue
- [ ] If no → retrieval issue

---

## What's Normal?

For a well-tuned RAG system with reasonable evaluation:
- **hits@10**: 0.8-0.95 (80-95% of queries find something relevant in top-10)
- **recall@10**: 0.4-0.7 (depends on average qrels per query)
- **MRR**: 0.5-0.8 (relevant results appear high in ranking)

Your current metrics (hits@10=0.64, recall@10=0.032, MRR=0.31) suggest either:
1. Overly strict evaluation (most likely based on the qrel data)
2. Genuine retrieval issues (less likely if manual tests work well)

These tools help you determine which it is!

---

## Next Steps After Diagnosis

### If Evaluation is the Problem:
1. Regenerate synthetic data with stricter relevance criteria
2. Use lenient matching for ongoing monitoring
3. Create a small hand-curated validation set (20-50 queries)
4. Document what "good" looks like for your use case

### If Retrieval is the Problem:
1. Analyze embedding quality (are similar concepts close?)
2. Review chunk size (too large? too small?)
3. Test query preprocessing (does it help or hurt?)
4. Consider hybrid retrieval (keyword + semantic)
5. Tune similarity thresholds

### For Ongoing Monitoring:
1. Track both strict AND lenient metrics
2. Use hand-curated validation set as north star
3. Monitor MRR (more stable than recall)
4. Watch for distribution shift over time

---

## Files in This Folder

- `diagnostic_analyzer.py` - Main comprehensive diagnostic tool
- `query_inspector.py` - Interactive query inspection
- `alternative_metrics.py` - Calculate metrics with different strategies
- `README.md` - This file

All tools read from the standard evaluation output files:
- `../output/queries.jsonl`
- `../output/qrels.tsv`
- `../output/corpus.jsonl`
- `../results/retrieval_results.jsonl`

No configuration needed - just run them!

