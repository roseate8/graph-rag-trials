# Fix Options Analysis - Qrel Inflation Problem

## 📋 Executive Summary

**Problem:** Average 62 relevant chunks per query (should be 5-10)  
**Root Cause:** Same-document bonus in `silver_labeler.py` marks all chunks from the gold chunk's document as relevant  
**Impact:** Recall@10 = 3.2% is mathematically inevitable; actual RAG performance (77.5% hits@10) is hidden

**Decision Needed:** How to define "relevant" for evaluation?

---

## 🎯 Core Question: What Does "Relevant" Mean?

Before choosing a fix, we need to decide the philosophy:

### Philosophy A: Answer-Centric (Strict)
**Principle:** Only chunks that contain or directly support the answer are relevant

**Pros:**
- Most objective and reproducible
- Focuses on what actually helps answer the question
- No arbitrary boundaries (document, position, etc.)
- Makes evaluation dataset portable (works on different chunking strategies)

**Cons:**
- May miss contextual chunks that help humans understand
- Could under-count chunks that provide necessary background

**Best for:** Production monitoring, comparing retrieval systems

---

### Philosophy B: Context-Aware (Moderate)
**Principle:** Chunks near the answer or in the same section are relevant

**Pros:**
- Recognizes that RAG needs context, not just the answer
- More forgiving of boundary effects
- Reflects how humans actually consume information

**Cons:**
- Requires defining "near" (±2 chunks? ±5? same section?)
- Dependent on chunking strategy
- Less portable across different document structures

**Best for:** Evaluating complete RAG pipelines (retrieval + generation)

---

### Philosophy C: Document-Aware (Lenient)
**Principle:** Any chunk from the right document is relevant (current approach)

**Pros:**
- Very forgiving
- Works well when documents are small/focused

**Cons:**
- **Breaks down with large documents** (10-K forms with 300+ chunks)
- Creates huge qrel inflation
- Makes metrics uninterpretable

**Best for:** Small, focused documents (blog posts, articles) - NOT financial reports

---

## 🔍 Deep Dive: Why Current Approach Fails

### The Multi-Document Problem

Your corpus contains:
- **Short documents:** Press releases (5-10 chunks)
- **Medium documents:** Presentations (20-40 chunks)  
- **Long documents:** 10-K forms (100-300 chunks)

**Current logic:**
```python
if chunk_doc_id in gold_doc_ids:
    return 1 or 2  # All chunks from same doc are relevant
```

**Result by document type:**
- Press release query → 5-10 relevant chunks ✅ Reasonable
- Presentation query → 20-40 relevant chunks ⚠️ High but manageable
- 10-K form query → 100-150 relevant chunks ❌ Impossible to achieve good recall

### Real Example Analysis

Let me check a few actual cases from your data:

**Query q0016 (Failed - 0/10 matches):**
- Question: "How are our consumption-based plans for Elastic Cloud expected to change over time?"
- Expected Answer: "continue to increase"
- **60 relevant chunks** marked (all from Form 10-K 2022)
- Retrieved: Form 10-K 2025_chunk_71, Form 10-K 2022_chunk_101
- **Reality:** Retrieved more recent data (2025 vs 2022)! This might be BETTER than the gold answer!

**Query q0112 (Failed - 0/10 matches):**
- Question: "The new platform announced by Elastic for enhancing RAG is called ___"
- Expected Answer: "Elastic AI Ecosystem"
- **45 relevant chunks** marked
- Retrieved: Form 10-K 2025_chunk_35 (most recent), Second Quarter Fiscal 2024 Financial Results
- **Reality:** Retrieved recent announcements, but doesn't match gold chunks from Q2 FY2025

**Query q0067 (Failed - 0/10 matches):**
- Question: "What types of losses can occur if a company's security is breached?"
- Expected Answer: "loss of intellectual property, data, or customers' data"
- **54 relevant chunks** marked (dutch-board-report-fiscal-year-2024-1)
- Retrieved: Form 10-K 2022_chunk_87, Form 10-K 2025_chunk_57
- **Reading the retrieved text:** "If our security measures are breached, a security incident occurs, or unauthorized access to or other processing of confidential information..."
- **This DOES answer the question!** But from Form 10-K 2022, not dutch-board-report-2024

### Key Insight

Many "failures" are actually:
1. **Different year, same answer:** Retrieves 2025 data when gold is 2022
2. **Different document, same content:** Retrieves 10-K when gold is board report
3. **Better/more recent answer:** Newer data might be more relevant!

The evaluation is penalizing the RAG for being smart about recency and source quality!

---

## 🛠️ Fix Options Detailed Analysis

### Option 1: Pure Answer-Based (Remove Same-Doc Bonus)

**Implementation:**
```python
def _compute_relevance(self, query, chunk, chunk_to_doc):
    content = chunk.get('content', '')
    
    # 1. Exact match → rel=3
    if has_exact_match(query.answer, content):
        return 3
    
    # 2. High token-F1 with answer → rel=3
    f1_score = compute_token_f1(query.answer, content)
    if f1_score >= 0.7:
        return 3
    
    # 3. Mid token-F1 → rel=2
    if f1_score >= 0.4:
        return 2
    
    # 4. Low F1 + LLM judge → 0-3
    if self.config.enable_llm_judge and f1_score >= 0.3:
        return self._llm_judge(query, chunk)
    
    # 5. Default → not relevant
    return 0
```

**Predicted Impact:**
- Average relevant chunks: **3-7 per query** (measured by token-F1)
- Recall@10: **15-25%** (realistic)
- Precision@10: **15-25%** (same magnitude as recall)
- Hits@10: **65-75%** (slight improvement over current 64%)

**Pros:**
- ✅ Completely objective (based on answer presence)
- ✅ Works across all document types
- ✅ Portable (doesn't depend on chunking strategy)
- ✅ Makes metrics interpretable
- ✅ Focuses on actual answer retrieval

**Cons:**
- ⚠️ May mark only 1-2 chunks per query as rel=3 (very strict)
- ⚠️ Ignores contextual chunks that provide background
- ⚠️ Doesn't account for multi-hop queries needing multiple chunks

**Best For:** Single-hop queries, objective benchmarking

---

### Option 2: Adjacent Chunks Window

**Implementation:**
```python
def _compute_relevance(self, query, chunk, chunk_to_doc):
    content = chunk.get('content', '')
    chunk_id = chunk.get('chunk_id', '')
    
    # 1-3. Same as Option 1 (exact match, high F1, mid F1)
    # ... [same logic] ...
    
    # 4. Check if within window of gold chunks
    chunk_doc_id = chunk_to_doc.get(chunk_id, '')
    
    if chunk_doc_id:
        for gold_chunk_id in query.gold_chunk_ids:
            if self._is_adjacent_chunk(chunk_id, gold_chunk_id, window=3):
                # Within ±3 chunks of gold
                semantic_sim = self._compute_semantic_similarity(query.query_text, content)
                if semantic_sim >= 0.75:
                    return 2
                elif semantic_sim >= 0.60:
                    return 1
    
    return 0

def _is_adjacent_chunk(self, chunk_id, gold_chunk_id, window=3):
    """Check if chunks are within window positions."""
    # Extract chunk numbers
    # Form_10-K_2022_chunk_83 → extract 83
    try:
        base1, num1 = chunk_id.rsplit('_chunk_', 1)
        base2, num2 = gold_chunk_id.rsplit('_chunk_', 1)
        
        if base1 == base2:  # Same document
            return abs(int(num1) - int(num2)) <= window
    except:
        pass
    return False
```

**Configuration:**
```python
# In config.py
adjacent_chunk_window: int = 3  # ±3 chunks from gold
adjacent_chunk_semantic_threshold_high: float = 0.75  # For rel=2
adjacent_chunk_semantic_threshold_low: float = 0.60   # For rel=1
```

**Predicted Impact:**
- Average relevant chunks: **8-15 per query** (gold + ±3 adjacent)
- Recall@10: **20-35%**
- Precision@10: **20-30%**
- Hits@10: **70-80%**

**Pros:**
- ✅ Recognizes that context matters
- ✅ Limited scope (±3 chunks ≈ 1-2K tokens context)
- ✅ Still objective (fixed window size)
- ✅ Works for multi-chunk answers

**Cons:**
- ⚠️ Arbitrary window size (why 3? why not 2 or 5?)
- ⚠️ Depends on chunking strategy (different chunk sizes = different context)
- ⚠️ Still adds 6-10 chunks per gold chunk

**Best For:** RAG systems that use context windows, multi-chunk answers

---

### Option 3: Document-Size-Aware

**Implementation:**
```python
def _compute_relevance(self, query, chunk, chunk_to_doc):
    content = chunk.get('content', '')
    chunk_id = chunk.get('chunk_id', '')
    
    # 1-3. Same as Option 1 (exact match, high F1, mid F1)
    # ... [same logic] ...
    
    # 4. Check if same document (but adapt based on doc size)
    chunk_doc_id = chunk_to_doc.get(chunk_id, '')
    
    if chunk_doc_id in gold_doc_ids:
        # Count total chunks in this document
        doc_chunk_count = self._get_doc_chunk_count(chunk_doc_id)
        
        if doc_chunk_count <= 20:
            # Small doc: all chunks might be relevant
            semantic_sim = self._compute_semantic_similarity(query.query_text, content)
            if semantic_sim >= 0.75:
                return 2
            elif semantic_sim >= 0.60:
                return 1
        
        elif doc_chunk_count <= 50:
            # Medium doc: only high semantic similarity
            semantic_sim = self._compute_semantic_similarity(query.query_text, content)
            if semantic_sim >= 0.80:
                return 2
        
        # Large doc (>50 chunks): only answer-based relevance (no bonus)
    
    return 0
```

**Predicted Impact:**
- Average relevant chunks: **10-20 per query** (varies by document type)
- Small docs: Most chunks marked relevant
- Large docs: Only answer-containing chunks
- Recall@10: **20-30%**

**Pros:**
- ✅ Adapts to document structure
- ✅ Works well across document types
- ✅ Prevents runaway inflation on large docs

**Cons:**
- ⚠️ Complex logic (harder to understand)
- ⚠️ Arbitrary thresholds (20 chunks? 50 chunks?)
- ⚠️ Different treatment for different documents (less consistent)
- ⚠️ Small docs still get inflated qrels

**Best For:** Mixed-size document collections (like yours)

---

### Option 4: Hybrid (Answer + Limited Adjacent)

**Implementation:**
```python
def _compute_relevance(self, query, chunk, chunk_to_doc):
    content = chunk.get('content', '')
    chunk_id = chunk.get('chunk_id', '')
    
    # 1. Exact match → rel=3
    if has_exact_match(query.answer, content):
        return 3
    
    # 2. High token-F1 → rel=3
    f1_score = compute_token_f1(query.answer, content)
    if f1_score >= 0.7:
        return 3
    
    # 3. Mid token-F1 → rel=2
    if f1_score >= 0.4:
        return 2
    
    # 4. Adjacent to high-relevance chunk → rel=1 or 2
    for gold_chunk_id in query.gold_chunk_ids:
        if self._is_adjacent_chunk(chunk_id, gold_chunk_id, window=2):
            # Only ±2 chunks (tighter than Option 2)
            semantic_sim = self._compute_semantic_similarity(query.query_text, content)
            if semantic_sim >= 0.80:  # Higher threshold
                return 2
            elif semantic_sim >= 0.70:
                return 1
    
    return 0
```

**Key Differences from Option 2:**
- Smaller window (±2 instead of ±3)
- Higher semantic threshold (0.80 vs 0.75)
- Both must be true (adjacent AND high similarity)

**Predicted Impact:**
- Average relevant chunks: **5-12 per query**
- Recall@10: **18-28%**
- Balance between strict and context-aware

**Pros:**
- ✅ Best of both worlds
- ✅ Focuses on answer but includes immediate context
- ✅ Tighter constraints than Option 2
- ✅ Still objective

**Cons:**
- ⚠️ More complex than Option 1
- ⚠️ Still has arbitrary parameters (window=2, threshold=0.80)

**Best For:** Balanced approach for production use

---

## 📊 Side-by-Side Comparison

| Aspect | Option 1<br/>Pure Answer | Option 2<br/>Adjacent Window | Option 3<br/>Doc-Size Aware | Option 4<br/>Hybrid |
|--------|-------------------------|------------------------------|----------------------------|-------------------|
| **Avg Relevant Chunks** | 3-7 | 8-15 | 10-20 | 5-12 |
| **Expected Recall@10** | 15-25% | 20-35% | 20-30% | 18-28% |
| **Expected Hits@10** | 65-75% | 70-80% | 70-80% | 68-78% |
| **Objectivity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Context-Aware** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Implementation Effort** | Easy | Medium | Hard | Medium |
| **Maintenance** | Easy | Medium | Hard | Medium |
| **Best For** | Benchmarking | Context-heavy RAG | Mixed docs | Production |

---

## 💭 My Recommendation

### Short Term: **Option 1 (Pure Answer-Based)** ⭐ RECOMMENDED

**Why:**
1. **Immediate fix** - Simplest to implement and understand
2. **Objective** - No arbitrary parameters to tune
3. **Portable** - Works regardless of document type/size
4. **Interpretable metrics** - Recall/precision mean what they say
5. **Reveals real issues** - If metrics are still low, it's genuinely a retrieval problem

**Implementation:**
- Remove lines 149-167 in `silver_labeler.py`
- That's it! No new parameters, no complex logic

**Expected Outcome:**
- Recall@10: 15-25% (realistic for answer-focused eval)
- If this is still too low → genuine retrieval issues to fix
- If this is reasonable → you now have trustworthy metrics

### Medium Term: **Option 4 (Hybrid)**

After getting clean metrics with Option 1, you might want to add limited context awareness:
- Implement ±2 chunk window with high semantic threshold
- Compare metrics before/after to see impact
- Decide if the added complexity is worth it

### Long Term: **Separate Evaluation Sets**

Create two evaluation modes:
1. **Strict** (Option 1): For benchmarking and monitoring
2. **Context-Aware** (Option 4): For evaluating full RAG pipeline

Use strict for daily monitoring, context-aware for deep dives.

---

## 🔬 Validation Plan

Whichever option you choose, validate with this process:

### Step 1: Implement & Regenerate
```bash
# Apply the fix to silver_labeler.py
cd evals/synthetic-eval
python main.py --only-labeling  # Regenerate qrels
```

### Step 2: Run Diagnostics
```bash
cd calculate-metrics/temp_diagnostics
python quick_check.py  # Should show avg 3-15 relevant chunks
```

### Step 3: Spot Check Queries
```bash
python query_inspector.py --random 10
```

Manually review:
- Are only truly relevant chunks marked?
- Are we missing obviously relevant chunks?
- Do the relevance scores make sense?

### Step 4: Run Full Evaluation
```bash
cd ..
python main.py
```

Check if metrics are:
- **Interpretable** (recall ≈ precision)
- **Reasonable** (hits@10 > 70%)
- **Stable** (MRR consistent with hits)

### Step 5: Compare Before/After
- Before: recall@10 = 3.2%, avg relevant = 62
- After: recall@10 = ???, avg relevant = ???
- Improvement should be clear!

---

## ❓ Questions for You

Before I implement any fix, please confirm:

1. **Philosophy:** Do you care more about **answer retrieval** (Option 1) or **context retrieval** (Options 2-4)?

2. **Use Case:** Is this evaluation for:
   - Monitoring retrieval quality over time? → Option 1
   - Comparing different retrieval systems? → Option 1
   - Evaluating complete RAG pipeline (retrieval + generation)? → Option 4
   - All of the above? → Option 1 now, add Option 4 later

3. **Multi-hop queries:** Do they need multiple chunks to answer? If yes, Options 2 or 4 might be better.

4. **Risk tolerance:**
   - Conservative (Option 1): Clean metrics, might be "too strict"
   - Moderate (Option 4): Balance, slightly more complex
   - Aggressive (Option 2): More forgiving, might be "too lenient"

5. **Quick test:** Want me to implement Option 1 and show you a sample of results before full regeneration?

Let me know your thoughts and I'll proceed accordingly!

