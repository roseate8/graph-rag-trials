# Concrete Examples: Why Metrics Are Misleading

## 🎯 Real Query Analysis

Let me show you EXACTLY what's happening with real queries from your dataset.

---

## Example 1: Query q0067 (Marked as "Failed")

### The Question
**"What types of losses can occur if a company's security is breached?"**

Expected Answer: "loss of intellectual property, data, or customers' data"

### What the Evaluation Says
- ❌ **FAILED** - 0/10 matches
- 54 relevant chunks expected (all from dutch-board-report-fiscal-year-2024-1)
- Recall = 0%, Precision = 0%

### What Actually Got Retrieved (Rank #1, Score 1.0)

**Chunk:** `Form 10-K 2022_chunk_87`

**Content:**
> ## If our security measures are breached, a security incident occurs, or unauthorized access to or other processing of confidential information, including personal data, otherwise occurs, our software may be perceived as not being secure, customers may reduce the use of or stop using our products, and we may incur significant liabilities.
>
> These attacks may come from individual hackers, criminal groups, and state-sponsored organizations, and security breaches and incidents may arise from...

### The Truth
**This chunk PERFECTLY answers the question!**

It explicitly mentions:
- ✅ Loss of customer trust ("customers may reduce the use")
- ✅ Loss of confidential information
- ✅ Loss of personal data
- ✅ Significant liabilities

**Why it failed:** Wrong document!
- Expected: dutch-board-report-fiscal-year-2024-1
- Retrieved: Form 10-K 2022

Both contain the SAME security risk disclosures (required by law), but the evaluation only accepts one specific document.

**Verdict:** False negative - retrieved answer is actually CORRECT and comprehensive!

---

## Example 2: Query q0016 (Marked as "Failed")

### The Question
**"How are our consumption-based plans for Elastic Cloud expected to change over time?"**

Expected Answer: "continue to increase"

### What the Evaluation Says
- ❌ **FAILED** - 0/10 exact matches
- ⚠️ But 6/10 are from the right documents!
- 60 relevant chunks expected (52 from Form 10-K 2022, others from newer reports)
- Recall = 0%

### What Actually Got Retrieved

**Top 5 results:**
1. `Form 10-K 2025_chunk_71` (Score 1.0) - Most recent 10-K
2. `Form 10-K 2022_chunk_101` (Score 0.97) - From same doc as gold chunk!
3. `dutch-board-report-fiscal-year-2024-1_chunk_101` (Score 0.93) - Recent board report
4. `Form 10-K 2025_chunk_16` (Score 0.91) - Most recent 10-K
5. `Form 10-K 2025_chunk_151` (Score 0.86) - Most recent 10-K

### The Truth

**The RAG is being SMART about recency!**

- Gold chunk: Form 10-K 2022_chunk_83 (from **2022**)
- Retrieved: Form 10-K 2025 chunks (from **2025**)

**Question for you:** If a user asks "How are consumption-based plans EXPECTED to change?", should the system return:
- A) 2022 data (what it says in the gold label)
- B) 2025 data (most recent information)

**I'd argue B is better!** The RAG is prioritizing recent information, which is exactly what you want for forward-looking questions!

**Why it failed:** The gold chunk is from 2022, but the RAG found more recent answers from 2025.

**Verdict:** False negative - retrieved answer is actually BETTER than the gold answer!

---

## Example 3: The Same-Document Inflation Problem

### Query q0016 Details

**Expected relevant chunks: 60 total**

Breaking down by document:
- **52 chunks** from Form 10-K 2022 (entire document!)
- 3 chunks from Form 10-K 2024
- 3 chunks from annual-report-fiscal-year-2024
- 1 chunk from Form 10-K 2025
- 1 chunk from dutch-board-report-fiscal-year-2024-1

### Why 52 Chunks from Form 10-K 2022?

The gold chunk is `Form 10-K 2022_chunk_83`.

Current labeling logic:
```python
if chunk is from same document as gold_chunk:
    if semantic_similarity >= 0.75:
        mark as relevant (score 2)
    else:
        mark as relevant (score 1)
```

**Result:** ALL 52 chunks from Form 10-K 2022 marked as relevant!

These include completely unrelated sections:
- Form 10-K 2022_chunk_15: "We make some features available for free..."
- Form 10-K 2022_chunk_155: "Unlike some companies, we do not build..."
- Form 10-K 2022_chunk_17: Business description
- Form 10-K 2022_chunk_22: Revenue recognition
- ... and 48 more chunks!

**Mathematical Impact:**
- To get recall@10 = 0.50, you'd need to retrieve 30 of the 60 relevant chunks!
- Current recall@10 = 0%, even though 6/10 are from the right documents
- If we only marked 5 chunks as relevant, recall would be 0/5 = 0%, but that's more meaningful than 0/60

---

## Example 4: What Good Performance Looks Like

### Query q0014 (Marked as "Success")

**Question:** "The impact of reduced subscription renewals is ___ in our current financial statements."

**Expected Answer:** "not reflected in full in our results of operations until future periods"

### What Got Retrieved

**Top 2 (both scored perfectly):**
1. ✅ `annual-report-fiscal-year-2024_chunk_73` (Score 1.0)
2. ✅ `Form 10-K 2024_chunk_73` (Score 1.0)

**Content of both chunks:**
> ## Because we recognize the vast majority of the revenue from subscriptions, either based on actual consumption or ratably over the term of the relevant subscription period, downturns or upturns in sales are not immediately reflected in full in our results of operations.

**Result:**
- 2 exact matches in top-10
- But still 54 total "relevant" chunks (entire document marked)
- Recall = 2/54 = 0.037 (3.7%)

### The Paradox

Even when the RAG performs PERFECTLY (finds exact answer at rank #1 AND #2), the recall is only 3.7%!

**Why?** Because 52 other chunks from the same document are also marked as "relevant."

---

## 📊 Summary Statistics

### From Our Analysis

**72% of queries** (36/50 sampled) find exact chunk matches in top-10
**80% of queries** (40/50) find the correct document in top-10

But overall metrics show:
- Recall@10 = 3.2%
- Hits@10 = 64.3%

### The Disconnect

| Metric | What It Shows | What It Means |
|--------|---------------|---------------|
| **Hits@10 = 64%** | 64% of queries find at least one relevant chunk | ✅ Reasonable! |
| **Recall@10 = 3.2%** | Retrieves 3.2% of all relevant chunks | ❌ Misleading! (60+ chunks marked per query) |
| **Lenient Hits@10 = 77.5%** | 77.5% find the right document | ✅ Actually quite good! |
| **Lenient Recall@10 = 59.7%** | Retrieves 60% of relevant documents | ✅ This is the real performance! |

---

## 🎯 The Core Problem

### Current Labeling Logic

```python
# Step 1: Mark gold chunks as rel=3
if chunk_id in gold_chunk_ids:
    return 3

# Step 2: Mark high F1 chunks as rel=3
if token_f1(answer, chunk) >= 0.7:
    return 3

# Step 3: Mark mid F1 chunks as rel=2  
if token_f1(answer, chunk) >= 0.4:
    return 2

# Step 4: Mark ALL chunks from same document as rel=1 or 2
if chunk.doc_id == gold_chunk.doc_id:
    if semantic_similarity >= 0.75:
        return 2  # "supporting context"
    else:
        return 1  # "same topic"
```

**The problem is Step 4!**

For a 10-K form with 200 chunks:
- Step 1: 1 chunk marked (gold)
- Step 2: 0-2 chunks (high F1)
- Step 3: 1-3 chunks (mid F1)
- **Step 4: 195 chunks marked!** (all remaining chunks from doc)

**Result:** 200 chunks marked as relevant when only 5-6 actually contain answer-related content!

---

## 💡 What Happens After the Fix

### Option 1: Remove Step 4 (Same-Document Bonus)

**New Logic:**
```python
# Step 1: Gold chunks → rel=3
if chunk_id in gold_chunk_ids:
    return 3

# Step 2: High F1 → rel=3
if token_f1(answer, chunk) >= 0.7:
    return 3

# Step 3: Mid F1 → rel=2
if token_f1(answer, chunk) >= 0.4:
    return 2

# Step 4: Default → not relevant
return 0
```

**Expected result for Query q0016:**
- Gold chunk: 1 (Form 10-K 2022_chunk_83)
- High F1 chunks: 2-3 (chunks that mention "consumption-based" and "increase")
- Mid F1 chunks: 1-2 (chunks with partial mentions)
- **Total: 4-6 chunks** instead of 60!

**Expected metrics:**
- Recall@10 = 0/5 = 0% (still fails because retrieved 2025 data, not 2022)
- But this is HONEST failure - it genuinely didn't retrieve the 2022 chunks
- You can then investigate: Should we penalize recency preference?

### For Query q0014 (the successful one):

**Current:**
- 2 exact matches found
- 54 chunks marked relevant
- Recall = 2/54 = 3.7%

**After fix:**
- 2 exact matches found
- 5 chunks marked relevant (only those with answer content)
- Recall = 2/5 = 40%

**This makes sense!** Found 2 out of 5 answer-containing chunks = 40% recall.

---

## 🎓 Key Insights

### 1. Document-Level Matching Hides Problems

When you mark all chunks from a document as relevant, you can't tell if:
- The system found the right section (good)
- The system found the right document but wrong section (okay)
- The system found any random chunk from the document (bad)

They all count as "success" but have very different quality!

### 2. Large Documents Break This Approach

- 5-chunk document + same-doc bonus = 5 relevant chunks ✅ Manageable
- 200-chunk document + same-doc bonus = 200 relevant chunks ❌ Impossible

### 3. False Negatives Are Common

Both examples above (q0067, q0016) show the RAG retrieving CORRECT or BETTER answers but being marked as failures.

This means:
- Your monitoring will show declining metrics even as RAG improves
- You can't trust the evaluation to guide improvements
- A/B testing becomes meaningless (can't tell which system is better)

### 4. The Fix Is Straightforward

Remove the document-level bonus, keep only answer-based relevance:
- Objective (based on answer presence)
- Scalable (works for any document size)
- Interpretable (recall means what it says)
- Honest (false negatives become real debugging opportunities)

---

## ❓ Questions for Decision

1. **Is q0067 a success or failure?**
   - Retrieved perfect answer from Form 10-K 2022
   - Expected answer from dutch-board-report-2024
   - **Your call:** Same answer, different doc - is this acceptable?

2. **Is q0016 a success or failure?**
   - Retrieved 2025 data (most recent)
   - Expected 2022 data (what was used to generate query)
   - **Your call:** Newer data for "expected to change" - is this better?

3. **What's your priority?**
   - A) Exact replication (retrieve the exact chunk used to generate query)
   - B) Answer correctness (retrieve any chunk with correct answer)
   - C) Recency (prefer newer data when available)
   
   I'd recommend **B** for production RAG!

4. **Context windows:**
   - Do you need adjacent chunks for context? (e.g., for generation)
   - Or is the single best chunk enough?
   
   This determines whether Option 1 (strict) or Option 4 (hybrid) is better.

---

Let me know your thoughts and which direction you'd like to go!

