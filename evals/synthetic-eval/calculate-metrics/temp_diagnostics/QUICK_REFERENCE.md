# Quick Reference Card

## 🔴 The Problem in One Sentence
Average 62 relevant chunks per query (should be 5-10) makes recall@10 = 3.2% inevitable, hiding that your RAG actually achieves 77.5% hits@10.

---

## 📊 Key Numbers

| Metric | Official | Reality | Status |
|--------|----------|---------|--------|
| **Avg Relevant Chunks** | 62 | Should be 5-10 | ❌ CRITICAL |
| **Recall@10** | 3.2% | 59.7% (lenient) | ❌ Deflated |
| **Hits@10** | 64.3% | 77.5% (lenient) | ✅ Good! |
| **MRR** | 31.2% | 43.0% (lenient) | ✅ Reasonable |

---

## 🎯 The Fix (Recommended)

**File:** `evals/synthetic-eval/silver_labeler.py`  
**Action:** Delete/comment lines 149-167 (same-document bonus)  
**Time:** 2 minutes  
**Impact:** Avg relevant chunks: 62 → 5-10, Recall: 3.2% → 15-25%

---

## 📂 Files to Read

**Start here:**
1. `EXECUTIVE_SUMMARY.md` - Overview (this path)
2. `CONCRETE_EXAMPLES.md` - Real query examples (eye-opening!)
3. `FIX_OPTIONS_ANALYSIS.md` - 4 detailed fix options

**Deep dives:**
4. `DIAGNOSTIC_RESULTS.md` - Complete technical analysis
5. `FINDINGS.md` - Initial investigation notes
6. `README.md` - How to use the diagnostic tools

---

## 🛠️ Diagnostic Tools

All in `evals/synthetic-eval/calculate-metrics/temp_diagnostics/`:

```bash
# Quick 5-second check
python quick_check.py

# Full analysis (2 minutes)
python diagnostic_analyzer.py

# Compare strict vs lenient metrics
python alternative_metrics.py

# Inspect specific queries
python query_inspector.py q0067
python query_inspector.py --failed 5
python query_inspector.py --random 10
```

---

## ✅ Action Checklist

- [ ] Read `CONCRETE_EXAMPLES.md` (10 min)
- [ ] Read `FIX_OPTIONS_ANALYSIS.md` (15 min)
- [ ] Decide on fix approach (Option 1, 2, 3, or 4)
- [ ] Apply fix to `silver_labeler.py`
- [ ] Regenerate: `python main.py --only-labeling`
- [ ] Verify: `python quick_check.py` (should show avg 5-10)
- [ ] Re-evaluate: `python main.py` in calculate-metrics
- [ ] Compare before/after metrics

**Total Time:** 2-3 hours

---

## 🎓 Key Insights

1. **Your RAG is good** - 77.5% hits@10 is solid performance
2. **Evaluation is broken** - 62 relevant chunks per query is insane
3. **Root cause identified** - Same-document bonus in silver_labeler.py
4. **Fix is simple** - Remove 18 lines of code
5. **Impact is huge** - Metrics will jump 5-10x and become trustworthy

---

## 💡 Real Query Examples

### Query q0067: "What losses occur from security breaches?"
- ❌ **Marked as FAILURE**
- ✅ **Actually retrieved perfect answer** (Form 10-K 2022_chunk_87)
- 🤔 **Why failed?** Different document than expected (both correct)

### Query q0016: "How will consumption-based plans change?"
- ❌ **Marked as FAILURE** (0/10 matches)
- ✅ **Actually retrieved 2025 data** (more recent than 2022 gold)
- 🤔 **Why failed?** Penalized for preferring recent information!

### Query q0014: "Impact of reduced renewals is ___"
- ✅ **Marked as SUCCESS** (2 exact matches in top-2)
- ⚠️ **Recall = 3.7%** even with perfect retrieval
- 🤔 **Why so low?** 54 chunks marked relevant (entire document)

---

## 🎯 Recommended Option

### Option 1: Pure Answer-Based (RECOMMENDED)

**What:** Remove same-document bonus, only mark answer-containing chunks  
**Why:** Simplest, most objective, most portable  
**Impact:** Avg chunks 62 → 5-10, Recall 3.2% → 15-25%  
**Effort:** 2 minutes to implement  

**Code change:**
```python
# In silver_labeler.py, DELETE lines 149-167:
# (The if chunk_doc_id in gold_doc_ids: block)
```

---

## 📞 Ready to Proceed?

Let me know which option you prefer and I'll:
1. ✅ Apply the fix
2. ✅ Show you the code changes
3. ✅ Help regenerate the qrels
4. ✅ Compare before/after metrics

---

## 🔗 Quick Links

- **Problem explanation:** `CONCRETE_EXAMPLES.md`
- **All fix options:** `FIX_OPTIONS_ANALYSIS.md`
- **Technical details:** `DIAGNOSTIC_RESULTS.md`
- **Tool usage:** `README.md`

**Location:** `evals/synthetic-eval/calculate-metrics/temp_diagnostics/`

