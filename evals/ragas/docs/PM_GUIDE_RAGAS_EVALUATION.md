# Ragas Evaluation System - Product Manager Guide

**Last Updated**: November 2025  
**Audience**: Technical Product Managers  
**Purpose**: Understand Ragas evaluation strategy, workflow, and key decision points

---

## 1. What Problem Are We Solving?

### The Challenge

You've built a RAG (Retrieval-Augmented Generation) system that answers user questions by:
1. Finding relevant document chunks from your knowledge base
2. Feeding those chunks to an LLM
3. Getting back an answer

**The question is: How do you know if it's working well?**

Traditional testing approaches have problems:
- **Manual Testing**: Doesn't scale - you can only test 10-20 queries manually
- **User Feedback**: Too slow - takes weeks/months to collect meaningful data
- **Fixed Test Sets**: Miss edge cases and don't evolve with your content
- **No Ground Truth**: You don't have "correct answers" to compare against

### The Solution: Ragas

Ragas solves this by **automatically generating synthetic test data** that mimics real user questions, then evaluating your RAG system against multiple quality dimensions.

Think of it as: **"Having an AI generate 100+ realistic test questions from your actual documents, then grading your system's answers on multiple criteria."**

---

## 2. How Does Ragas Work? (Business View)

### The Three-Stage Process

```
STAGE 1: Generate Test Questions (One-Time Setup)
Your Documents in Milvus → Ragas Analyzer → 100+ Synthetic Questions

STAGE 2: Run Your RAG System (Testing Phase)
Each Test Question → Your RAG System → Answer + Retrieved Chunks

STAGE 3: Evaluate Quality (Automated Grading)
Ragas Evaluator → Scores on 7 Dimensions → Quality Report
```

Let me break down each stage:

---

### **STAGE 1: Synthetic Test Generation** (What This Project Does)

**What Happens:**
1. Ragas connects to your Milvus database containing your document chunks
2. It reads 500 random chunks from your knowledge base
3. Using GPT-4, it generates diverse test questions based on those chunks
4. It creates 100 question-answer pairs with known "ground truth"

**Why This Matters:**
- **Scales with your content**: When you add new documents, regenerate tests
- **Covers edge cases**: Generates simple questions, complex reasoning, multi-document questions
- **Has ground truth**: You know what the correct answer should be
- **No manual work**: Fully automated process

**Business Value:**
- **One-time cost**: ~$2-5 in OpenAI API costs for 100 questions
- **Reusable**: Same test set can be run repeatedly as you improve your RAG system
- **Comprehensive**: Tests multiple query types (simple, reasoning, multi-context)

---

### **STAGE 2: Running Your RAG System** (Separate Activity)

This is NOT part of the Ragas implementation, but here's what happens:

**For each test question:**
1. Your RAG system retrieves relevant chunks from Milvus
2. It formats those chunks into a prompt
3. It sends the prompt to your LLM
4. It gets back an answer

**What You Collect:**
- The question
- The answer your system generated
- The chunks your system retrieved
- The "ground truth" answer from Ragas

---

### **STAGE 3: Evaluation** (What This Project Also Does)

**What Ragas Measures:**

| Metric | What It Measures | Why It Matters |
|--------|------------------|----------------|
| **Faithfulness** | Is the answer accurate based on retrieved chunks? | Prevents hallucination |
| **Answer Relevancy** | Does the answer actually address the question? | User satisfaction |
| **Context Recall** | Did you retrieve the RIGHT chunks? | Retrieval quality |
| **Context Precision** | Are the retrieved chunks ranked properly? | Noise reduction |
| **Answer Similarity** | How close is your answer to ground truth? | Overall correctness |
| **Answer Correctness** | Semantic + factual correctness combined | Comprehensive quality |

**Output:**
- Overall scores (0.0 - 1.0 scale) for each metric
- Per-question breakdown showing failures
- Trends over time as you improve your system

**Business Value:**
- **Objective metrics**: Not subjective "looks good to me"
- **Comparable**: Track improvements over time (e.g., "faithfulness up 15%")
- **Diagnostic**: Know WHERE your system fails (retrieval? generation?)

---

## 3. Your Implementation: The Data Flow

### What You Actually Built

```
┌─────────────────────────────────────────────────────────┐
│  YOUR DOCUMENT PIPELINE (Already Exists)               │
│                                                          │
│  Documents → Chunking → Embeddings → Milvus Storage    │
│  (vector-ingest/main.py)                                │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ document_chunks collection
                         │ (localhost:19530)
                         ↓
┌─────────────────────────────────────────────────────────┐
│  RAGAS TEST GENERATION (evals/ragas/)                   │
│                                                          │
│  Step 1: Load Documents (milvus_loader.py)             │
│  ├─ Connect to Milvus                                   │
│  ├─ Load 500 random chunks                              │
│  └─ Convert to LangChain format                         │
│                                                          │
│  Step 2: Generate Tests (generate_testset.py)          │
│  ├─ Use GPT-4o-mini to analyze chunks                   │
│  ├─ Create diverse question types:                      │
│  │  • 40% Simple (fact-based)                           │
│  │  • 30% Reasoning (inference needed)                  │
│  │  • 20% Multi-context (needs multiple chunks)         │
│  │  • 10% Conditional (if-then logic)                   │
│  └─ Generate 100 question-answer pairs                  │
│                                                          │
│  Output: testset.csv, testset.json                      │
└─────────────────────────────────────────────────────────┘
                         │
                         │ testset files
                         ↓
┌─────────────────────────────────────────────────────────┐
│  YOUR RAG SYSTEM (retrieval/)                           │
│                                                          │
│  For each test question:                                │
│  ├─ Retrieve chunks from Milvus                         │
│  ├─ Optionally decompose complex queries                │
│  ├─ Optionally re-rank results                          │
│  ├─ Format context + question                           │
│  └─ Generate answer with LLM                            │
│                                                          │
│  Output: answers + retrieved_contexts                   │
└─────────────────────────────────────────────────────────┘
                         │
                         │ results dataset
                         ↓
┌─────────────────────────────────────────────────────────┐
│  RAGAS EVALUATION (evaluate_rag.py)                     │
│                                                          │
│  Compare: Your Answer vs Ground Truth vs Context        │
│                                                          │
│  Calculate 7 metrics for each question:                 │
│  ├─ Faithfulness                                        │
│  ├─ Answer Relevancy                                    │
│  ├─ Context Recall                                      │
│  ├─ Context Precision                                   │
│  ├─ Answer Similarity                                   │
│  ├─ Answer Correctness                                  │
│  └─ Context Relevancy                                   │
│                                                          │
│  Output: evaluation_results.csv, scores_summary.json    │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Key PM Decision Points

### Decision Point 1: Test Set Size

**What**: How many synthetic questions to generate

**Configuration**: `evals/ragas/config.py` → `RAGAS_CONFIG["testset_size"]`

**Trade-offs**:
| Size | Cost (OpenAI) | Coverage | Time to Generate | When to Use |
|------|---------------|----------|------------------|-------------|
| 50 | $1-2 | Basic | 5-10 min | Initial testing |
| 100 | $2-5 | Good | 10-20 min | **Default/Recommended** |
| 250 | $5-12 | Comprehensive | 30-60 min | Production readiness |
| 500+ | $10-25 | Exhaustive | 1-2 hours | Large-scale validation |

**Recommendation**: Start with 100, scale to 250 when preparing for production.

---

### Decision Point 2: Question Distribution

**What**: Mix of question complexity types

**Configuration**: `evals/ragas/config.py` → `RAGAS_CONFIG["distributions"]`

**Current Settings**:
```python
{
    "simple": 0.4,        # 40% - Direct fact retrieval
    "reasoning": 0.3,     # 30% - Inference required
    "multi_context": 0.2, # 20% - Multiple chunks needed
    "conditional": 0.1    # 10% - If-then logic
}
```

**When to Adjust**:

| Use Case | Adjust Distribution | Reason |
|----------|---------------------|--------|
| **FAQ/Help Desk** | simple: 0.6, reasoning: 0.2 | Users ask straightforward questions |
| **Research/Analysis** | reasoning: 0.4, multi_context: 0.3 | Complex queries are the norm |
| **Technical Docs** | conditional: 0.2, multi_context: 0.3 | Procedural "how-to" questions |
| **General Purpose** | **Keep default** | Balanced coverage |

---

### Decision Point 3: Document Sampling Strategy

**What**: How to select which 500 chunks to use for test generation

**Configuration**: `evals/ragas/config.py` → `RAGAS_CONFIG["sample_strategy"]`

**Options**:
- `"random"` (default): Pure random sampling
- `"sequential"`: First N documents (for deterministic testing)

**When It Matters**:
- **Random**: Best for general coverage, different results each run
- **Sequential**: Reproducible results, good for A/B testing specific improvements

**Recommendation**: Use `"random"` unless you're doing controlled experiments.

---

### Decision Point 4: Evaluation Frequency

**What**: How often to regenerate tests and re-evaluate

**Strategy Options**:

| Trigger | Frequency | Rationale |
|---------|-----------|-----------|
| **Content Update** | Every major doc update | New content = new test questions needed |
| **System Change** | After retrieval/ranking changes | Measure impact of improvements |
| **Weekly Regression** | Every Monday | Catch degradation early |
| **Pre-Release Gate** | Before each deployment | Quality checkpoint |

**Recommendation**: 
- Regenerate test set: Monthly or on major content updates
- Re-run evaluation: Before each significant system change

---

### Decision Point 5: Quality Thresholds

**What**: Minimum acceptable scores for production

**Where to Set**:
This is business-driven based on your quality standards.

**Industry Benchmarks** (from Ragas research):

| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| Faithfulness | <0.6 | 0.6-0.75 | 0.75-0.85 | >0.85 |
| Answer Relevancy | <0.5 | 0.5-0.7 | 0.7-0.85 | >0.85 |
| Context Recall | <0.5 | 0.5-0.7 | 0.7-0.85 | >0.85 |
| Context Precision | <0.4 | 0.4-0.6 | 0.6-0.8 | >0.8 |
| Answer Correctness | <0.5 | 0.5-0.65 | 0.65-0.8 | >0.8 |

**Your Action**: Define minimum scores for your use case:
- **Customer-facing**: Target "Good" or "Excellent"
- **Internal tools**: "Acceptable" may suffice
- **High-stakes (legal, medical)**: Only "Excellent" is acceptable

---

## 5. Cost Analysis

### One-Time Setup Costs

**OpenAI API Usage** (for test generation):
- **Model**: GPT-4o-mini
- **Input**: ~500 document chunks (avg 500 tokens each) = 250K tokens
- **Output**: ~100 questions + answers (avg 100 tokens each) = 10K tokens
- **Cost**: ~$2-5 per test set generation

**Time Investment**:
- Generation time: 10-20 minutes for 100 questions
- Manual review: 30-60 minutes to spot-check quality

### Per-Evaluation Costs

**For 100 test questions**:
- **Model**: GPT-4o-mini (for evaluation)
- **Input**: ~100 questions × (question + answer + context) = ~50K tokens
- **Cost**: ~$0.50-1.00 per evaluation run

**Total Monthly Cost Estimate**:
- 1 test set generation: $5
- 4 weekly evaluations: $4
- **Total: ~$10/month**

This is negligible compared to the cost of manual testing or poor user experiences.

---

## 6. Integration with Your Existing System

### What You Already Have

Your project has three independent pipelines:

```
1. INGESTION (vector-ingest/)
   └─ Documents → Chunks → Milvus

2. RETRIEVAL (retrieval/)
   └─ Query → Milvus Search → LLM → Answer

3. EVALUATION (evals/ragas/)
   └─ Milvus → Test Generation → Evaluate Retrieval
```

### The Integration Points

| Pipeline | Shares With Ragas | How |
|----------|-------------------|-----|
| **Ingestion** | Milvus storage | Ragas reads from same `document_chunks` collection |
| **Ingestion** | API key management | Both use `llm_utils` for secure OpenAI keys |
| **Retrieval** | Milvus connection | Ragas loader uses same connection config |
| **Retrieval** | Evaluation target | Ragas tests the retrieval pipeline output |

**Key Insight**: Ragas doesn't modify or interfere with your production pipelines. It's read-only and runs independently.

---

## 7. Workflow: How to Actually Use This

### Initial Setup (First Time)

```bash
# 1. Ensure you have documents ingested into Milvus
cd vector-ingest
python main.py --input input/your-docs/

# 2. Generate synthetic test set
cd evals/ragas
python generate_testset.py --testset-size 100

# This creates:
# - output/testset.csv
# - output/testset.json
# - output/generation_report.txt
```

**You will be prompted for your OpenAI API key** (secured via `llm_utils`, not stored permanently)

---

### Regular Testing Workflow

```bash
# 1. Run your RAG system on each test question
#    (You need to implement this - it calls retrieval/core.py)
cd retrieval
python your_rag_runner.py --testset ../evals/ragas/output/testset.csv

# This should produce:
# - results.csv with columns: [question, answer, retrieved_contexts]

# 2. Evaluate results with Ragas
cd evals/ragas
python evaluate_rag.py --results ../retrieval/results.csv

# This creates:
# - output/evaluation_results.csv
# - output/scores_summary.json
```

---

### Interpreting Results

**Good Evaluation Output Example**:
```json
{
  "average_scores": {
    "faithfulness": 0.82,
    "answer_relevancy": 0.78,
    "context_recall": 0.75,
    "context_precision": 0.68,
    "answer_correctness": 0.73
  },
  "total_questions": 100,
  "evaluation_time": "2024-11-04 14:23:00"
}
```

**What This Tells You**:
- ✅ **Faithfulness (0.82)**: Good - answers are mostly accurate to retrieved content
- ⚠️ **Context Precision (0.68)**: Acceptable - some noise in retrieved chunks
- ⚠️ **Context Recall (0.75)**: Could improve - missing some relevant chunks

**Action**: Focus on improving retrieval (context recall/precision) rather than generation.

---

## 8. When Things Go Wrong: Diagnostics

### Low Faithfulness Score (<0.6)

**Symptom**: Answers don't match retrieved content  
**Root Cause**: LLM is hallucinating or ignoring context  
**Fix**: 
- Improve system prompt (tell LLM to stick to context)
- Reduce temperature (make it less creative)
- Check if context chunks are actually relevant

---

### Low Context Recall (<0.5)

**Symptom**: Not retrieving the right chunks  
**Root Cause**: Poor retrieval strategy  
**Fix**:
- Increase `top_k` (retrieve more chunks)
- Enable query decomposition (`enable_query_decomposition=True`)
- Check embedding quality (maybe re-embed documents)

---

### Low Context Precision (<0.4)

**Symptom**: Retrieving too much noise  
**Root Cause**: Irrelevant chunks ranked highly  
**Fix**:
- Enable re-ranking (`enable_reranking=True`)
- Adjust similarity threshold
- Improve chunking strategy (better chunk boundaries)

---

### Low Answer Relevancy (<0.5)

**Symptom**: Answers don't address the question  
**Root Cause**: LLM is going off-topic  
**Fix**:
- Improve prompt structure
- Check if retrieved chunks are actually relevant
- Try different LLM model

---

## 9. Advanced Strategy: Iterative Improvement Loop

```
┌─────────────────────────────────────────────────┐
│ Week 1: Baseline                                 │
│ • Generate 100 test questions                    │
│ • Run evaluation → Scores: F=0.65, R=0.58       │
│ • Identify: Low recall = retrieval problem       │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│ Week 2: Fix Retrieval                           │
│ • Enable query decomposition                     │
│ • Increase top_k from 10 → 20                    │
│ • Re-run evaluation → Scores: F=0.67, R=0.72    │
│ • Improvement: Recall up 24%! ✓                  │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│ Week 3: Reduce Noise                            │
│ • Enable cross-encoder re-ranking                │
│ • Re-run evaluation → Scores: F=0.78, R=0.74    │
│ • Improvement: Faithfulness up 16% ✓             │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│ Week 4: Production Ready                        │
│ • All scores >0.7 (acceptable threshold)         │
│ • Generate larger test set (250 questions)       │
│ • Final validation → Deploy to production        │
└─────────────────────────────────────────────────┘
```

**Key PM Activity**: Track week-over-week improvements, prioritize fixes based on biggest gaps.

---

## 10. Operational Considerations

### When to Regenerate Test Sets

**Triggers**:
- ✅ Added >20% new documents
- ✅ Changed document domain (e.g., added legal docs to tech docs)
- ✅ Major content overhaul
- ❌ Minor edits to existing docs (not needed)
- ❌ After every evaluation (unnecessary)

**Frequency**: Monthly or quarterly for stable content bases

---

### Security & Compliance

**API Key Management**:
- ✅ Uses secure `llm_utils` (no environment variables)
- ✅ Keys stored in memory only (300-minute timeout)
- ✅ User prompted each session
- ✅ Automatic cleanup on exit

**Data Privacy**:
- ⚠️ Your document content is sent to OpenAI for test generation
- ⚠️ Ensure compliance with your data handling policies
- ✅ Generated test sets contain no PII (if your docs don't)

**Action**: Review your organization's AI usage policy before running on production data.

---

### Resource Requirements

**Milvus**:
- Same instance as your production Milvus
- Read-only access to `document_chunks` collection
- No additional storage needed

**Compute**:
- Test generation: Light (runs on laptop)
- Evaluation: Light (mostly API calls)
- No GPU required

**Network**:
- Requires internet for OpenAI API calls
- ~5-10 MB data transfer per test set generation

---

## 11. Success Metrics & Reporting

### KPIs to Track

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Average Faithfulness** | Overall answer accuracy | >0.75 |
| **Average Context Recall** | Retrieval effectiveness | >0.70 |
| **Trend Over Time** | Are you improving? | Upward |
| **Test Coverage** | % of docs with generated questions | >80% |
| **Eval Run Frequency** | How often you test | Weekly |

---

### Executive Summary Template

Use this for stakeholder reporting:

```
RAG System Quality Report - Week of [DATE]

Overall Health: [GREEN/YELLOW/RED]

Key Metrics:
• Faithfulness: 0.78 (↑ 12% from last month)
• Answer Relevancy: 0.76 (→ no change)
• Context Recall: 0.71 (↓ 3% - needs attention)

Test Coverage:
• 100 synthetic questions across 4 document types
• Last regenerated: [DATE]

Actions Taken:
• Enabled query decomposition (improved recall)
• Increased top_k to 20 (improved coverage)

Next Steps:
• Investigate recall drop - check new document quality
• Plan re-ranking implementation (target: +10% precision)
```

---

## 12. FAQ (Product Manager Edition)

### Q: Why synthetic tests instead of real user queries?

**A**: Three reasons:
1. **Coverage**: Real queries are biased toward easy questions that already work
2. **Ground Truth**: You know the correct answer for synthetic questions
3. **Scale**: Generate 100+ tests in minutes vs. waiting months for user data

You should do BOTH - synthetic for breadth, real queries for validation.

---

### Q: Can I use this in CI/CD?

**A**: Yes, but be mindful of:
- **Cost**: Every pipeline run costs $1-2 in OpenAI API calls
- **Time**: Evaluation takes 5-10 minutes for 100 questions
- **Recommendation**: Run on feature branches, not every commit

---

### Q: What if my scores are terrible (all <0.5)?

**A**: Don't panic. Common causes:
1. **Bad test generation**: Check if generated questions make sense
2. **Wrong configuration**: Verify Milvus connection, embedding model
3. **RAG system issues**: Your retrieval might actually be broken (good to know!)

Start by manually reviewing 10 test questions + answers.

---

### Q: How does this compare to the BIER evaluation?

**A**: Different approaches:

| Aspect | Ragas | BIER |
|--------|-------|------|
| **Test Data** | Synthetic (auto-generated) | Real-world (HotpotQA dataset) |
| **Setup Time** | 10 minutes | Hours (need external dataset) |
| **Cost** | $2-10/month | Free (one-time download) |
| **Coverage** | Your specific documents | General knowledge |
| **Best For** | Domain-specific RAG | General-purpose systems |

**Use both**: BIER for baseline capability, Ragas for domain-specific quality.

---

## 13. Next Steps & Rollout Plan

### Phase 1: Proof of Concept (Week 1)
- [ ] Generate initial test set (50 questions)
- [ ] Manually review 10 questions for quality
- [ ] Run baseline evaluation
- [ ] Present results to team

### Phase 2: Full Evaluation (Week 2-3)
- [ ] Generate production test set (100-250 questions)
- [ ] Run comprehensive evaluation
- [ ] Identify top 3 improvement areas
- [ ] Create improvement roadmap

### Phase 3: Continuous Monitoring (Ongoing)
- [ ] Schedule weekly evaluation runs
- [ ] Set up alerts for score drops
- [ ] Regenerate test sets monthly
- [ ] Track trends in dashboard

---

## 14. References & Further Reading

### Internal Documentation
- `evals/ragas/README.md` - Implementation details
- `evals/ragas/docs/SECURE_API_KEY_INTEGRATION.md` - Security approach
- `retrieval/RETRIEVAL.md` - RAG system architecture

### External Resources
- [Ragas Official Docs](https://docs.ragas.io/)
- [RAG Evaluation Best Practices](https://arxiv.org/abs/2309.15217)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

## Appendix: Configuration Reference

### Quick Config Checklist

```python
# evals/ragas/config.py

# How many test questions to generate
RAGAS_CONFIG["testset_size"] = 100

# Question complexity mix (must sum to 1.0)
RAGAS_CONFIG["distributions"] = {
    "simple": 0.4,
    "reasoning": 0.3,
    "multi_context": 0.2,
    "conditional": 0.1
}

# How many documents to sample for test generation
RAGAS_CONFIG["max_documents"] = 500

# Which OpenAI models to use
RAGAS_CONFIG["generator_model"] = "gpt-4o-mini"
RAGAS_CONFIG["embeddings_model"] = "text-embedding-3-small"
```

---

**Questions?** Contact the engineering team or refer to the technical documentation.

**Last Updated**: November 2025  
**Document Owner**: Product Management  
**Next Review**: Monthly or on major system changes

