# Re-ranking System Integration

## Overview

The re-ranking system has been successfully integrated into the main workflow, providing enhanced document retrieval through a two-stage process:

1. **Initial Retrieval**: Retrieve `10*K` chunks (default: 100 chunks when K=10)
2. **Re-ranking**: Use cross-encoder re-ranker to select the top `K` most relevant chunks (default: 10 final chunks)

## Integration Points

### 1. RAG UI (rag-ui/app.py)

**Default Configuration:**
- Re-ranking is **enabled by default**
- Retrieval multiplier: **10x** (100 initial → 10 final chunks)
- Model: `cross-encoder/ms-marco-MiniLM-L12-v2`

**API Parameters:**
```javascript
// POST /api/query
{
  "query": "Your question here",
  "top_k": 10,                    // Final number of chunks
  "enable_reranking": true,       // Enable/disable re-ranking
  "retrieval_multiplier": 10,     // Initial retrieval multiplier
  "model": "gpt-4o-mini",
  "temperature": 0.1
}
```

**Enhanced Metrics:**
- `reranking_enabled`: Boolean indicating if re-ranking was used
- `initial_chunks`: Number of chunks initially retrieved
- `final_chunks`: Number of chunks after re-ranking
- `retrieval_method`: Description of retrieval pattern

### 2. Core RAG System (naive-rag/core.py)

**Default Behavior:**
- CLI now uses re-ranking by default
- Factory functions support re-ranking configuration
- Enhanced conversation history with re-ranking metrics

**New Function:**
```python
def test_reranking_comparison(test_query="...", top_k=10):
    """Compare performance with and without re-ranking"""
    # Returns detailed comparison results
```

### 3. Retrieval System (naive-rag/retrieval.py)

**Enhanced Capabilities:**
- Two-stage retrieval process
- Configurable retrieval multiplier
- Graceful fallback if re-ranking fails
- Comprehensive timing metrics

## Configuration Options

### Re-ranker Configuration

```python
from naive_rag.re_rankers import ReRankerConfig

# Default configuration
config = ReRankerConfig()

# Performance-optimized configuration
fast_config = ReRankerConfig(
    batch_size=16,
    max_length=256,
    enable_caching=True
)

# Accuracy-optimized configuration
accurate_config = ReRankerConfig(
    batch_size=4,
    max_length=1024,
    normalize_scores=True
)
```

### System Integration

```python
from naive_rag import create_rag_system

# Default: re-ranking enabled
rag = create_rag_system(
    llm_type="openai",
    enable_reranking=True,
    retrieval_multiplier=10
)

# Custom re-ranker configuration
rag = create_rag_system(
    llm_type="openai",
    enable_reranking=True,
    reranker_config={
        "batch_size": 8,
        "device": "cuda"
    }
)
```

## Performance Benefits

### Quality Improvements
- **Better Relevance**: Cross-encoder re-ranker provides superior semantic matching
- **Larger Candidate Pool**: Initial 100-chunk retrieval captures more relevant documents
- **Context Preservation**: Higher-quality chunks improve LLM response accuracy

### Monitoring Capabilities
- **Detailed Timing**: Separate metrics for vector search and re-ranking
- **Resource Tracking**: Memory and CPU usage monitoring
- **Comparison Tools**: Built-in A/B testing functionality

## Usage Examples

### Basic Usage

```python
# Simple query with re-ranking (default)
from naive_rag import ask_rag

response = ask_rag(
    "What are the key financial metrics?",
    enable_reranking=True  # Default
)
```

### Performance Testing

```python
# Compare retrieval methods
from naive_rag.core import test_reranking_comparison

results = test_reranking_comparison(
    test_query="What are the revenue trends?",
    top_k=10
)

print(f"Traditional: {results['without_reranking']['total_time']:.2f}s")
print(f"Re-ranking: {results['with_reranking']['total_time']:.2f}s")
print(f"Pattern: {results['comparison']['initial_vs_final']}")
```

### Web UI Integration

The RAG UI automatically uses re-ranking with:
- Real-time performance metrics
- Configurable parameters via API
- Enhanced conversation history
- Visual indicators of retrieval method

## System Architecture

```
User Query
    ↓
RAG System (enable_reranking=True)
    ↓
Vector Search (retrieve 100 chunks)
    ↓
Cross-encoder Re-ranker (rank to top 10)
    ↓
Context Formatting
    ↓
LLM Generation
    ↓
Response with Metrics
```

## Cleanup Summary

- ✅ Removed `test_reranking.py` test file
- ✅ Integrated test functionality into `core.py`
- ✅ Enhanced main workflow with re-ranking
- ✅ Updated UI with re-ranking metrics
- ✅ No remaining temporary/test files

The re-ranking system is now fully integrated and ready for production use with enhanced retrieval quality and comprehensive monitoring capabilities.


graph TD
    A["User Query"] --> B["RAG System"]
    B --> C["Enable Re-ranking?"]
    
    C -->|No| D["Direct Vector Search<br/>Get top K chunks"]
    C -->|Yes| E["Two-Stage Retrieval"]
    
    E --> F["Stage 1: Vector Search<br/>Retrieve 10*K chunks<br/>(Default: K=10 → 100 chunks)"]
    F --> G["Stage 2: Cross-encoder Re-ranker<br/>cross-encoder/ms-marco-MiniLM-L12-v2"]
    G --> H["Return top K chunks<br/>(10 final chunks)"]
    
    D --> I["Context Formatting"]
    H --> I
    I --> J["LLM Generation<br/>(OpenAI GPT)"]
    J --> K["Response with Metrics"]
    
    subgraph "Re-ranking Components"
        L["base.py<br/>BaseReRanker Interface"]
        M["config.py<br/>ReRankerConfig"]
        N["reranker_model.py<br/>Cross-encoder Implementation"]
    end
    
    subgraph "Performance Tracking"
        P["Vector Search Time"]
        Q["Re-ranking Time"]
        R["Context Format Time"]
        S["LLM Generation Time"]
        T["Total Pipeline Time"]
    end
    
    subgraph "Integration Points"
        U["rag-ui/app.py<br/>Web API + UI"]
        V["core.py<br/>Main RAG Pipeline"]
        W["retrieval.py<br/>Two-Stage Retrieval"]
    end
    
    G -.-> Q
    F -.-> P
    I -.-> R
    J -.-> S
    K -.-> T