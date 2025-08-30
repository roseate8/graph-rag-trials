# Embeddings Module

This module provides comprehensive embedding generation and vector storage capabilities for the Graph-RAG system using Milvus as the vector database.

## Components

### 📦 Core Components

1. **EmbeddingService** - Generates embeddings using sentence transformers
2. **MilvusVectorStore** - Handles vector storage and similarity search with Milvus
3. **EmbeddingsManager** - Coordinates embedding generation and storage
4. **Configuration** - Centralized configuration for different deployment scenarios

## 🚀 Quick Start

### Basic Usage

```python
from src.embeddings import EmbeddingsManager, MilvusConfig

# Configure for local Milvus
config = MilvusConfig.get_default_config()

# Initialize manager
manager = EmbeddingsManager(
    embedding_model="BAAI/bge-small-en-v1.5",
    milvus_config=config
)

# Use with context manager for automatic cleanup
with manager:
    # Process and store chunks
    success = manager.process_and_store_chunks(chunks)
    
    # Search for similar content
    results = manager.search_similar_chunks("your query here", top_k=5)
```

### Configuration Options

#### Local Milvus
```python
config = MilvusConfig.get_default_config()
# Uses localhost:19530 by default
```

#### Zilliz Cloud (Managed Milvus)
```python
config = MilvusConfig.get_cloud_config()
config.update({
    "host": "your-endpoint.zillizcloud.com",
    "user": "your_username", 
    "password": "your_password"
})
```

#### Custom Configuration
```python
config = MilvusConfig.get_custom_config(
    host="your-host.com",
    port="19530",
    collection_name="my_documents"
)
```

## 🔧 Installation

### Required Dependencies

```bash
pip install -r requirements_embeddings.txt
```

Or install individually:
```bash
pip install sentence-transformers pymilvus torch numpy
```

### Milvus Setup

#### Option 1: Local Milvus with Docker
```bash
# Download docker-compose.yml for Milvus
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

#### Option 2: Zilliz Cloud (Recommended for production)
1. Sign up at [zilliz.com](https://zilliz.com)
2. Create a cluster
3. Get your connection credentials
4. Update configuration with your endpoint and credentials

## 📋 Features

### Embedding Generation
- **Models Supported**: BGE-small/base/large, all-MiniLM, all-mpnet
- **Automatic Batching**: Efficient processing of large document sets
- **Dimension Detection**: Automatic embedding dimension detection

### Vector Storage
- **Milvus Integration**: High-performance vector database
- **Index Types**: IVF_FLAT, IVF_SQ8, HNSW
- **Similarity Metrics**: Inner Product (IP), Euclidean (L2)
- **Scalability**: Handles millions of vectors

### Search Capabilities
- **Text Search**: Query with natural language
- **Embedding Search**: Direct embedding vector search
- **Top-K Results**: Configurable result count
- **Metadata Filtering**: Filter by document properties

## 🔍 API Reference

### EmbeddingsManager

```python
class EmbeddingsManager:
    def __init__(embedding_model: str, milvus_config: Dict)
    def initialize_vector_store() -> bool
    def process_and_store_chunks(chunks: List[Chunk]) -> bool
    def search_similar_chunks(query_text: str, top_k: int) -> List[Dict]
    def search_similar_by_embedding(embedding: List[float], top_k: int) -> List[Dict]
    def get_stats() -> Dict[str, Any]
```

### MilvusVectorStore

```python
class MilvusVectorStore:
    def __init__(collection_name: str, host: str, port: str, ...)
    def connect() -> bool
    def create_collection() -> bool
    def create_index(index_type: str, metric_type: str) -> bool
    def insert_chunks(chunks: List[Dict]) -> bool
    def search_similar(query_embedding: List[float], top_k: int) -> List[Dict]
```

## 🔧 Configuration Reference

### Supported Embedding Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| BAAI/bge-small-en-v1.5 | 384 | Fast, good quality |
| BAAI/bge-base-en-v1.5 | 768 | Balanced speed/quality |
| BAAI/bge-large-en-v1.5 | 1024 | Best quality, slower |
| all-MiniLM-L6-v2 | 384 | General purpose |
| all-mpnet-base-v2 | 768 | High quality |

### Index Types

| Index | Best For | Memory Usage |
|-------|----------|--------------|
| IVF_FLAT | High accuracy | High |
| IVF_SQ8 | Balanced | Medium |
| HNSW | Fast search | High |

## 🐛 Troubleshooting

### Common Issues

**1. Milvus Connection Failed**
```
Error: Failed to connect to Milvus
```
- Check if Milvus is running: `docker ps`
- Verify host/port configuration
- Check firewall settings

**2. Import Error: pymilvus**
```
ModuleNotFoundError: No module named 'pymilvus'
```
```bash
pip install pymilvus
```

**3. Dimension Mismatch**
```
Error: Embedding dimension mismatch
```
- Ensure `embedding_dim` in config matches your model
- BGE-small: 384, BGE-base: 768, BGE-large: 1024

### Performance Tips

1. **Batch Processing**: Process chunks in batches of 32-128
2. **Index Selection**: Use IVF_FLAT for accuracy, IVF_SQ8 for memory efficiency
3. **Memory**: Allocate sufficient RAM (2GB+ for small collections)
4. **Network**: Use local Milvus for development, cloud for production

## 📊 Monitoring

### Collection Statistics
```python
stats = manager.get_stats()
print(f"Total entities: {stats['num_entities']}")
```

### Performance Metrics
- **Insert Speed**: ~1000 chunks/second
- **Search Speed**: <100ms for top-10 results
- **Memory Usage**: ~4 bytes per dimension per vector

## 🚀 Integration with Main Pipeline

The embeddings module is automatically integrated with the main processing pipeline in `main.py`. Chunks are:

1. **Generated** by TextChunker
2. **Processed** by PostProcessor  
3. **Embedded** by EmbeddingService
4. **Stored** (optionally in Milvus for future search)

For vector storage integration, modify main.py to use EmbeddingsManager instead of just EmbeddingService.