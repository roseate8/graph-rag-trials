# Milvus Standalone Vector Storage

This module provides Milvus standalone integration for vector storage and similarity search.

## Quick Start

### 1. Install Dependencies
```bash
python src/embeddings/milvus_setup.py install
```

### 2. Start Milvus with Docker
```bash
# Option A: Using setup script
python src/embeddings/milvus_setup.py start

# Option B: Using Docker Compose (recommended)
docker-compose up -d milvus-standalone
```

### 3. Initialize Collection
```bash
python src/embeddings/milvus_setup.py init
```

### 4. Upload Existing Embeddings
```bash
# First generate embeddings with main pipeline
python main.py

# Then upload to Milvus
python src/embeddings/milvus_setup.py upload
```

### 5. Test Search
```bash
python src/embeddings/milvus_setup.py search
```

## Usage in Code

### Simple Embedding Generation (No Milvus)
```python
from src.embeddings import create_embeddings_manager

# Just generate embeddings
manager = create_embeddings_manager(enable_milvus=False)
embedded_chunks = manager.process_chunks(chunks)
```

### With Milvus Vector Storage
```python
from src.embeddings import create_embeddings_manager

# Generate embeddings and store in Milvus
with create_embeddings_manager(enable_milvus=True) as manager:
    # Connect and initialize
    manager.connect_milvus()
    manager.initialize_milvus()
    
    # Process chunks (generates embeddings + stores in Milvus)
    embedded_chunks = manager.process_chunks(chunks)
    
    # Search for similar content
    results = manager.search_similar("What are the main risks?", top_k=5)
```

## Docker Services

The `docker-compose.yml` includes:

- **milvus-standalone**: Main Milvus server (port 19530)
- **etcd**: Metadata storage for Milvus  
- **minio**: Object storage for Milvus
- **attu**: Web UI for Milvus management (port 3000)

## Configuration Profiles

- **development**: Simple IVF_FLAT index, smaller parameters
- **production**: HNSW index for best performance (default)
- **testing**: FLAT index (no index) for small test datasets

## Management Commands

```bash
# Install dependencies
python src/embeddings/milvus_setup.py install

# Start/stop Milvus
python src/embeddings/milvus_setup.py start
python src/embeddings/milvus_setup.py stop

# Initialize collection and index
python src/embeddings/milvus_setup.py init

# Upload embeddings from processed chunks
python src/embeddings/milvus_setup.py upload

# Test similarity search
python src/embeddings/milvus_setup.py search

# Show statistics
python src/embeddings/milvus_setup.py stats

# Health check
python src/embeddings/milvus_setup.py health
```

## Web Interfaces

- **Attu (Milvus GUI)**: http://localhost:3000
- **MinIO Console**: http://localhost:9001 (admin/minioadmin)
- **Milvus Health**: http://localhost:9091/healthz

## Integration with Main Pipeline

The main.py pipeline can optionally use Milvus by importing from the embeddings module:

```python
from src.embeddings import create_embeddings_manager

# In main.py, replace EmbeddingService with EmbeddingsManager
embeddings_manager = create_embeddings_manager(enable_milvus=True)
```

This keeps main.py clean while providing optional Milvus functionality.