# 🚀 Milvus Cloud Integration Guide

Complete setup guide for integrating your Graph-RAG system with Milvus Cloud vector database.

## 📋 Prerequisites

1. **Milvus Cloud Account**: $100 free credits
2. **Python Environment**: Your existing vector-ingest setup
3. **Processed Chunks**: Run `python main.py` first to generate embeddings

## 🏗️ Step 1: Create Milvus Cluster

1. **Log into Milvus Cloud Dashboard**
2. **Create a new cluster** (choose free tier)
3. **Note your connection details**:
   - **Cluster Endpoint**: `https://in03-xxx.api.gcp-us-west1.zillizcloud.com`
   - **API Token**: Generated in the dashboard
   - **Database**: Usually `default`

## 🔧 Step 2: Install Dependencies

```bash
# Install Milvus Python SDK
pip install pymilvus>=2.3.0

# Install environment variable support
pip install python-dotenv
```

## ⚙️ Step 3: Configure Environment

Create `.env` file in your `vector-ingest` directory:

```bash
# Copy example file
cp .env.example .env
```

Edit `.env` with your Milvus credentials:

```env
# Milvus Cloud Configuration
MILVUS_ENDPOINT=https://your-cluster-endpoint
MILVUS_TOKEN=your_api_token_here
MILVUS_COLLECTION_NAME=document_chunks
MILVUS_DATABASE=default
```

## 🎯 Step 4: Setup Collection

Use the automated setup script:

```bash
# Setup Milvus collection and index
python setup_milvus.py --setup
```

This will:
- ✅ Connect to your Milvus cluster
- ✅ Create `document_chunks` collection with proper schema
- ✅ Create HNSW index for fast similarity search

## 📤 Step 5: Upload Your Chunks

### Option A: Upload during processing (Recommended)

```bash
# Process documents AND upload to Milvus
python main.py --milvus
```

### Option B: Upload existing chunks

```bash
# Upload previously processed chunks
python setup_milvus.py --upload
```

## 🔍 Step 6: Test Similarity Search

```bash
# Test search functionality
python setup_milvus.py --test
```

Expected output:
```
🔍 Testing similarity search...
Query: 'What are the main risk factors for the business?'
✅ Found 3 similar chunks:

1. Score: 0.856
   Chunk: Form 10-K 2025_chunk_15
   Content: Risk factors include market competition, regulatory changes...

2. Score: 0.823
   Chunk: Form 10-K 2025_chunk_42
   Content: The company faces risks related to cybersecurity threats...

3. Score: 0.791
   Chunk: Form 10-K 2022_chunk_18
   Content: Economic downturns may significantly impact our revenue...
```

## 📊 Step 7: Monitor Your Collection

```bash
# View collection statistics
python setup_milvus.py --stats
```

Output shows:
- Collection name and entity count
- Index information
- Storage usage

## 🔄 Complete Workflow

```bash
# 1. Process documents with embeddings
python main.py --verbose

# 2. Setup Milvus (first time only)
python setup_milvus.py --setup

# 3. Upload chunks to vector store
python main.py --milvus

# 4. Test similarity search
python setup_milvus.py --test
```

## 📈 Usage Examples

### Basic Processing + Upload
```bash
python main.py --milvus --verbose
```

### Clear Cache + Reprocess + Upload
```bash
python main.py --clear-cache --milvus
```

### Upload Only (if chunks already exist)
```bash
python setup_milvus.py --upload
```

## 🏗️ Collection Schema

Your Milvus collection uses this optimized schema:

| Field | Type | Description |
|-------|------|-------------|
| `id` | INT64 | Auto-generated primary key |
| `chunk_id` | VARCHAR(512) | Unique chunk identifier |
| `doc_id` | VARCHAR(256) | Source document ID |
| `content` | VARCHAR(65535) | Chunk text content |
| `word_count` | INT64 | Number of words in chunk |
| `section_path` | VARCHAR(1024) | Document section path |
| `embedding` | FLOAT_VECTOR(384) | BGE-small-en-v1.5 embedding |

## 🚀 Performance Optimization

- **Index Type**: HNSW for fast approximate nearest neighbor search
- **Metric**: Inner Product (IP) for cosine similarity
- **Batch Size**: 100 chunks per upload batch
- **Search Params**: Optimized for accuracy vs speed balance

## 🔧 Troubleshooting

### Connection Issues
```bash
# Check your endpoint and token
python -c "import os; print(f'Endpoint: {os.getenv(\"MILVUS_ENDPOINT\")}'); print(f'Token: {os.getenv(\"MILVUS_TOKEN\")[:10]}...')"
```

### Collection Already Exists
```bash
# Delete and recreate collection
python -c "
from embeddings.vector_store import MilvusVectorStore
from embeddings.config import MilvusConfig
vs = MilvusVectorStore(**MilvusConfig.get_cloud_config())
vs.connect()
vs.delete_collection()
"
```

### Low Search Quality
- Ensure embeddings are properly generated
- Check if index was created correctly
- Verify similar documents exist in your collection

## 💡 Next Steps

1. **Integrate with RAG Pipeline**: Use search results for context retrieval
2. **Add Metadata Filtering**: Filter by document type, date, etc.
3. **Implement Hybrid Search**: Combine semantic + keyword search
4. **Monitor Costs**: Track usage in Milvus Cloud dashboard

## 📚 Additional Resources

- [Milvus Documentation](https://milvus.io/docs)
- [PyMilvus SDK Reference](https://milvus.io/api-reference/pymilvus)
- [Zilliz Cloud Console](https://cloud.zilliz.com/)

Your Graph-RAG system is now powered by enterprise-grade vector search! 🎉