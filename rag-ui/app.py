"""
Flask backend API for RAG UI - connects to naive RAG system.
"""

import sys
import os
import json
import logging
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from flask import Flask, request, jsonify, Response, send_from_directory, send_file
from flask_cors import CORS
import threading
import queue

# Suppress protobuf warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# Add paths to sys.path - use absolute paths to handle Streamlit working directory changes
CURRENT_FILE = Path(__file__).absolute()
PROJECT_ROOT = CURRENT_FILE.parent.parent
NAIVE_RAG_PATH = PROJECT_ROOT / "naive-rag"
VECTOR_INGEST_PATH = PROJECT_ROOT / "vector-ingest" / "src"

sys.path.insert(0, str(NAIVE_RAG_PATH))
sys.path.insert(0, str(VECTOR_INGEST_PATH))

# Import RAG system
from core import RAGSystem, create_rag_system
from llm import SecureOpenAIClient, MockLLMClient
from chunking.processors.llm_utils import set_openai_api_key, has_openai_api_key

# Setup logging to capture RAG system logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global RAG system instance
rag_system: Optional[RAGSystem] = None
conversation_history_file = Path("conversation_history.json")

# Thread-safe logging queue for real-time log streaming
log_queue = queue.Queue()


class LogCapture(logging.Handler):
    """Custom logging handler to capture logs for real-time streaming."""
    
    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),  # Use getMessage() instead of format()
                "logger": record.name
            }
            log_queue.put(log_entry)
            print(f"LOG CAPTURED: [{record.levelname}] {record.getMessage()}")  # Debug print
        except Exception as e:
            print(f"Error capturing log: {e}")


# Add log capture handler
log_capture = LogCapture()
log_capture.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Add to all relevant loggers to capture all terminal output
loggers_to_capture = [
    '__main__',  # Flask app logs
    'core',      # RAG system logs
    'llm',       # LLM client logs
    'retrieval', # Retrieval logs
    'embeddings.milvus_store',  # Milvus logs
    'embeddings.embedding_service',  # Embedding logs
]

for logger_name in loggers_to_capture:
    logger_instance = logging.getLogger(logger_name)
    logger_instance.addHandler(log_capture)
    logger_instance.setLevel(logging.DEBUG)  # Ensure we capture all levels

# Also add to root logger to catch anything else
root_logger = logging.getLogger()
root_logger.addHandler(log_capture)


def load_conversation_history() -> List[Dict]:
    """Load conversation history from JSON file."""
    if not conversation_history_file.exists():
        return []
    
    try:
        with open(conversation_history_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")
        return []


def get_resource_info() -> Dict[str, Any]:
    """Get current resource usage information."""
    try:
        import psutil
        
        # Check GPU availability
        gpu_info = {
            "gpu_available": False,
            "gpuutil_not_installed": True
        }
        
        # Try to get GPU info if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = {
                    "gpu_available": True,
                    "gpuutil_not_installed": False,
                    "gpu_count": len(gpus),
                    "gpu_memory_used": gpus[0].memoryUsed,
                    "gpu_memory_total": gpus[0].memoryTotal
                }
        except ImportError:
            pass  # Keep default GPU info
        
        # Get system memory and CPU
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            "memory_used_mb": round(memory.used / 1024 / 1024, 1),
            "memory_total_mb": round(memory.total / 1024 / 1024, 1),
            "memory_percent": round(memory.percent, 1),
            "cpu_percent": round(cpu_percent, 1),
            "gpu": gpu_info
        }
    except Exception as e:
        logger.error(f"Error getting resource info: {e}")
        return {
            "memory_used_mb": 0.0,
            "memory_total_mb": 0.0,
            "memory_percent": 0.0,
            "cpu_percent": 0.0,
            "gpu": {"gpu_available": False, "gpuutil_not_installed": True}
        }


def save_conversation_entry(query: str, result_data: Dict, initial_resources: Dict, final_resources: Dict, timing_data: Dict, enable_reranking: bool = False):
    """Save a conversation entry to history in the exact format specified."""
    try:
        history = load_conversation_history()
        
        # Calculate peak resources
        peak_memory_mb = max(initial_resources.get("memory_used_mb", 0), final_resources.get("memory_used_mb", 0))
        peak_cpu_percent = max(initial_resources.get("cpu_percent", 0), final_resources.get("cpu_percent", 0))
        
        # Create the entry in the exact format specified
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-3] + "Z",
            "query": query,
            "result": {
                "retrieved_chunks": result_data.get("retrieved_chunks", []),
                "context_length": result_data.get("context_length", 0),
                "llm_response": {
                    "query": query,
                    "method": "layout_aware_chunking",
                    "context": result_data.get("context", ""),
                    "answer": result_data.get("response", ""),
                    "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                    "tokens_used": result_data.get("response_tokens", 0)
                },
                "timing": {
                    "vector_search_time": timing_data.get("retrieval_time", 0),
                    "rerank_time": timing_data.get("retrieval_time", 0) if enable_reranking else 0.0,
                    "context_time": timing_data.get("context_time", 0),
                    "llm_time": timing_data.get("generation_time", 0),
                    "total_time": timing_data.get("total_time", 0)
                },
                "resources": {
                    "initial": initial_resources,
                    "final": final_resources,
                    "peak_memory_mb": peak_memory_mb,
                    "peak_cpu_percent": peak_cpu_percent
                }
            },
            "tokens_used": result_data.get("response_tokens", 0),
            "timing": {
                "vector_search_time": timing_data.get("retrieval_time", 0),
                "rerank_time": timing_data.get("retrieval_time", 0) if enable_reranking else 0.0,
                "context_time": timing_data.get("context_time", 0),
                "llm_time": timing_data.get("generation_time", 0),
                "total_time": timing_data.get("total_time", 0)
            },
            "resources": {
                "initial": initial_resources,
                "final": final_resources,
                "peak_memory_mb": peak_memory_mb,
                "peak_cpu_percent": peak_cpu_percent
            }
        }
        
        history.append(entry)
        
        # Keep only last 100 conversations
        if len(history) > 100:
            history = history[-100:]
        
        with open(conversation_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved conversation entry to {conversation_history_file}")
    
    except Exception as e:
        logger.error(f"Error saving conversation history: {e}")


def init_rag_system():
    """Initialize the RAG system with re-ranking enabled."""
    global rag_system
    
    try:
        logger.info("Initializing RAG system with re-ranking...")
        rag_system = create_rag_system(
            llm_type="openai", 
            enable_reranking=True,
            retrieval_multiplier=10  # 10*K initial retrieval for re-ranking
        )
        
        if not rag_system.connect():
            logger.error("Failed to connect to Milvus")
            return False
        
        stats = rag_system.get_system_stats()
        num_entities = stats.get('retriever_stats', {}).get('num_entities', 0)
        rerank_config = stats.get('reranking_config', {})
        
        logger.info(f"Connected to collection with {num_entities} document chunks")
        logger.info(f"Re-ranking enabled: {rerank_config.get('enabled', False)}")
        logger.info(f"Retrieval multiplier: {rerank_config.get('retrieval_multiplier', 'N/A')}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        return False


# Static file serving routes
@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_file('index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files (CSS, JS, images)."""
    return send_from_directory('.', filename)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "rag_connected": rag_system is not None and rag_system.connected if rag_system else False,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/system-stats', methods=['GET'])
def get_system_stats():
    """Get RAG system statistics."""
    global rag_system
    
    # Initialize RAG system on first use if not already initialized
    if not rag_system:
        logger.info("Initializing RAG system for system stats...")
        if not init_rag_system():
            return jsonify({"error": "Failed to initialize RAG system"}), 503
    
    if not rag_system or not rag_system.connected:
        return jsonify({"error": "RAG system not connected"}), 503
    
    try:
        stats = rag_system.get_system_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/validate-api-key', methods=['POST'])
def validate_api_key():
    """Validate and store OpenAI API key."""
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        
        logger.info(f"Received API key validation request with key length: {len(api_key) if api_key else 0}")
        
        if not api_key:
            return jsonify({"error": "API key is required"}), 400
        
        # Debug: Log the first few characters to verify format
        logger.info(f"API key starts with: {api_key[:10]}...")
        
        # Store the API key securely
        set_openai_api_key(api_key)
        logger.info("API key stored successfully")
        
        # Test the API key with a simple request
        test_client = SecureOpenAIClient(model="gpt-3.5-turbo")
        if test_client.can_generate():
            logger.info("API key validated successfully")
            return jsonify({"status": "valid", "message": "API key is valid"})
        else:
            logger.warning("API key validation failed - can_generate() returned False")
            return jsonify({"error": "Invalid API key"}), 400
    
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a RAG query with re-ranking support."""
    global rag_system
    
    # Initialize RAG system on first use
    if not rag_system:
        logger.info("Initializing RAG system on first query...")
        if not init_rag_system():
            return jsonify({"error": "Failed to initialize RAG system"}), 503
    
    if not rag_system or not rag_system.connected:
        return jsonify({"error": "RAG system not connected"}), 503
    
    if not has_openai_api_key():
        return jsonify({"error": "OpenAI API key not configured"}), 400
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 10)  # Default to 10 for re-ranking
        model = data.get('model', 'gpt-4o-mini')
        temperature = data.get('temperature', 0.1)
        enable_reranking = data.get('enable_reranking', True)  # Default enabled
        retrieval_multiplier = data.get('retrieval_multiplier', 10)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        logger.info(f"Processing query: {query[:50]}...")
        logger.info(f"Re-ranking: {'enabled' if enable_reranking else 'disabled'}, multiplier: {retrieval_multiplier}")
        
        # Update RAG system configuration for this query
        rag_system.enable_reranking = enable_reranking
        rag_system.retrieval_multiplier = retrieval_multiplier
        rag_system.retriever.enable_reranking = enable_reranking
        
        # Update LLM client with user preferences
        rag_system.llm_client = SecureOpenAIClient(
            model=model,
            temperature=temperature
        )
        
        # Capture initial resources
        initial_resources = get_resource_info()
        start_time = time.time()
        
        # Process the query
        result = rag_system.query(query, top_k=top_k)
        
        total_time = time.time() - start_time
        final_resources = get_resource_info()
        
        # Format sources for UI
        sources = []
        retrieved_chunks_data = []
        for chunk in result.retrieved_chunks:
            # Create comprehensive metadata dict from chunk attributes
            metadata = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "word_count": chunk.word_count,
                "section_path": chunk.section_path,
                "chunk_type": getattr(chunk, 'chunk_type', 'unknown'),
                "regions": getattr(chunk, 'regions', None),
                "product_version": getattr(chunk, 'product_version', None),
                "folder_path": getattr(chunk, 'folder_path', None),
                "structural_metadata": getattr(chunk, 'structural_metadata', None),
                "entity_metadata": getattr(chunk, 'entity_metadata', None)
            }

            # DEBUG: Add mock metadata to test UI display
            if metadata.get("regions") is None:
                metadata["regions"] = ["United States", "Europe"]
            if metadata.get("product_version") is None:
                metadata["product_version"] = "v2.4"
            if metadata.get("folder_path") is None:
                metadata["folder_path"] = ["financial-reports", "annual-reports"]
            if metadata.get("structural_metadata") is None:
                metadata["structural_metadata"] = {
                    "element_type": "paragraph",
                    "page_number": 15,
                    "is_heading": False
                }
            if metadata.get("entity_metadata") is None:
                metadata["entity_metadata"] = {
                    "organizations": ["Elastic N.V.", "SEC"],
                    "locations": ["San Francisco", "Amsterdam"],
                    "financial_metrics": ["revenue", "COGS"]
                }
            
            source = {
                "snippet": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "full_content": chunk.content,
                "score": chunk.similarity_score,
                "rerank_score": getattr(chunk, 'rerank_score', None),
                "rerank_probability": getattr(chunk, 'rerank_probability', None),
                "metadata": metadata,
                "chunk_type": metadata.get('chunk_type', 'unknown')
            }
            sources.append(source)
            
            # Data for conversation history
            retrieved_chunks_data.append({
                "content": chunk.content,
                "score": chunk.similarity_score,
                "metadata": metadata
            })
        
        # Prepare metrics for UI with re-ranking info
        metrics = {
            "total_time": f"{total_time:.2f}s",
            "tokens_used": f"{result.context_token_count + (result.response_tokens or 0)}",
            "cpu_peak": f"{final_resources.get('cpu_percent', 0):.1f}%",
            "memory_peak": f"{final_resources.get('memory_used_mb', 0):.1f} MB",
            "retrieval_time": f"{result.retrieval_time:.2f}s" if result.retrieval_time else "N/A",
            "generation_time": f"{result.generation_time:.2f}s" if result.generation_time else "N/A",
            "model": result.model_used,
            "reranking_enabled": enable_reranking,
            "initial_chunks": top_k * retrieval_multiplier if enable_reranking else top_k,
            "final_chunks": len(result.retrieved_chunks),
            "retrieval_method": f"Re-ranked ({top_k * retrieval_multiplier} → {top_k})" if enable_reranking else f"Direct ({top_k})"
        }
        
        # Prepare data for conversation history in the exact format
        result_data = {
            "retrieved_chunks": retrieved_chunks_data,
            "context_length": result.context_token_count,
            "response": result.response,
            "response_tokens": result.response_tokens or 0
        }
        
        timing_data = {
            "retrieval_time": result.retrieval_time or 0,
            "context_time": 0,  # Not tracked separately in current implementation
            "generation_time": result.generation_time or 0,
            "total_time": total_time
        }
        
        # Save to conversation history in the exact format
        save_conversation_entry(query, result_data, initial_resources, final_resources, timing_data, enable_reranking)
        
        response_data = {
            "response": result.response,
            "sources": sources,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Query processed successfully in {total_time:.2f}s")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500


@app.route('/api/logs/stream')
def stream_logs():
    """Stream logs in real-time using Server-Sent Events."""
    def generate():
        while True:
            try:
                # Get log entry with shorter timeout for more responsiveness
                log_entry = log_queue.get(timeout=5)
                yield f"data: {json.dumps(log_entry)}\n\n"
            except queue.Empty:
                # Send heartbeat more frequently
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
            except Exception as e:
                logger.error(f"Error in log stream: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
    })


@app.route('/api/conversation-history', methods=['GET'])
def get_conversation_history():
    """Get conversation history."""
    try:
        history = load_conversation_history()
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error loading conversation history: {e}")
        return jsonify({"error": "Failed to load conversation history"}), 500


@app.route('/api/conversation-history/<int:conversation_id>', methods=['GET'])
def get_conversation_details(conversation_id):
    """Get specific conversation details."""
    try:
        history = load_conversation_history()
        conversation = next((c for c in history if c['id'] == conversation_id), None)
        
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify(conversation)
    except Exception as e:
        logger.error(f"Error loading conversation details: {e}")
        return jsonify({"error": "Failed to load conversation details"}), 500


if __name__ == '__main__':
    # Start server - RAG system will be initialized on first query
    logger.info("Starting RAG API server...")
    logger.info("RAG system will be initialized on first query to improve startup time")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)