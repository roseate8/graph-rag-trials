"""
Chunk sampler with stratified sampling based on topic clustering.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add retrieval module to path
retrieval_path = Path(__file__).parent.parent.parent / "retrieval"
sys.path.insert(0, str(retrieval_path))

from retrieval.retrieval import MilvusRetriever

logger = logging.getLogger(__name__)


class ChunkSampler:
    """
    Samples chunks from Milvus collection using stratified sampling.
    
    Uses K-means clustering on embeddings to identify topics,
    then samples proportionally from each cluster.
    """
    
    def __init__(self, config, retriever: MilvusRetriever):
        """
        Initialize chunk sampler.
        
        Args:
            config: SyntheticEvalConfig instance
            retriever: MilvusRetriever instance (connected)
        """
        self.config = config
        self.retriever = retriever
        
        logger.info(f"Initialized ChunkSampler for collection: {config.collection_name}")
    
    def fetch_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Fetch all chunks from Milvus collection.
        
        Returns:
            List of chunk dictionaries with metadata
        """
        logger.info("Fetching all chunks from Milvus...")
        
        # Get collection stats first
        stats = self.retriever.get_collection_stats()
        total_entities = stats.get('num_entities', 0)
        
        logger.info(f"Collection has {total_entities} entities")
        
        # Query all chunks using empty vector search with high limit
        # Create a dummy query embedding
        dummy_query = "sample query for fetching all chunks"
        
        try:
            # Use the retriever's embedding service to generate query embedding
            query_embedding = self.retriever._get_query_embedding(dummy_query)
            
            # Search with high limit to get all chunks
            all_results = self.retriever.milvus_store.search_similar(
                query_embedding=query_embedding,
                top_k=min(total_entities, 16384),  # Milvus limit
                output_fields=[
                    "chunk_id", "doc_id", "content", "word_count", "section_path",
                    "chunk_type", "regions", "product_version", "folder_path",
                    "structural_metadata", "entity_metadata", "embedding"
                ]
            )
            
            logger.info(f"Fetched {len(all_results)} chunks from Milvus")
            
            # Convert to standard dictionary format
            chunks = []
            for result in all_results:
                chunk = {
                    'chunk_id': result.get('chunk_id', ''),
                    'doc_id': result.get('doc_id', ''),
                    'content': result.get('content', ''),
                    'word_count': result.get('word_count', 0),
                    'section_path': result.get('section_path', ''),
                    'chunk_type': result.get('chunk_type'),
                    'regions': result.get('regions'),
                    'product_version': result.get('product_version'),
                    'folder_path': result.get('folder_path'),
                    'structural_metadata': result.get('structural_metadata'),
                    'entity_metadata': result.get('entity_metadata'),
                    'embedding': result.get('embedding', []),
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
            raise
    
    def cluster_chunks(self, chunks: List[Dict], n_clusters: int = None) -> Dict[int, List[int]]:
        """
        Cluster chunks by topic using K-means on embeddings.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
            n_clusters: Number of clusters (default from config)
            
        Returns:
            Dictionary mapping cluster_id -> list of chunk indices
        """
        if n_clusters is None:
            n_clusters = self.config.num_clusters
        
        logger.info(f"Clustering {len(chunks)} chunks into {n_clusters} topics...")
        
        # Extract embeddings
        embeddings = []
        valid_indices = []
        
        for i, chunk in enumerate(chunks):
            emb = chunk.get('embedding')
            if emb and len(emb) > 0:
                embeddings.append(emb)
                valid_indices.append(i)
        
        if not embeddings:
            logger.error("No embeddings found in chunks!")
            raise ValueError("Chunks must have embeddings for clustering")
        
        logger.info(f"Found {len(embeddings)} chunks with embeddings")
        
        # Convert to numpy array
        X = np.array(embeddings, dtype=np.float32)
        
        # Perform K-means clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(embeddings)),
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(X)
        
        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in zip(valid_indices, labels):
            clusters[int(label)].append(idx)
        
        logger.info(f"Created {len(clusters)} clusters")
        for cluster_id, indices in clusters.items():
            logger.info(f"  Cluster {cluster_id}: {len(indices)} chunks")
        
        return dict(clusters)
    
    def stratified_sample(
        self,
        chunks: List[Dict],
        total_samples: int = None
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Perform stratified sampling across topic clusters.
        
        Args:
            chunks: List of all chunks
            total_samples: Number of samples to draw (default from config)
            
        Returns:
            Tuple of (sampled_chunks, sampling_stats)
        """
        if total_samples is None:
            total_samples = self.config.target_sample_size
        
        logger.info(f"Performing stratified sampling to select {total_samples} chunks...")
        
        # First, cluster the chunks
        clusters = self.cluster_chunks(chunks, self.config.num_clusters)
        
        # Calculate samples per cluster (proportional to cluster size)
        total_chunks = sum(len(indices) for indices in clusters.values())
        samples_per_cluster = {}
        
        for cluster_id, indices in clusters.items():
            proportion = len(indices) / total_chunks
            n_samples = int(proportion * total_samples)
            # Ensure at least 1 sample per cluster if possible
            samples_per_cluster[cluster_id] = max(1, n_samples)
        
        # Adjust if we're over/under the target
        current_total = sum(samples_per_cluster.values())
        if current_total > total_samples:
            # Reduce from largest clusters
            while current_total > total_samples:
                largest_cluster = max(samples_per_cluster, key=samples_per_cluster.get)
                if samples_per_cluster[largest_cluster] > 1:
                    samples_per_cluster[largest_cluster] -= 1
                    current_total -= 1
                else:
                    break
        elif current_total < total_samples:
            # Add to largest clusters
            while current_total < total_samples:
                largest_cluster = max(clusters, key=lambda c: len(clusters[c]))
                if samples_per_cluster[largest_cluster] < len(clusters[largest_cluster]):
                    samples_per_cluster[largest_cluster] += 1
                    current_total += 1
                else:
                    break
        
        # Sample from each cluster
        sampled_indices = []
        np.random.seed(42)  # For reproducibility
        
        for cluster_id, n_samples in samples_per_cluster.items():
            cluster_indices = clusters[cluster_id]
            
            # Sample without replacement
            if n_samples >= len(cluster_indices):
                # Take all if we need more than available
                sampled_indices.extend(cluster_indices)
            else:
                sampled = np.random.choice(
                    cluster_indices,
                    size=n_samples,
                    replace=False
                )
                sampled_indices.extend(sampled.tolist())
        
        # Get the sampled chunks
        sampled_chunks = [chunks[i] for i in sampled_indices]
        
        # Prepare sampling stats
        stats = {
            'total_chunks': len(chunks),
            'num_clusters': len(clusters),
            'target_samples': total_samples,
            'actual_samples': len(sampled_chunks),
            'samples_per_cluster': samples_per_cluster,
            'cluster_sizes': {cid: len(indices) for cid, indices in clusters.items()}
        }
        
        logger.info(f"Sampled {len(sampled_chunks)} chunks from {len(clusters)} clusters")
        logger.info(f"Samples per cluster: {samples_per_cluster}")
        
        return sampled_chunks, stats

