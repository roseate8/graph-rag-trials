"""
Chunk sampler with stratified sampling based on topic clustering.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
        Fetch all chunks from Milvus collection efficiently.
        
        Returns:
            List of chunk dictionaries with metadata
        """
        logger.info("Fetching all chunks from Milvus...")
        
        # Get collection stats first
        stats = self.retriever.get_collection_stats()
        total_entities = stats.get('num_entities', 0)
        
        logger.info(f"Collection has {total_entities} entities")
        
        try:
            # Use a simpler approach - multiple small queries with different search terms
            search_terms = [
                "revenue", "financial", "elastic", "data", "search", "security", 
                "product", "customer", "growth", "technology", "platform", "cloud"
            ]
            
            all_results = []
            max_per_query = 100  # Small batches to avoid ef parameter issues
            
            for term in search_terms:
                try:
                    logger.debug(f"Searching for '{term}'...")
                    results = self.retriever.retrieve(
                        query=term,
                        top_k=max_per_query,
                        min_similarity=0.0
                    )
                    
                    # Convert RetrievedChunk objects to dictionaries
                    for chunk in results:
                        chunk_dict = {
                            'chunk_id': chunk.chunk_id,
                            'doc_id': chunk.doc_id,
                            'content': chunk.content,
                            'word_count': chunk.word_count,
                            'section_path': chunk.section_path,
                            'chunk_type': chunk.chunk_type,
                            'regions': chunk.regions,
                            'product_version': chunk.product_version,
                            'folder_path': chunk.folder_path,
                            'structural_metadata': chunk.structural_metadata,
                            'entity_metadata': chunk.entity_metadata,
                            'embedding': [],  # We'll get embeddings later if needed
                        }
                        all_results.append(chunk_dict)
                    
                    if len(all_results) >= self.config.target_sample_size * 5:  # Get 5x more than needed
                        break
                        
                except Exception as e:
                    logger.warning(f"Error searching for '{term}': {e}")
                    continue
            
            logger.info(f"Fetched {len(all_results)} chunks from Milvus")
            
            # Remove duplicates by chunk_id
            seen_ids = set()
            chunks = []
            for result in all_results:
                chunk_id = result.get('chunk_id', '')
                if chunk_id and chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    chunks.append(result)
            
            logger.info(f"Processed {len(chunks)} chunks with embeddings")
            return chunks
            
        except Exception as e:
            logger.error(f"Error fetching chunks: {e}")
            raise
    
    def cluster_chunks(self, chunks: List[Dict], n_clusters: int = None) -> Dict[int, List[int]]:
        """
        Cluster chunks by topic using simple heuristics (no embeddings needed).
        
        Args:
            chunks: List of chunk dictionaries
            n_clusters: Number of clusters (default from config)
            
        Returns:
            Dictionary mapping cluster_id -> list of chunk indices
        """
        if n_clusters is None:
            n_clusters = self.config.num_clusters
        
        logger.info(f"Clustering {len(chunks)} chunks into {n_clusters} topics using content-based clustering...")
        
        if not chunks:
            logger.error("No chunks provided for clustering!")
            raise ValueError("Chunks must be provided for clustering")
        
        # Simple clustering based on document ID and section path
        clusters = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            # Use doc_id and section_path for simple clustering
            doc_id = chunk.get('doc_id', 'unknown')
            section = chunk.get('section_path', 'unknown')
            
            # Create a simple hash-based cluster assignment
            cluster_key = hash(f"{doc_id}_{section}") % n_clusters
            clusters[cluster_key].append(i)
        
        # Ensure we have at least some distribution
        if len(clusters) < n_clusters:
            # Redistribute chunks more evenly
            all_indices = list(range(len(chunks)))
            clusters = defaultdict(list)
            for i, idx in enumerate(all_indices):
                cluster_id = i % n_clusters
                clusters[cluster_id].append(idx)
        
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

