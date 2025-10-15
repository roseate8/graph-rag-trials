"""
Main orchestrator for synthetic evaluation dataset generation.

Usage:
    python -m evals.synthetic-eval.main
"""

import sys
import logging
import json
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
current_dir = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))
vector_ingest_path = project_root / "vector-ingest" / "src"
sys.path.insert(0, str(vector_ingest_path))

from config import SyntheticEvalConfig
from chunk_sampler import ChunkSampler
from fact_extractor import FactExtractor
from query_generator import QueryGenerator
from silver_labeler import SilverLabeler
from output_formatter import OutputFormatter

# Import dependencies
from chunking.processors.llm_utils import SecureAPIKeyManager
from retrieval.retrieval import MilvusRetriever


# Setup logging with tqdm compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose httpx logging during progress bars
logging.getLogger('httpx').setLevel(logging.WARNING)


def main():
    """Main entry point for synthetic evaluation dataset generation."""
    
    logger.info("=" * 80)
    logger.info("SYNTHETIC EVALUATION DATASET GENERATOR")
    logger.info("=" * 80)
    
    # 1. Load configuration
    logger.info("\n[1/6] Loading configuration...")
    config = SyntheticEvalConfig()
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Collection: {config.collection_name}")
    logger.info(f"  Target questions: {config.target_questions}")
    logger.info(f"  Target sample size: {config.target_sample_size}")
    
    # 2. Initialize LLM manager
    logger.info("\n[2/6] Initializing LLM...")
    llm_manager = SecureAPIKeyManager()
    logger.info("  LLM manager initialized (secure API key management)")
    
    # 3. Initialize Milvus retriever (read-only)
    logger.info("\n[3/6] Connecting to Milvus...")
    retriever = MilvusRetriever(
        embedding_model=config.embedding_model,
        milvus_profile=config.milvus_profile,
        collection_name=config.collection_name,
        enable_reranking=False
    )
    
    if not retriever.connect():
        logger.error("Failed to connect to Milvus!")
        return 1
    
    logger.info(f"  Connected to collection: {config.collection_name}")
    
    # Get collection stats
    stats = retriever.get_collection_stats()
    logger.info(f"  Total entities: {stats.get('num_entities', 'N/A')}")
    
    try:
        # 4. Sample chunks
        logger.info("\n[4/6] STEP 1: Sampling chunks...")
        sampler = ChunkSampler(config, retriever)
        
        logger.info("  Fetching all chunks from Milvus...")
        all_chunks = sampler.fetch_all_chunks()
        logger.info(f"  Fetched {len(all_chunks)} total chunks")
        
        logger.info("  Performing stratified sampling...")
        sampled_chunks, sampling_stats = sampler.stratified_sample(
            all_chunks,
            config.target_sample_size
        )
        logger.info(f"  Sampled {len(sampled_chunks)} chunks from {sampling_stats['num_clusters']} clusters")
        
        # 5. Extract facts
        logger.info("\n[5/6] STEP 2: Extracting atomic facts...")
        extractor = FactExtractor(config, llm_manager)
        
        all_facts = []
        fact_types = defaultdict(int)

        # Process chunks with parallel async processing and progress bar
        logger.info(f"  Processing {len(sampled_chunks)} chunks in parallel (concurrency=5)...")
        logger.info(f"  Progress: Processing in batches of 20, 5 concurrent LLM calls")

        # Create progress bar for individual chunks
        pbar = tqdm(total=len(sampled_chunks), desc="Extracting facts", unit="chunk", ncols=100)

        # Process in batches
        batch_size = 20  # Process 20 chunks at a time
        total_batches = (len(sampled_chunks) + batch_size - 1) // batch_size
        chunks_processed = 0

        for batch_idx in range(0, len(sampled_chunks), batch_size):
            batch = sampled_chunks[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1

            try:
                # Progress callback to update progress bar as chunks complete
                def progress_callback(completed_in_batch):
                    pbar.update(1)
                    pbar.set_postfix({"facts": len(all_facts), "batch": f"{batch_num}/{total_batches}"})

                # Extract facts from batch in parallel
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                batch_results = extractor.extract_facts_batch(batch, concurrency=5, progress_callback=progress_callback)
                logger.debug(f"Batch {batch_num}/{total_batches} completed")

                # Process results
                for idx, facts in enumerate(batch_results):
                    if isinstance(facts, Exception):
                        logger.warning(f"  Exception in chunk {batch[idx].get('chunk_id', 'unknown')}: {facts}")
                        continue  # Skip failed chunks
                    if isinstance(facts, list):
                        all_facts.extend(facts)
                        # Count fact types
                        for fact in facts:
                            fact_types[fact.fact_type] += 1

                chunks_processed += len(batch)

                # Limit total facts to prevent memory issues
                if len(all_facts) >= config.max_facts_per_chunk * len(sampled_chunks):
                    pbar.write(f"  Reached fact limit ({len(all_facts)} facts), stopping extraction")
                    break

            except Exception as e:
                logger.error(f"  Error processing batch {batch_num}: {e}", exc_info=True)
                # Still update progress bar for failed batch
                pbar.update(len(batch))
                continue

        pbar.close()
        
        logger.info(f"  Extracted {len(all_facts)} total facts")
        logger.info(f"  Average facts per chunk: {len(all_facts) / len(sampled_chunks):.1f}")
        logger.info(f"  Fact type distribution:")
        for fact_type, count in sorted(fact_types.items()):
            logger.info(f"    - {fact_type}: {count}")
        
        # Save intermediate facts if enabled
        if config.save_intermediate:
            # Ensure output directory exists
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            facts_path = Path(config.output_dir) / "intermediate_facts.jsonl"
            logger.info(f"  Saving facts to {facts_path}")
            with open(facts_path, 'w', encoding='utf-8') as f:
                for fact in all_facts:
                    f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + '\n')
        
        # 6. Generate queries
        logger.info("\n[6/6] STEP 3: Generating queries...")
        generator = QueryGenerator(config, llm_manager)
        
        queries, query_stats = generator.generate_all_queries(all_facts)
        
        logger.info(f"  Generated {len(queries)} total queries")
        logger.info(f"    - Single-hop: {query_stats.get('single_hop', 0)}")
        logger.info(f"    - Multi-hop: {query_stats.get('multi_hop', 0)}")
        
        # Save intermediate queries if enabled
        if config.save_intermediate:
            # Ensure output directory exists
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            queries_path = Path(config.output_dir) / "intermediate_queries.jsonl"
            logger.info(f"  Saving queries to {queries_path}")
            with open(queries_path, 'w', encoding='utf-8') as f:
                for query in queries:
                    f.write(json.dumps(query.to_dict(), ensure_ascii=False) + '\n')
        
        # 7. Assign silver labels
        logger.info("\nSTEP 4: Assigning silver labels...")
        labeler = SilverLabeler(config, llm_manager, retriever)
        
        qrels = labeler.batch_label_queries(queries, all_chunks)
        
        logger.info(f"  Labeled {len(qrels)} queries")
        
        # Compute label statistics
        label_stats = labeler.compute_label_statistics(qrels)
        
        # 8. Write output
        logger.info("\nSTEP 5: Writing output files...")
        formatter = OutputFormatter(config.output_dir)
        
        # Combine all stats
        all_stats = {
            'sampling': sampling_stats,
            'fact_extraction': {
                'total_facts': len(all_facts),
                'avg_facts_per_chunk': len(all_facts) / len(sampled_chunks),
                'fact_types': dict(fact_types)
            },
            'query_generation': query_stats,
            'silver_labeling': label_stats
        }
        
        # Write all files
        output_files = formatter.write_all(queries, qrels, all_chunks, all_stats)
        
        logger.info(f"\nOutput files written to: {config.output_dir}")
        for file_type, file_path in output_files.items():
            logger.info(f"  - {file_type}: {Path(file_path).name}")
        
        # 9. Optional: Validate retrieval
        if config.validate_retrieval:
            logger.info("\nSTEP 6: Validating retrieval...")
            validate_retrieval(queries, retriever, config.validation_top_k)
        
        logger.info("\n" + "=" * 80)
        logger.info("GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"\nSummary:")
        logger.info(f"  Total chunks: {len(all_chunks)}")
        logger.info(f"  Sampled chunks: {len(sampled_chunks)}")
        logger.info(f"  Extracted facts: {len(all_facts)}")
        logger.info(f"  Generated queries: {len(queries)}")
        logger.info(f"  Output directory: {config.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nGeneration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nError during generation: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        retriever.disconnect()
        logger.info("Disconnected from Milvus")


def validate_retrieval(queries, retriever, top_k=10):
    """
    Validate that gold chunks are retrievable.
    
    Args:
        queries: List of Query objects
        retriever: MilvusRetriever instance
        top_k: Number of top results to retrieve
    """
    logger.info(f"Validating retrieval for {len(queries)} queries (top-{top_k})...")
    
    recall_scores = []
    
    for i, query in enumerate(queries):
        if i % 50 == 0:
            logger.info(f"  Validating query {i}/{len(queries)}...")
        
        # Retrieve top-k
        results = retriever.retrieve(query.query_text, top_k=top_k, min_similarity=0.0)
        retrieved_ids = [r.chunk_id for r in results]
        
        # Compute recall
        gold_retrieved = [cid for cid in query.gold_chunk_ids if cid in retrieved_ids]
        recall = len(gold_retrieved) / len(query.gold_chunk_ids) if query.gold_chunk_ids else 0.0
        
        recall_scores.append(recall)
    
    # Compute average recall
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    # Compute recall@k metrics
    recall_1 = sum(1 for r in recall_scores if r == 1.0) / len(recall_scores) * 100
    recall_partial = sum(1 for r in recall_scores if r > 0) / len(recall_scores) * 100
    
    logger.info(f"\nRetrieval Validation Results:")
    logger.info(f"  Average Recall@{top_k}: {avg_recall:.2%}")
    logger.info(f"  Queries with perfect recall (100%): {recall_1:.1f}%")
    logger.info(f"  Queries with partial recall (>0%): {recall_partial:.1f}%")
    
    if avg_recall < 0.5:
        logger.warning("  ⚠ Low recall detected! Consider increasing top_k or reviewing query generation.")


if __name__ == "__main__":
    sys.exit(main())

