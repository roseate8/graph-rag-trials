"""
Main orchestrator for synthetic evaluation dataset generation.

Usage:
    python -m evals.synthetic-eval.main
    python -m evals.synthetic-eval.main --skip-sampling --skip-facts
    python -m evals.synthetic-eval.main --only-queries --input-facts output/intermediate_facts.jsonl
    python -m evals.synthetic-eval.main --only-labeling --input-queries output/intermediate_queries.jsonl
"""

import sys
import logging
import json
import argparse
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Synthetic evaluation dataset generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python -m evals.synthetic-eval.main

  # Run with custom query targets
  python -m evals.synthetic-eval.main --target-questions 400 --multi-hop-ratio 0.25

  # Only regenerate queries from existing facts with new targets
  python -m evals.synthetic-eval.main --only-queries --input-facts output/intermediate_facts.jsonl --target-questions 400 --multi-hop-ratio 0.25

  # Only redo silver labeling from existing queries
  python -m evals.synthetic-eval.main --only-labeling --input-queries output/intermediate_queries.jsonl
        """
    )

    # Skip flags
    parser.add_argument('--skip-sampling', action='store_true',
                        help='Skip chunk sampling step (requires existing sampled chunks or --input-facts)')
    parser.add_argument('--skip-facts', action='store_true',
                        help='Skip fact extraction step (requires --input-facts)')
    parser.add_argument('--skip-queries', action='store_true',
                        help='Skip query generation step (requires --input-queries)')

    # Input file paths
    parser.add_argument('--input-facts', type=str,
                        help='Path to existing facts JSONL file (relative to synthetic-eval dir or absolute)')
    parser.add_argument('--input-queries', type=str,
                        help='Path to existing queries JSONL file (relative to synthetic-eval dir or absolute)')
    parser.add_argument('--input-chunks', type=str,
                        help='Path to existing chunks JSONL file (relative to synthetic-eval dir or absolute)')

    # Only flags (shortcuts)
    parser.add_argument('--only-queries', action='store_true',
                        help='Only run query generation (implies --skip-sampling --skip-facts)')
    parser.add_argument('--only-labeling', action='store_true',
                        help='Only run silver labeling (implies --skip-sampling --skip-facts --skip-queries)')

    # Config overrides
    parser.add_argument('--target-questions', type=int,
                        help='Override target_questions from config (e.g., 400)')
    parser.add_argument('--multi-hop-ratio', type=float,
                        help='Override multi_hop_ratio from config (e.g., 0.25 for 25%%)')

    args = parser.parse_args()

    # Handle "only" shortcuts
    if args.only_labeling:
        args.skip_sampling = True
        args.skip_facts = True
        args.skip_queries = True
    elif args.only_queries:
        args.skip_sampling = True
        args.skip_facts = True

    # Validation
    if args.skip_facts and not args.input_facts:
        parser.error('--skip-facts requires --input-facts')
    if args.skip_queries and not args.input_queries:
        parser.error('--skip-queries requires --input-queries')
    if args.multi_hop_ratio is not None and not (0.0 <= args.multi_hop_ratio <= 1.0):
        parser.error('--multi-hop-ratio must be between 0.0 and 1.0')

    return args


def load_facts_from_file(file_path: Path) -> list:
    """Load facts from JSONL file."""
    from fact_extractor import AtomicFact

    logger.info(f"Loading facts from {file_path}")
    facts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                fact_dict = json.loads(line)
                # Reconstruct AtomicFact object
                fact = AtomicFact(**fact_dict)
                facts.append(fact)

    logger.info(f"Loaded {len(facts)} facts")
    return facts


def load_queries_from_file(file_path: Path) -> list:
    """Load queries from JSONL file."""
    from query_generator import Query

    logger.info(f"Loading queries from {file_path}")
    queries = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                query_dict = json.loads(line)
                # Reconstruct Query object
                query = Query(**query_dict)
                queries.append(query)

    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_chunks_from_file(file_path: Path) -> list:
    """Load chunks from JSONL file."""
    logger.info(f"Loading chunks from {file_path}")
    chunks = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line)
                chunks.append(chunk)

    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def main():
    """Main entry point for synthetic evaluation dataset generation."""

    logger.info("=" * 80)
    logger.info("SYNTHETIC EVALUATION DATASET GENERATOR")
    logger.info("=" * 80)

    # Parse command line arguments
    args = parse_args()

    # 1. Load configuration
    logger.info("\n[1/6] Loading configuration...")
    config = SyntheticEvalConfig()

    # Apply CLI overrides
    if args.target_questions is not None:
        logger.info(f"  Overriding target_questions: {config.target_questions} -> {args.target_questions}")
        config.target_questions = args.target_questions
    if args.multi_hop_ratio is not None:
        logger.info(f"  Overriding multi_hop_ratio: {config.multi_hop_ratio} -> {args.multi_hop_ratio}")
        config.multi_hop_ratio = args.multi_hop_ratio

    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Collection: {config.collection_name}")
    logger.info(f"  Target questions: {config.target_questions}")
    logger.info(f"  Multi-hop ratio: {config.multi_hop_ratio}")
    logger.info(f"  Target sample size: {config.target_sample_size}")

    # Show pipeline plan
    logger.info("\nPipeline plan:")
    logger.info(f"  Sampling: {'SKIP' if args.skip_sampling else 'RUN'}")
    logger.info(f"  Fact extraction: {'SKIP' if args.skip_facts else 'RUN'}")
    logger.info(f"  Query generation: {'SKIP' if args.skip_queries else 'RUN'}")
    logger.info(f"  Silver labeling: RUN")

    # Resolve file paths relative to current directory
    current_dir = Path(__file__).parent

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
        # Initialize variables
        sampled_chunks = None
        all_chunks = None
        sampling_stats = None

        # 4. Sample chunks (or load existing)
        if args.skip_sampling:
            logger.info("\n[4/6] STEP 1: Sampling chunks... SKIPPED")

            # If we have input chunks, load them
            if args.input_chunks:
                chunks_path = Path(args.input_chunks)
                if not chunks_path.is_absolute():
                    chunks_path = current_dir / chunks_path
                sampled_chunks = load_chunks_from_file(chunks_path)
                all_chunks = sampled_chunks  # For labeling step
                sampling_stats = {
                    'total_chunks': len(sampled_chunks),
                    'actual_samples': len(sampled_chunks),
                    'num_clusters': 'unknown'
                }
            # Otherwise, we'll fetch all chunks later for labeling
            else:
                logger.info("  No input chunks provided, will fetch all chunks for labeling")
        else:
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

        # 5. Extract facts (or load existing)
        if args.skip_facts:
            logger.info("\n[5/6] STEP 2: Extracting atomic facts... SKIPPED")

            # Load facts from file
            facts_path = Path(args.input_facts)
            if not facts_path.is_absolute():
                facts_path = current_dir / facts_path
            all_facts = load_facts_from_file(facts_path)

            # Compute fact type distribution
            fact_types = defaultdict(int)
            for fact in all_facts:
                fact_types[fact.fact_type] += 1

        else:
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

        # 6. Generate queries (or load existing)
        if args.skip_queries:
            logger.info("\n[6/6] STEP 3: Generating queries... SKIPPED")

            # Load queries from file
            queries_path = Path(args.input_queries)
            if not queries_path.is_absolute():
                queries_path = current_dir / queries_path
            queries = load_queries_from_file(queries_path)

            # Compute query stats
            query_stats = {
                'total_queries': len(queries),
                'single_hop': sum(1 for q in queries if q.query_type == 'single_hop'),
                'multi_hop': sum(1 for q in queries if q.query_type == 'multi_hop'),
                'query_styles': defaultdict(int)
            }
            for q in queries:
                query_stats['query_styles'][q.question_style] += 1
            query_stats['query_styles'] = dict(query_stats['query_styles'])

        else:
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

        # Fetch all chunks if we don't have them
        if all_chunks is None:
            logger.info("  Fetching all chunks from Milvus for labeling...")
            sampler = ChunkSampler(config, retriever)
            all_chunks = sampler.fetch_all_chunks()
            logger.info(f"  Fetched {len(all_chunks)} chunks")

        qrels = labeler.batch_label_queries(queries, all_chunks)
        
        logger.info(f"  Labeled {len(qrels)} queries")
        
        # Compute label statistics
        label_stats = labeler.compute_label_statistics(qrels)
        
        # 8. Write output
        logger.info("\nSTEP 5: Writing output files...")
        formatter = OutputFormatter(config.output_dir)
        
        # Combine all stats
        num_sampled = len(sampled_chunks) if sampled_chunks else 0
        all_stats = {
            'sampling': sampling_stats if sampling_stats else {'skipped': True},
            'fact_extraction': {
                'total_facts': len(all_facts),
                'avg_facts_per_chunk': len(all_facts) / num_sampled if num_sampled > 0 else 0,
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
        logger.info(f"  Total chunks: {len(all_chunks) if all_chunks else 0}")
        logger.info(f"  Sampled chunks: {num_sampled}")
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

