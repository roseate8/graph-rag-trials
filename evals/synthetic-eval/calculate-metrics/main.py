"""
Main entry point for metrics calculation.

Enhanced CLI with comprehensive argument parsing for flexible evaluation.

Usage:
    # Basic usage (all queries)
    python main.py

    # Limit number of queries
    python main.py --num-queries 50

    # Filter by query type
    python main.py --query-type single_hop

    # Custom K values
    python main.py --k-values 1 5 10 20

    # Adjust concurrency
    python main.py --batch-size 20 --max-concurrent 20

    # Disable re-ranking for faster evaluation
    python main.py --no-reranking

    # Verbose mode
    python main.py --verbose

    # Custom output directory
    python main.py --output-dir results_experiment_1
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from config import EvalConfig
from evaluator import Evaluator


def setup_logging(verbose: bool = False, output_dir: str = "results"):
    """Setup logging configuration with optional verbose mode."""
    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'{output_dir}/evaluation.log', mode='w', encoding='utf-8')
        ]
    )

    # Suppress verbose logging from dependencies unless in verbose mode
    suppress_level = logging.INFO if verbose else logging.WARNING
    for logger_name in ['httpx', 'urllib3', 'pymilvus', 'sentence_transformers', 'torch']:
        logging.getLogger(logger_name).setLevel(suppress_level)


def parse_args():
    """Parse command line arguments with comprehensive options."""
    parser = argparse.ArgumentParser(
        description='Synthetic Evaluation Metrics Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate first 50 queries only
  python main.py --num-queries 50

  # Evaluate only single-hop queries
  python main.py --query-type single_hop

  # Use custom K values
  python main.py --k-values 1 5 10 20 50

  # Fast evaluation (no re-ranking, fewer queries)
  python main.py --no-reranking --num-queries 50

  # High concurrency for faster processing
  python main.py --batch-size 25 --max-concurrent 25

  # Evaluate specific query IDs
  python main.py --query-ids q0001 q0002 q0003
        """
    )

    # Query filtering options
    query_group = parser.add_argument_group('Query Filtering')
    query_group.add_argument(
        '--num-queries', '-n',
        type=int,
        help='Limit number of queries to evaluate (default: all)'
    )
    query_group.add_argument(
        '--query-type', '-qt',
        choices=['single_hop', 'multi_hop', 'all'],
        default='all',
        help='Filter queries by type (default: all)'
    )
    query_group.add_argument(
        '--query-ids',
        nargs='+',
        help='Evaluate specific query IDs only'
    )
    query_group.add_argument(
        '--skip-queries',
        type=int,
        default=0,
        help='Skip first N queries (for pagination)'
    )

    # Evaluation configuration
    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument(
        '--k-values',
        nargs='+',
        type=int,
        help='K values for metrics calculation (default: 1 3 5 10 20 50 100)'
    )
    eval_group.add_argument(
        '--collection',
        default='elastic_embeddings_m3',
        help='Milvus collection name (default: elastic_embeddings_m3)'
    )
    eval_group.add_argument(
        '--embedding-model',
        default='BAAI/bge-m3',
        help='Embedding model name (default: BAAI/bge-m3)'
    )
    eval_group.add_argument(
        '--no-reranking',
        action='store_true',
        help='Disable re-ranking for faster evaluation'
    )

    # Performance options
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument(
        '--batch-size', '-b',
        type=int,
        default=15,
        help='Batch size for concurrent retrieval (default: 15)'
    )
    perf_group.add_argument(
        '--max-concurrent', '-mc',
        type=int,
        default=15,
        help='Maximum concurrent operations (default: 15)'
    )

    # Input/Output options
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument(
        '--queries-file',
        default='../output/queries.jsonl',
        help='Path to queries JSONL file'
    )
    io_group.add_argument(
        '--qrels-file',
        default='../output/qrels.tsv',
        help='Path to qrels TSV file'
    )
    io_group.add_argument(
        '--output-dir', '-o',
        default='results',
        help='Output directory for results (default: results)'
    )

    # Logging options
    log_group = parser.add_argument_group('Logging Options')
    log_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    log_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (only errors and warnings)'
    )

    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and show query count without running evaluation'
    )
    analysis_group.add_argument(
        '--show-config',
        action='store_true',
        help='Display configuration and exit'
    )

    return parser.parse_args()


def create_config_from_args(args) -> EvalConfig:
    """Create EvalConfig from parsed arguments."""
    config = EvalConfig()

    # Apply argument overrides
    if args.k_values:
        config.k_values = sorted(args.k_values)

    config.collection_name = args.collection
    config.embedding_model = args.embedding_model
    config.enable_reranking = not args.no_reranking
    config.batch_size = args.batch_size
    config.max_concurrent = args.max_concurrent

    # Update file paths
    config.queries_file = args.queries_file
    config.qrels_file = args.qrels_file
    config.results_dir = args.output_dir

    # Update all result file paths with new output dir
    config.retrieval_results_file = f"{args.output_dir}/retrieval_results.jsonl"
    config.metrics_overall_file = f"{args.output_dir}/metrics_overall.json"
    config.metrics_by_type_file = f"{args.output_dir}/metrics_by_type.json"
    config.metrics_by_k_file = f"{args.output_dir}/metrics_by_k.json"
    config.detailed_report_file = f"{args.output_dir}/detailed_report.txt"
    config.failed_queries_file = f"{args.output_dir}/failed_queries.jsonl"

    return config


def display_config(config: EvalConfig, args):
    """Display current configuration."""
    print("\n" + "=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"\n[Query Filtering]")
    print(f"  - Number of queries: {args.num_queries or 'all'}")
    print(f"  - Query type filter: {args.query_type}")
    if args.query_ids:
        print(f"  - Specific query IDs: {len(args.query_ids)} queries")
    if args.skip_queries > 0:
        print(f"  - Skip first: {args.skip_queries} queries")

    print(f"\n[Retrieval Configuration]")
    print(f"  - Collection: {config.collection_name}")
    print(f"  - Embedding model: {config.embedding_model}")
    print(f"  - Re-ranking: {'Enabled' if config.enable_reranking else 'Disabled'}")

    print(f"\n[Metrics Configuration]")
    print(f"  - K values: {config.k_values}")

    print(f"\n[Performance Settings]")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max concurrent: {config.max_concurrent}")

    print(f"\n[Input/Output]")
    print(f"  - Queries file: {config.queries_file}")
    print(f"  - Qrels file: {config.qrels_file}")
    print(f"  - Output directory: {config.results_dir}")

    print("\n" + "=" * 80 + "\n")


async def main():
    """Main async entry point with CLI argument support."""
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    log_level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    setup_logging(args.verbose, args.output_dir)
    logger = logging.getLogger(__name__)

    # Set root logger level for quiet mode
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Create configuration from arguments
        config = create_config_from_args(args)

        # Show configuration if requested
        if args.show_config:
            display_config(config, args)
            return 0

        logger.info("Starting synthetic evaluation metrics calculation...")

        # Display configuration summary
        if not args.quiet:
            display_config(config, args)

        # Validate input files exist
        queries_path = Path(config.queries_file)
        qrels_path = Path(config.qrels_file)

        if not queries_path.exists():
            logger.error(f"Queries file not found: {queries_path}")
            logger.error("Please run the synthetic evaluation generator first.")
            return 1

        if not qrels_path.exists():
            logger.error(f"Qrels file not found: {qrels_path}")
            logger.error("Please run the synthetic evaluation generator first.")
            return 1

        # Create evaluator
        evaluator = Evaluator(config)

        # Apply query filtering if specified
        if args.num_queries or args.query_type != 'all' or args.query_ids or args.skip_queries:
            await apply_query_filters(evaluator, args, logger)

        # Dry run mode - just show what would be evaluated
        if args.dry_run:
            logger.info("\n[DRY RUN MODE] - No evaluation will be performed")
            logger.info(f"\nWould evaluate {len(evaluator.queries)} queries:")

            # Show query type breakdown
            from collections import Counter
            type_counts = Counter(
                q.get('metadata', {}).get('query_type', 'unknown')
                for q in evaluator.queries
            )
            for qtype, count in type_counts.items():
                logger.info(f"  - {qtype}: {count} queries")

            logger.info(f"\nWith settings:")
            logger.info(f"  - K values: {config.k_values}")
            logger.info(f"  - Re-ranking: {config.enable_reranking}")
            logger.info(f"  - Batch size: {config.batch_size}")

            return 0

        # Run evaluation
        success = await evaluator.run()

        if success:
            logger.info("\n" + "=" * 80)
            logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"\nResults saved to: {config.results_dir}/")
            logger.info(f"\nKey files:")
            logger.info(f"  - Overall metrics:    {config.metrics_overall_file}")
            logger.info(f"  - By query type:      {config.metrics_by_type_file}")
            logger.info(f"  - By K values:        {config.metrics_by_k_file}")
            logger.info(f"  - Detailed report:    {config.detailed_report_file}")
            logger.info(f"  - Retrieval results:  {config.retrieval_results_file}")

            if Path(config.failed_queries_file).exists():
                logger.warning(f"  - Failed queries:     {config.failed_queries_file}")

            logger.info("\n" + "=" * 80)
            return 0
        else:
            logger.error("\nEvaluation failed!")
            return 1

    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}", exc_info=True)
        return 1


async def apply_query_filters(evaluator, args, logger):
    """Apply query filtering based on CLI arguments."""
    import json

    # Load queries first
    success = evaluator.load_queries()
    if not success:
        raise RuntimeError("Failed to load queries")

    original_count = len(evaluator.queries)

    # Filter by specific query IDs
    if args.query_ids:
        query_id_set = set(args.query_ids)
        evaluator.queries = [q for q in evaluator.queries if q['_id'] in query_id_set]
        logger.info(f"Filtered to {len(evaluator.queries)} queries matching specified IDs")

    # Filter by query type
    if args.query_type != 'all':
        evaluator.queries = [
            q for q in evaluator.queries
            if q.get('metadata', {}).get('query_type') == args.query_type
        ]
        logger.info(f"Filtered to {len(evaluator.queries)} {args.query_type} queries")

    # Skip first N queries
    if args.skip_queries > 0:
        evaluator.queries = evaluator.queries[args.skip_queries:]
        logger.info(f"Skipped first {args.skip_queries} queries")

    # Limit number of queries
    if args.num_queries:
        evaluator.queries = evaluator.queries[:args.num_queries]
        logger.info(f"Limited to first {args.num_queries} queries")

    logger.info(f"Query filtering: {original_count} → {len(evaluator.queries)} queries")


if __name__ == "__main__":
    # Run async main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

