"""
Main entry point for metrics calculation.

Optimized for clean startup and minimal overhead.

Usage:
    python -m main
    or
    python main.py
"""

import sys
import asyncio
import logging
from pathlib import Path

from config import EvalConfig
from evaluator import Evaluator


def setup_logging():
    """Setup logging configuration. Optimized: lazy log file creation."""
    # Ensure results directory exists before creating log file
    Path('results').mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('results/evaluation.log', mode='w', encoding='utf-8')
        ]
    )

    # Suppress verbose logging from dependencies - optimized list
    for logger_name in ['httpx', 'urllib3', 'pymilvus', 'sentence_transformers']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


async def main():
    """Main async entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting synthetic evaluation metrics calculation...")
    
    try:
        # Load configuration
        config = EvalConfig()
        
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
        
        # Run evaluation
        success = await evaluator.run()
        
        if success:
            logger.info("\n✓ Evaluation completed successfully!")
            logger.info(f"Results saved to: {config.results_dir}/")
            logger.info(f"\nKey files:")
            logger.info(f"  - {config.metrics_overall_file}")
            logger.info(f"  - {config.metrics_by_type_file}")
            logger.info(f"  - {config.detailed_report_file}")
            return 0
        else:
            logger.error("\n✗ Evaluation failed!")
            return 1
    
    except KeyboardInterrupt:
        logger.warning("\n⚠ Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Run async main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

