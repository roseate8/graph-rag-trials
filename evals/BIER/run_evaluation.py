#!/usr/bin/env python3
"""
Main script to run BEIR evaluation on GraphRAG pipeline.

Usage:
    python run_evaluation.py --quick                    # Quick evaluation
    python run_evaluation.py --dataset nfcorpus        # Single dataset
    python run_evaluation.py --suite standard          # Standard suite
"""

import sys
import argparse
from pathlib import Path
import logging

from graph_rag_adapter import GraphRAGAdapter
from beir_evaluator import BEIREvaluator
from config_utils import load_config, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BEIR evaluation for GraphRAG pipeline')
    
    # Evaluation mode
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick evaluation on subset of datasets')
    parser.add_argument('--dataset', type=str, 
                       help='Single dataset to evaluate')
    parser.add_argument('--suite', type=str, choices=['quick', 'standard', 'full'],
                       default='quick', help='Dataset suite to evaluate')
    
    # Configuration
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file')
    parser.add_argument('--embedding-model', type=str,
                       help='Override embedding model')
    parser.add_argument('--collection-name', type=str,
                       help='Override collection name')
    
    # Evaluation parameters  
    parser.add_argument('--top-k', type=int, default=10,
                       help='Top-k value for evaluation')
    parser.add_argument('--enable-reranking', action='store_true',
                       help='Enable re-ranking')
    parser.add_argument('--disable-reranking', action='store_true',
                       help='Disable re-ranking')
    
    # Output options
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    parser.add_argument('--save-per-query', action='store_true',
                       help='Save per-query results')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--debug', action='store_true',
                       help='Debug logging')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.debug else ('INFO' if args.verbose else 'WARNING')
    logger = get_logger(__name__, level=log_level)
    
    logger.info("Starting BIER evaluation")
    
    try:
        # Load configuration
        config_path = Path(args.config) if args.config else None
        config = load_config(config_path)
        
        # Override config with command line arguments
        retrieval_config = config['retrieval'].copy()
        if args.embedding_model:
            retrieval_config['embedding_model'] = args.embedding_model
        if args.collection_name:
            retrieval_config['collection_name'] = args.collection_name
        if args.enable_reranking:
            retrieval_config['enable_reranking'] = True
        elif args.disable_reranking:
            retrieval_config['enable_reranking'] = False
        
        # Initialize adapter
        logger.info("Initializing GraphRAG adapter...")
        adapter = GraphRAGAdapter(**retrieval_config)
        
        # Initialize evaluator
        results_dir = Path(args.output_dir) if args.output_dir else None
        evaluator = BEIREvaluator(adapter, results_dir=results_dir)
        
        # Run evaluation based on mode
        if args.quick:
            logger.info("Running quick evaluation...")
            results_df = evaluator.quick_eval()
            print("\nQuick Evaluation Results:")
            print(results_df.to_string(index=False))
            
        elif args.dataset:
            logger.info(f"Evaluating single dataset: {args.dataset}")
            result = evaluator.evaluate_dataset(
                dataset=args.dataset,
                top_k_values=[args.top_k],
                metrics=config['evaluation']['metrics'],
                save_results=True
            )
            print(f"\nResults for {args.dataset}:")
            for metric, value in result.metrics.items():
                print(f"  {metric}: {value:.4f}")
                
        else:
            logger.info(f"Evaluating dataset suite: {args.suite}")
            datasets = config['evaluation']['datasets'][args.suite]
            results = evaluator.evaluate_multiple(
                datasets=datasets,
                top_k_values=config['evaluation']['top_k_values'],
                metrics=config['evaluation']['metrics'],
                save_summary=True
            )
            
            print(f"\nResults for {args.suite} suite:")
            for dataset, result in results.items():
                print(f"\n{dataset}:")
                for metric, value in result.metrics.items():
                    print(f"  {metric}: {value:.4f}")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Cleanup
        try:
            if 'adapter' in locals():
                adapter.close()
        except:
            pass


if __name__ == '__main__':
    main()