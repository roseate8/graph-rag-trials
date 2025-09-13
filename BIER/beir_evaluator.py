"""
BEIR evaluation framework integration for GraphRAG pipeline.

This module provides evaluation capabilities using the BEIR framework
with our existing retrieval pipeline.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
import pandas as pd

try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    from beir.evaluation import EvaluateRetrieval
    from beir.evaluation.evaluate import EvaluateRetrieval as Evaluate
except ImportError as e:
    raise ImportError(f"BEIR not installed: {e}. Install with: pip install beir")

from graph_rag_adapter import GraphRAGAdapter

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    dataset: str
    metrics: Dict[str, float]
    per_query_results: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def save_json(self, filepath: Union[str, Path]):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class BEIREvaluator:
    """
    BEIR evaluation framework for GraphRAG pipeline.
    
    Provides standardized evaluation across multiple information retrieval datasets
    using our existing vector store and retrieval pipeline.
    """
    
    # Available BEIR datasets
    AVAILABLE_DATASETS = [
        "msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", 
        "fiqa", "arguana", "touche-2020", "cqadupstack", 
        "quora", "dbpedia-entity", "scidocs", "fever", "climate-fever",
        "scifact", "germanquad", "capice"
    ]
    
    def __init__(
        self,
        adapter: GraphRAGAdapter,
        data_dir: Optional[Union[str, Path]] = None,
        results_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 100
    ):
        """
        Initialize BEIR evaluator.
        
        Args:
            adapter: GraphRAG adapter instance
            data_dir: Directory to store BEIR datasets
            results_dir: Directory to save evaluation results
            batch_size: Batch size for evaluation
        """
        self.adapter = adapter
        self.batch_size = batch_size
        
        # Setup directories
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data"
        if results_dir is None:
            results_dir = Path(__file__).parent.parent.parent / "results"
            
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BEIR evaluator initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def download_dataset(self, dataset: str) -> Tuple[str, Dict, Dict, Dict]:
        """
        Download and load a BEIR dataset.
        
        Args:
            dataset: Dataset name (e.g., 'msmarco', 'nfcorpus')
            
        Returns:
            Tuple of (data_path, corpus, queries, qrels)
        """
        if dataset not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset '{dataset}' not available. Choose from: {self.AVAILABLE_DATASETS}")
        
        logger.info(f"Downloading dataset: {dataset}")
        
        # Download dataset to our data directory
        dataset_path = self.data_dir / dataset
        
        try:
            # Use BEIR's download utility
            data_path = util.download_and_unzip(
                url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip",
                out_dir=str(self.data_dir)
            )
            
            # Load the dataset
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
            
            logger.info(f"Dataset '{dataset}' loaded successfully")
            logger.info(f"  Corpus: {len(corpus)} documents")
            logger.info(f"  Queries: {len(queries)} queries") 
            logger.info(f"  Qrels: {len(qrels)} relevance judgments")
            
            return data_path, corpus, queries, qrels
            
        except Exception as e:
            logger.error(f"Failed to download/load dataset '{dataset}': {e}")
            raise
    
    def evaluate_dataset(
        self,
        dataset: str,
        top_k_values: List[int] = [1, 3, 5, 10, 100],
        metrics: List[str] = ["ndcg", "map", "recall", "precision"],
        save_results: bool = True
    ) -> EvaluationResult:
        """
        Evaluate on a single BEIR dataset.
        
        Args:
            dataset: Dataset name
            top_k_values: List of k values for evaluation
            metrics: List of metrics to compute
            save_results: Whether to save results to file
            
        Returns:
            EvaluationResult object
        """
        logger.info(f"Starting evaluation on dataset: {dataset}")
        
        # Download and load dataset
        try:
            data_path, corpus, queries, qrels = self.download_dataset(dataset)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return EvaluationResult(dataset=dataset, metrics={})
        
        # Create BEIR evaluator with our adapter
        evaluator = EvaluateRetrieval(self.adapter, score_function="dot")
        
        # Run evaluation
        logger.info("Running retrieval evaluation...")
        try:
            # Get maximum k value for retrieval
            max_k = max(top_k_values)
            
            # Perform evaluation
            results = evaluator.evaluate(
                corpus=corpus,
                queries=queries, 
                qrels=qrels,
                k_values=top_k_values,
                ignore_identical_ids=True
            )
            
            # Extract metrics
            evaluation_metrics = {}
            for metric in metrics:
                if metric in results:
                    for k in top_k_values:
                        key = f"{metric}@{k}"
                        if key in results[metric]:
                            evaluation_metrics[key] = results[metric][key]
            
            logger.info(f"Evaluation completed for {dataset}")
            logger.info(f"Results: {evaluation_metrics}")
            
            # Create result object
            result = EvaluationResult(
                dataset=dataset,
                metrics=evaluation_metrics,
                metadata={
                    'num_queries': len(queries),
                    'num_corpus': len(corpus),
                    'num_qrels': len(qrels),
                    'adapter_info': {
                        'embedding_model': self.adapter.embedding_model,
                        'enable_reranking': self.adapter.enable_reranking,
                        'collection_name': self.adapter.collection_name
                    }
                }
            )
            
            # Save results if requested
            if save_results:
                result_file = self.results_dir / f"{dataset}_results.json"
                result.save_json(result_file)
                logger.info(f"Results saved to: {result_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed for dataset {dataset}: {e}")
            return EvaluationResult(dataset=dataset, metrics={})
    
    def evaluate_multiple(
        self,
        datasets: List[str],
        top_k_values: List[int] = [1, 3, 5, 10, 100],
        metrics: List[str] = ["ndcg", "map", "recall", "precision"],
        save_summary: bool = True
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate on multiple BEIR datasets.
        
        Args:
            datasets: List of dataset names
            top_k_values: List of k values for evaluation
            metrics: List of metrics to compute
            save_summary: Whether to save summary results
            
        Returns:
            Dict mapping dataset -> EvaluationResult
        """
        logger.info(f"Starting evaluation on {len(datasets)} datasets")
        
        results = {}
        summary_data = []
        
        for dataset in datasets:
            try:
                result = self.evaluate_dataset(
                    dataset=dataset,
                    top_k_values=top_k_values,
                    metrics=metrics,
                    save_results=True
                )
                results[dataset] = result
                
                # Add to summary
                summary_row = {'dataset': dataset}
                summary_row.update(result.metrics)
                summary_data.append(summary_row)
                
            except Exception as e:
                logger.error(f"Failed to evaluate dataset {dataset}: {e}")
                results[dataset] = EvaluationResult(dataset=dataset, metrics={})
        
        # Save summary if requested
        if save_summary and summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.results_dir / "evaluation_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Summary saved to: {summary_file}")
            
            # Also save as JSON
            summary_json = self.results_dir / "evaluation_summary.json"
            with open(summary_json, 'w') as f:
                json.dump(summary_data, f, indent=2)
        
        logger.info(f"Completed evaluation on {len(datasets)} datasets")
        return results
    
    def quick_eval(
        self,
        datasets: Optional[List[str]] = None,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Quick evaluation on selected datasets with minimal configuration.
        
        Args:
            datasets: List of datasets (uses default subset if None)
            top_k: Single k value for evaluation
            
        Returns:
            DataFrame with results summary
        """
        if datasets is None:
            # Default subset for quick evaluation
            datasets = ["nfcorpus", "fiqa", "scifact"]
        
        results = self.evaluate_multiple(
            datasets=datasets,
            top_k_values=[top_k],
            metrics=["ndcg", "map", "recall"],
            save_summary=False
        )
        
        # Convert to DataFrame
        summary_data = []
        for dataset, result in results.items():
            row = {'dataset': dataset}
            row.update(result.metrics)
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)