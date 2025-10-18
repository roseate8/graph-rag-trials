"""Evaluate RAG system using Ragas-generated synthetic testset."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
try:
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        context_relevancy,
        answer_similarity,
        answer_correctness,
    )
except ImportError:
    # Ragas 0.3.x API
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
        answer_similarity,
        answer_correctness,
    )
    context_relevancy = None

from config import RAGAS_CONFIG, OPENAI_CONFIG, validate_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RagasEvaluator:
    """Evaluate RAG system using Ragas metrics."""
    
    def __init__(self):
        """Initialize Ragas evaluator."""
        # Get API key using secure llm_utils
        api_key = OPENAI_CONFIG["get_api_key"]()
        if not api_key:
            raise ValueError("OpenAI API key not available. Please run the script to be prompted for your key.")
        
        self.llm = ChatOpenAI(
            model=RAGAS_CONFIG["generator_model"],
            api_key=api_key,
            temperature=0.0,
            timeout=OPENAI_CONFIG["timeout"],
            max_retries=OPENAI_CONFIG["max_retries"],
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=RAGAS_CONFIG["embeddings_model"],
            api_key=api_key,
        )
        
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_similarity,
            answer_correctness,
        ]
        if context_relevancy is not None:
            self.metrics.append(context_relevancy)
        
        logger.info("Initialized Ragas evaluator with secure API key management")
    
    def load_testset(self, testset_path: str) -> pd.DataFrame:
        """Load testset from CSV or JSON file."""
        path = Path(testset_path)
        if not path.exists():
            raise FileNotFoundError(f"Testset not found: {testset_path}")
        
        df = pd.read_csv(testset_path) if path.suffix == ".csv" else pd.read_json(testset_path)
        logger.info(f"Loaded {len(df)} samples from {testset_path}")
        return df
    
    def prepare_evaluation_dataset(self, testset_df: pd.DataFrame, rag_responses: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare dataset for Ragas evaluation."""
        eval_data = []
        response_map = {resp["question"]: resp for resp in rag_responses}
        
        for _, row in testset_df.iterrows():
            question = row["question"]
            response = response_map.get(question)
            
            if response:
                eval_data.append({
                    "question": question,
                    "answer": response["answer"],
                    "contexts": response["contexts"],
                    "ground_truth": row.get("ground_truth", row.get("answer", "")),
                })
        
        return pd.DataFrame(eval_data)
    
    def evaluate(self, eval_dataset: pd.DataFrame, metrics: List = None) -> Dict[str, Any]:
        """Run Ragas evaluation on dataset."""
        from datasets import Dataset
        
        metrics = metrics or self.metrics
        logger.info(f"Evaluating {len(eval_dataset)} samples with {len(metrics)} metrics")
        
        dataset = Dataset.from_pandas(eval_dataset)
        results = evaluate(dataset=dataset, metrics=metrics, llm=self.llm, embeddings=self.embeddings)
        
        logger.info("Evaluation completed")
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str, filename: str = "evaluation_results.json"):
        """Save evaluation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {k: float(v) for k, v in results.items() if isinstance(v, (int, float))},
        }
        
        with open(output_path / filename, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Saved results to {output_path / filename}")
        return results_dict
    
    def generate_report(self, results: Dict[str, Any], output_dir: str):
        """Generate human-readable evaluation report."""
        report_lines = [
            "=" * 80,
            "Ragas RAG Evaluation Report",
            "=" * 80,
            "",
            f"Generated: {results.get('timestamp', datetime.now().isoformat())}",
            "",
            "Metrics:",
            "",
        ]
        
        for metric_name, score in sorted(results.get("metrics", {}).items()):
            report_lines.append(f"  {metric_name}: {score:.4f}")
        
        report_lines.extend(["", "=" * 80])
        
        report_path = Path(output_dir) / "evaluation_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        
        print("\n".join(report_lines))


def main():
    """Main function for RAG evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system using Ragas testset")
    parser.add_argument("--testset", type=str, default="output/testset.csv", help="Path to testset file")
    parser.add_argument("--output-dir", type=str, default="output/evaluation", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    logger.info("To evaluate your RAG system:")
    logger.info("1. Load testset from: " + args.testset)
    logger.info("2. Generate answers using your RAG system")
    logger.info("3. Use RagasEvaluator.evaluate() with your responses")


if __name__ == "__main__":
    main()
