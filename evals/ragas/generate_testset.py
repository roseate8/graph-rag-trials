"""Generate synthetic test dataset using Ragas framework."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from langchain.docstore.document import Document as LangchainDocument

from config import RAGAS_CONFIG, OPENAI_CONFIG, OUTPUT_CONFIG, LOGGING_CONFIG, MILVUS_CONFIG, validate_config
from milvus_loader import load_documents_for_ragas

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RagasTestsetGenerator:
    """Generate synthetic test datasets using Ragas framework."""
    
    def __init__(self):
        """Initialize Ragas testset generator."""
        # Get API key using secure llm_utils
        api_key = OPENAI_CONFIG["get_api_key"]()
        if not api_key:
            raise ValueError("OpenAI API key not available. Please run the script to be prompted for your key.")
        
        self.generator_llm = ChatOpenAI(
            model=RAGAS_CONFIG["generator_model"],
            api_key=api_key,
            temperature=0.7,
            timeout=OPENAI_CONFIG["timeout"],
            max_retries=OPENAI_CONFIG["max_retries"],
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=RAGAS_CONFIG["embeddings_model"],
            api_key=api_key,
        )
        
        # Ragas 0.3.x API
        self.generator = TestsetGenerator(llm=self.generator_llm, embedding_model=self.embeddings)
        
        logger.info("Initialized Ragas testset generator with secure API key management")
    
    def generate_testset(self, documents, testset_size: int = None, distributions: dict = None):
        """Generate synthetic testset from documents."""
        testset_size = testset_size or RAGAS_CONFIG["testset_size"]
        distributions = distributions or RAGAS_CONFIG["distributions"]
        
        logger.info(f"Generating {testset_size} samples from {len(documents)} documents")
        
        # Ragas 0.3.x API - generate method signature changed
        testset = self.generator.generate_with_langchain_docs(
            documents,
            testset_size=testset_size
        )
        
        logger.info(f"Generated testset with {len(testset)} samples")
        return testset
    
    def save_testset(self, testset, output_dir: str = None):
        """Save testset to CSV and JSON files."""
        output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        df = testset.to_pandas()
        
        # Save as CSV and JSON
        csv_path = output_path / OUTPUT_CONFIG["testset_csv"]
        json_path = output_path / OUTPUT_CONFIG["testset_json"]
        
        df.to_csv(csv_path, index=False)
        df.to_json(json_path, orient="records", indent=2)
        
        logger.info(f"Saved testset to {csv_path} and {json_path}")
        return df
    
    def generate_report(self, df: pd.DataFrame, generation_time: float, num_documents: int, output_dir: str = None):
        """Generate statistics and report about the testset."""
        output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
        output_path = Path(output_dir)
        
        # Calculate statistics efficiently
        stats = {
            "timestamp": datetime.now().isoformat(),
            "generation_time_seconds": generation_time,
            "models": {
                "generator": RAGAS_CONFIG["generator_model"],
                "critic": RAGAS_CONFIG["critic_model"],
                "embeddings": RAGAS_CONFIG["embeddings_model"],
            },
            "dataset": {
                "total_samples": len(df),
                "source_documents": num_documents,
                "distributions": RAGAS_CONFIG["distributions"],
            },
            "statistics": {
                "question_length": {
                    "mean": float(df["question"].str.len().mean()) if "question" in df.columns else 0,
                    "min": int(df["question"].str.len().min()) if "question" in df.columns else 0,
                    "max": int(df["question"].str.len().max()) if "question" in df.columns else 0,
                },
                "answer_length": {
                    "mean": float(df["ground_truth"].str.len().mean()) if "ground_truth" in df.columns else 0,
                    "min": int(df["ground_truth"].str.len().min()) if "ground_truth" in df.columns else 0,
                    "max": int(df["ground_truth"].str.len().max()) if "ground_truth" in df.columns else 0,
                },
            },
        }
        
        # Add evolution distribution if available
        if "evolution_type" in df.columns:
            stats["evolution_distribution"] = df["evolution_type"].value_counts().to_dict()
        
        # Save stats as JSON
        stats_path = output_path / OUTPUT_CONFIG["stats_json"]
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Generate text report
        report_lines = [
            "=" * 80,
            "Ragas Synthetic Testset Generation Report",
            "=" * 80,
            "",
            f"Generated: {stats['timestamp']}",
            f"Generation Time: {generation_time:.2f} seconds",
            "",
            f"Models: {stats['models']['generator']} / {stats['models']['critic']}",
            f"Total Samples: {stats['dataset']['total_samples']}",
            f"Source Documents: {stats['dataset']['source_documents']}",
            "",
            f"Question Length: {stats['statistics']['question_length']['mean']:.1f} chars (avg)",
            f"Answer Length: {stats['statistics']['answer_length']['mean']:.1f} chars (avg)",
            "",
        ]
        
        if "evolution_distribution" in stats:
            report_lines.append("Evolution Distribution:")
            for k, v in stats["evolution_distribution"].items():
                report_lines.append(f"  {k}: {v}")
            report_lines.append("")
        
        report_lines.extend(["Sample Questions:", ""])
        for i, row in df.head(3).iterrows():
            report_lines.append(f"Q{i+1}: {row.get('question', 'N/A')[:150]}...")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_path = output_path / OUTPUT_CONFIG["report_txt"]
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Saved reports to {output_path}")
        print("\n".join(report_lines))
        
        return stats


def main(testset_size: int = None, max_documents: int = None, sample_strategy: str = None, output_dir: str = None):
    """Main function to generate synthetic testset."""
    logger.info("Starting Ragas synthetic testset generation")
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Use config defaults
    testset_size = testset_size or RAGAS_CONFIG["testset_size"]
    max_documents = max_documents or RAGAS_CONFIG["max_documents"]
    sample_strategy = sample_strategy or RAGAS_CONFIG["sample_strategy"]
    output_dir = output_dir or OUTPUT_CONFIG["output_dir"]
    
    # Load documents
    logger.info("Loading documents from Milvus...")
    try:
        # Create Milvus config
        from embeddings.milvus_config import MilvusConfig as MilvusConfigClass
        milvus_config = MilvusConfigClass(**MILVUS_CONFIG)
        
        documents = load_documents_for_ragas(
            max_documents=max_documents, 
            sample_strategy=sample_strategy,
            config=milvus_config
        )
        if not documents:
            logger.error("No documents loaded from Milvus")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load documents from Milvus: {e}", exc_info=True)
        sys.exit(1)
    
    # Generate testset
    logger.info("Generating testset...")
    generator = RagasTestsetGenerator()
    
    start_time = datetime.now()
    try:
        testset = generator.generate_testset(documents=documents, testset_size=testset_size)
    except Exception as e:
        logger.error(f"Failed to generate testset: {e}", exc_info=True)
        sys.exit(1)
    
    generation_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Generation completed in {generation_time:.2f} seconds")
    
    # Save results
    df = generator.save_testset(testset, output_dir=output_dir)
    generator.generate_report(df, generation_time, len(documents), output_dir=output_dir)
    
    logger.info("✓ Testset generation complete!")
    return testset, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic test dataset using Ragas")
    parser.add_argument("--testset-size", type=int, help=f"Number of samples (default: {RAGAS_CONFIG['testset_size']})")
    parser.add_argument("--max-documents", type=int, help=f"Max documents to load (default: {RAGAS_CONFIG['max_documents']})")
    parser.add_argument("--sample-strategy", choices=["random", "representative"], help="Sampling strategy")
    parser.add_argument("--output-dir", type=str, help=f"Output directory (default: {OUTPUT_CONFIG['output_dir']})")
    
    args = parser.parse_args()
    main(args.testset_size, args.max_documents, args.sample_strategy, args.output_dir)
