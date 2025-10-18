"""
Example: End-to-end Ragas testset generation and evaluation workflow.

This script demonstrates the complete workflow:
1. Load documents from Elasticsearch
2. Generate synthetic testset with Ragas
3. (Optional) Evaluate with your RAG system
"""

import logging
from pathlib import Path

from config import validate_config, RAGAS_CONFIG
from elasticsearch_loader import load_documents_for_ragas
from generate_testset import RagasTestsetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_generation():
    """Example: Generate a small testset."""
    logger.info("=== Example: Ragas Testset Generation ===")
    
    # Step 1: Validate configuration
    logger.info("Step 1: Validating configuration...")
    try:
        validate_config()
        logger.info("✓ Configuration valid")
    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
        logger.info("Please set OPENAI_API_KEY environment variable")
        return
    
    # Step 2: Load documents from Elasticsearch
    logger.info("\nStep 2: Loading documents from Elasticsearch...")
    try:
        documents = load_documents_for_ragas(
            max_documents=50,  # Small sample for example
            sample_strategy="random"
        )
        logger.info(f"✓ Loaded {len(documents)} documents")
        
        if documents:
            logger.info(f"\nSample document:")
            logger.info(f"  Content length: {len(documents[0].page_content)} chars")
            logger.info(f"  Preview: {documents[0].page_content[:150]}...")
    
    except Exception as e:
        logger.error(f"✗ Failed to load documents: {e}")
        logger.info("Check Elasticsearch connection and credentials")
        return
    
    # Step 3: Initialize generator
    logger.info("\nStep 3: Initializing Ragas generator...")
    try:
        generator = RagasTestsetGenerator()
        logger.info("✓ Generator initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize generator: {e}")
        return
    
    # Step 4: Generate testset
    logger.info("\nStep 4: Generating testset (this may take a few minutes)...")
    try:
        testset = generator.generate_testset(
            documents=documents,
            testset_size=10,  # Small for example
        )
        logger.info(f"✓ Generated testset with {len(testset)} samples")
    except Exception as e:
        logger.error(f"✗ Failed to generate testset: {e}")
        return
    
    # Step 5: Save testset
    logger.info("\nStep 5: Saving testset...")
    try:
        output_dir = "output/example"
        df = generator.save_testset(
            testset,
            output_dir=output_dir,
            filename_csv="example_testset.csv",
            filename_json="example_testset.json"
        )
        logger.info(f"✓ Saved testset to {output_dir}/")
        
        # Show sample
        if not df.empty:
            logger.info("\nSample question:")
            sample = df.iloc[0]
            logger.info(f"  Q: {sample.get('question', 'N/A')[:100]}...")
            logger.info(f"  A: {sample.get('ground_truth', 'N/A')[:100]}...")
    
    except Exception as e:
        logger.error(f"✗ Failed to save testset: {e}")
        return
    
    logger.info("\n=== Example Complete ===")
    logger.info(f"Next steps:")
    logger.info(f"  1. Review: {output_dir}/example_testset.csv")
    logger.info(f"  2. Generate full testset: python generate_testset.py")
    logger.info(f"  3. Integrate with your RAG system for evaluation")


def example_rag_integration():
    """Example: How to integrate testset with your RAG system."""
    logger.info("\n=== Example: RAG System Integration ===")
    
    logger.info("""
To evaluate your RAG system:

1. Load the generated testset:
   
   import pandas as pd
   testset = pd.read_csv('output/testset.csv')

2. For each question, query your RAG system:
   
   rag_responses = []
   for _, row in testset.iterrows():
       question = row['question']
       
       # YOUR RAG SYSTEM HERE
       answer, contexts = your_rag_system.query(question)
       
       rag_responses.append({
           'question': question,
           'answer': answer,
           'contexts': contexts  # List of retrieved context strings
       })

3. Evaluate with Ragas:
   
   from evaluate_rag import RagasEvaluator
   
   evaluator = RagasEvaluator()
   eval_dataset = evaluator.prepare_evaluation_dataset(
       testset_df=testset,
       rag_responses=rag_responses
   )
   
   results = evaluator.evaluate(eval_dataset)
   evaluator.save_results(results, 'output/evaluation')
   evaluator.generate_report(results, 'output/evaluation')

4. Review evaluation results:
   
   cat output/evaluation/evaluation_report.txt
""")


def example_custom_config():
    """Example: Customize generation parameters."""
    logger.info("\n=== Example: Custom Configuration ===")
    
    logger.info("""
Customize generation in config.py:

# Adjust testset size
RAGAS_CONFIG['testset_size'] = 200

# Change query distribution
RAGAS_CONFIG['distributions'] = {
    'simple': 0.5,        # More simple queries
    'reasoning': 0.3,     # Some reasoning
    'multi_context': 0.15, # Less multi-context
    'conditional': 0.05,  # Fewer conditional
}

# Change LLM models
RAGAS_CONFIG['generator_model'] = 'gpt-4'  # Higher quality
RAGAS_CONFIG['embeddings_model'] = 'text-embedding-3-large'  # Better embeddings

# Adjust document sampling
RAGAS_CONFIG['max_documents'] = 1000  # More source docs
RAGAS_CONFIG['sample_strategy'] = 'representative'  # Better coverage

Then regenerate:
    python generate_testset.py
""")


def main():
    """Run all examples."""
    logger.info("Starting Ragas Examples")
    logger.info("=" * 60)
    
    # Check if we should run actual generation
    import os
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("\n⚠ OPENAI_API_KEY not set!")
        logger.info("Set your API key to run examples:")
        logger.info("  export OPENAI_API_KEY='sk-...'")
        logger.info("\nShowing integration examples instead...\n")
        
        example_rag_integration()
        example_custom_config()
        
        logger.info("\nTo run full example:")
        logger.info("  export OPENAI_API_KEY='sk-...'")
        logger.info("  python example.py")
        return
    
    # Run actual generation example
    try:
        example_generation()
        example_rag_integration()
        example_custom_config()
    except KeyboardInterrupt:
        logger.info("\n\nExample interrupted by user")
    except Exception as e:
        logger.error(f"\n\nExample failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

