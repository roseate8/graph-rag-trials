#!/usr/bin/env python3
"""
BIER Quick Start - Simple script to run BEIR evaluation.

This is the easiest way to get started with BIER evaluation.
Just run: python quick_start.py
"""

import sys
from pathlib import Path

print("🎯 BIER Quick Start")
print("==================")

def quick_evaluation():
    """Run a quick evaluation on small datasets."""
    try:
        print("📦 Loading BIER components...")
        
        # Import components
        from graph_rag_adapter import GraphRAGAdapter
        from beir_evaluator import BEIREvaluator
        from config_utils import load_config
        
        print("⚙️  Loading configuration...")
        config = load_config()
        
        print("🔌 Connecting to GraphRAG pipeline...")
        # Initialize adapter with config
        adapter = GraphRAGAdapter(**config['retrieval'])
        
        print("📊 Initializing BEIR evaluator...")
        evaluator = BEIREvaluator(adapter)
        
        print("🚀 Running quick evaluation (this may take a few minutes)...")
        print("   Datasets: NFCorpus, FIQA, SciFact")
        
        # Run quick evaluation
        results_df = evaluator.quick_eval(
            datasets=["nfcorpus", "fiqa", "scifact"],
            top_k=10
        )
        
        print("\n✅ Evaluation Results:")
        print("=" * 60)
        print(results_df.to_string(index=False))
        
        # Save results
        results_file = Path("results") / "quick_evaluation_results.csv"
        results_file.parent.mkdir(exist_ok=True)
        results_df.to_csv(results_file, index=False)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        print("\n🎉 Quick evaluation completed successfully!")
        print("\nNext steps:")
        print("  - Check detailed results in results/ folder")
        print("  - Run full evaluation: python src/run_evaluation.py --suite standard")
        print("  - Customize settings in config/eval_config.yaml")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("\n💡 To fix this:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Ensure your vector store is running")
        return False
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("  - Check that Milvus/vector store is running")
        print("  - Verify configuration in config/eval_config.yaml")
        print("  - Check logs/ folder for detailed error messages")
        return False

def show_help():
    """Show help information."""
    print("\n📚 BIER Help")
    print("============")
    print("Available commands:")
    print("  python quick_start.py              # Run quick evaluation")
    print("  python src/run_evaluation.py       # Full evaluation script")
    print("  python src/bier/cli.py --help      # CLI interface")
    print("\nKey files:")
    print("  config/eval_config.yaml            # Configuration settings")
    print("  results/                           # Evaluation results")
    print("  README.md                          # Full documentation")
    print("  USAGE.md                           # Usage examples")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        show_help()
    else:
        success = quick_evaluation()
        sys.exit(0 if success else 1)