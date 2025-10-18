"""
Test script to validate Ragas setup and dependencies.

Run this to ensure everything is configured correctly before generating testsets.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_dependencies():
    """Test that all required dependencies are installed."""
    logger.info("Testing dependencies...")
    
    required_packages = [
        ("elasticsearch", "Elasticsearch client"),
        ("langchain", "LangChain framework"),
        ("langchain_openai", "LangChain OpenAI integration"),
        ("ragas", "Ragas framework"),
        ("pandas", "Data processing"),
    ]
    
    missing = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✓ {package} ({description})")
        except ImportError:
            logger.error(f"  ✗ {package} ({description}) - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✓ All dependencies installed\n")
    return True


def test_configuration():
    """Test configuration validity."""
    logger.info("Testing configuration...")
    
    try:
        from config import validate_config, ELASTICSEARCH_CONFIG, RAGAS_CONFIG
        
        # Test Elasticsearch config
        logger.info(f"  Elasticsearch URL: {ELASTICSEARCH_CONFIG['url']}")
        logger.info(f"  Elasticsearch Index: {ELASTICSEARCH_CONFIG['index_name']}")
        
        # Test Ragas config
        logger.info(f"  Testset Size: {RAGAS_CONFIG['testset_size']}")
        logger.info(f"  Generator Model: {RAGAS_CONFIG['generator_model']}")
        
        # Validate
        validate_config()
        logger.info("✓ Configuration valid\n")
        return True
    
    except ValueError as e:
        logger.error(f"✗ Configuration validation failed: {e}\n")
        return False
    except Exception as e:
        logger.error(f"✗ Configuration error: {e}\n")
        return False


def test_environment():
    """Test environment variables."""
    logger.info("Testing environment...")
    
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        masked_key = api_key[:7] + "..." + api_key[-4:] if len(api_key) > 11 else "***"
        logger.info(f"  ✓ OPENAI_API_KEY: {masked_key}")
        logger.info("✓ Environment configured\n")
        return True
    else:
        logger.warning("  ⚠ OPENAI_API_KEY not set")
        logger.info("  Set with: export OPENAI_API_KEY='sk-...'")
        logger.info("  Or create .env file\n")
        return False


def test_elasticsearch_connection():
    """Test Elasticsearch connection."""
    logger.info("Testing Elasticsearch connection...")
    
    try:
        from elasticsearch_loader import ElasticsearchDocumentLoader
        
        loader = ElasticsearchDocumentLoader()
        
        # Get stats
        stats = loader.get_index_stats()
        
        if stats:
            logger.info(f"  ✓ Connected to Elasticsearch")
            logger.info(f"  Index: {stats.get('index_name', 'N/A')}")
            logger.info(f"  Documents: {stats.get('document_count', 'N/A')}")
            logger.info("✓ Elasticsearch connection successful\n")
            loader.close()
            return True
        else:
            logger.error("  ✗ Could not retrieve index stats")
            loader.close()
            return False
    
    except Exception as e:
        logger.error(f"✗ Elasticsearch connection failed: {e}")
        logger.info("  Check network, credentials, and index name\n")
        return False


def test_document_loading():
    """Test document loading."""
    logger.info("Testing document loading...")
    
    try:
        from elasticsearch_loader import load_documents_for_ragas
        
        # Load small sample
        documents = load_documents_for_ragas(max_documents=5, sample_strategy="random")
        
        if documents:
            logger.info(f"  ✓ Loaded {len(documents)} documents")
            logger.info(f"  Sample content length: {len(documents[0].page_content)} chars")
            logger.info("✓ Document loading successful\n")
            return True
        else:
            logger.error("  ✗ No documents loaded")
            return False
    
    except Exception as e:
        logger.error(f"✗ Document loading failed: {e}\n")
        return False


def test_ragas_initialization():
    """Test Ragas generator initialization."""
    logger.info("Testing Ragas initialization...")
    
    try:
        from generate_testset import RagasTestsetGenerator
        
        generator = RagasTestsetGenerator()
        logger.info("  ✓ Ragas generator initialized")
        logger.info("✓ Ragas framework ready\n")
        return True
    
    except Exception as e:
        logger.error(f"✗ Ragas initialization failed: {e}")
        logger.info("  Check OpenAI API key and dependencies\n")
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger.info("=" * 60)
    logger.info("Ragas Setup Validation")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Dependencies
    results.append(("Dependencies", test_dependencies()))
    
    # Test 2: Configuration
    results.append(("Configuration", test_configuration()))
    
    # Test 3: Environment
    env_ok = test_environment()
    results.append(("Environment", env_ok))
    
    # Test 4: Elasticsearch (only if previous tests pass)
    if results[-1][1]:
        results.append(("Elasticsearch Connection", test_elasticsearch_connection()))
        
        # Test 5: Document Loading (only if ES connection works)
        if results[-1][1]:
            results.append(("Document Loading", test_document_loading()))
    
    # Test 6: Ragas Initialization (only if environment is OK)
    if env_ok:
        results.append(("Ragas Initialization", test_ragas_initialization()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n🎉 Setup complete! Ready to generate testsets.")
        logger.info("\nNext steps:")
        logger.info("  1. python generate_testset.py --testset-size 20")
        logger.info("  2. Review output/testset.csv")
        logger.info("  3. Scale up if quality is good")
        return True
    else:
        logger.info("\n⚠ Setup incomplete. Please fix the failing tests above.")
        logger.info("\nCommon fixes:")
        logger.info("  • Dependencies: pip install -r requirements.txt")
        logger.info("  • Environment: export OPENAI_API_KEY='sk-...'")
        logger.info("  • Elasticsearch: Check network and credentials")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
        sys.exit(1)

