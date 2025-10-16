"""Quick test to verify refactoring didn't break anything."""

import sys
from pathlib import Path

# Add paths like main.py does
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
vector_ingest_path = project_root / "vector-ingest" / "src"
sys.path.insert(0, str(vector_ingest_path))

print("Testing refactored code...")
print("-" * 60)

# Test 1: Import all modules
print("\n[Test 1] Import all modules")
try:
    from llm_client import LLMClient
    from fact_extractor import FactExtractor, AtomicFact
    from query_generator import QueryGenerator, Query
    from silver_labeler import SilverLabeler
    from config import SyntheticEvalConfig
    print("[PASS] All imports successful")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: JSON parsing
print("\n[Test 2] Test JSON parsing static method")
try:
    test_cases = [
        '{"test": "value"}',
        '```json\n{"test": "value"}\n```',
        '```\n{"test": "value"}\n```',
    ]
    for test_json in test_cases:
        result = LLMClient.parse_json_response(test_json)
        assert result == {"test": "value"}, f"Failed for: {test_json}"
    print("[PASS] JSON parsing works correctly")
except Exception as e:
    print(f"✗ JSON parsing failed: {e}")
    sys.exit(1)

# Test 3: Config instantiation
print("\n[Test 3] Test config instantiation")
try:
    config = SyntheticEvalConfig()
    print(f"  Model: {config.model_name}")
    print(f"  Target questions: {config.target_questions}")
    print(f"  Multi-hop ratio: {config.multi_hop_ratio}")
    print("✓ Config instantiation works")
except Exception as e:
    print(f"✗ Config failed: {e}")
    sys.exit(1)

# Test 4: LLMClient parameter preparation
print("\n[Test 4] Test LLM parameter preparation")
try:
    # Mock API key manager
    class MockAPIKeyManager:
        def get_api_key(self):
            return "test-key"

    llm_client = LLMClient(MockAPIKeyManager(), config)

    messages = [{"role": "user", "content": "test"}]
    params = llm_client._prepare_params(messages)

    assert "model" in params
    assert "messages" in params
    assert params["model"] == config.model_name

    # Check token parameter based on model
    if config.model_name.startswith("gpt-5"):
        assert "max_completion_tokens" in params
    else:
        assert "max_tokens" in params

    print(f"  Model: {params['model']}")
    print(f"  Token param: {list(set(params.keys()) & {'max_tokens', 'max_completion_tokens'})}")
    print("✓ Parameter preparation works")
except Exception as e:
    print(f"✗ Parameter preparation failed: {e}")
    sys.exit(1)

# Test 5: AtomicFact serialization
print("\n[Test 5] Test AtomicFact serialization")
try:
    fact = AtomicFact(
        fact_id="test_fact_1",
        chunk_id="chunk_123",
        fact_type="factual_claim",
        fact_text="Test fact",
        answer_span="Test",
        answer_start=0,
        answer_end=4,
        entities=["Test Entity"],
        metadata={"source": "test"}
    )
    fact_dict = fact.to_dict()
    assert fact_dict["fact_id"] == "test_fact_1"
    assert fact_dict["chunk_id"] == "chunk_123"
    print("✓ AtomicFact serialization works")
except Exception as e:
    print(f"✗ AtomicFact failed: {e}")
    sys.exit(1)

# Test 6: Query serialization
print("\n[Test 6] Test Query serialization")
try:
    query = Query(
        query_id="q001",
        query_text="What is the test?",
        answer="Test answer",
        gold_chunk_ids=["chunk_123"],
        query_type="single_hop",
        question_style="wh_question",
        metadata={"fact_id": "test_fact_1"}
    )
    query_dict = query.to_dict()
    assert query_dict["query_id"] == "q001"
    assert query_dict["query_type"] == "single_hop"
    print("✓ Query serialization works")
except Exception as e:
    print(f"✗ Query failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED - Refactoring is safe!")
print("=" * 60)
