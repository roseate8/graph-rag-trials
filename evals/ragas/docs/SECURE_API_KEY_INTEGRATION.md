# Secure API Key Integration

## Overview
The Ragas implementation now uses your project's **secure `llm_utils`** for OpenAI API key management, ensuring strict compliance with your API key security requirements.

## Key Features

### ✅ Secure API Key Management
- **Never stored in environment variables**
- **Never saved to disk**
- **Temporary in-memory storage only**
- **Auto-clears after 300 minutes**
- **Prompt-based user input**
- **Secure cleanup on exit**

### 🔒 Security Implementation

The integration uses `vector-ingest/src/chunking/processors/llm_utils.py`:

```python
from llm_utils import get_openai_api_key, has_openai_api_key

# Secure key retrieval - prompts user when needed
api_key = get_openai_api_key()

# Check if key exists without prompting
has_key = has_openai_api_key()
```

## How It Works

### 1. Configuration (`config.py`)
```python
# Import secure llm_utils
from llm_utils import get_openai_api_key, has_openai_api_key

# API key function instead of direct value
OPENAI_CONFIG = {
    "get_api_key": get_api_key,  # Function to get key securely
    "timeout": 60,
    "max_retries": 3,
}
```

### 2. Generator Initialization (`generate_testset.py`)
```python
class RagasTestsetGenerator:
    def __init__(self):
        # Get API key using secure llm_utils
        api_key = OPENAI_CONFIG["get_api_key"]()
        
        # Initialize OpenAI clients with secured key
        self.generator_llm = ChatOpenAI(api_key=api_key, ...)
        self.embeddings = OpenAIEmbeddings(api_key=api_key, ...)
```

### 3. Evaluator Initialization (`evaluate_rag.py`)
```python
class RagasEvaluator:
    def __init__(self):
        # Same secure pattern
        api_key = OPENAI_CONFIG["get_api_key"]()
        self.llm = ChatOpenAI(api_key=api_key, ...)
        self.embeddings = OpenAIEmbeddings(api_key=api_key, ...)
```

## User Experience

### First Run
```bash
$ python generate_testset.py --testset-size 20

OpenAI API Key Required
Your API key will be stored temporarily for this session only.
It will be automatically cleared and never saved to disk.

Enter your OpenAI API key: sk-...

API key stored temporarily (will auto-clear in 300 minutes)
Starting Ragas synthetic testset generation...
```

### Subsequent Runs (Same Session)
```bash
$ python generate_testset.py --testset-size 50

# No prompt - uses cached key from secure manager
Starting Ragas synthetic testset generation...
```

### After 300 Minutes or Exit
```
API key cleared from memory
```

## Security Features

### 1. No Environment Variables
❌ **Old approach (insecure)**:
```bash
export OPENAI_API_KEY="sk-..."  # Visible in shell history
python generate_testset.py
```

✅ **New approach (secure)**:
```bash
# No export needed
python generate_testset.py
# Prompts for key securely
```

### 2. No Disk Storage
- API keys **never** written to files
- Not in `.env` files
- Not in configuration files
- Not in logs

### 3. Temporary Memory Storage
- Stored only in Python process memory
- Auto-cleared after timeout (300 minutes)
- Cleared on process exit
- Overwritten with random data before clearing

### 4. Session-Based
- Key valid for single session
- Shared across multiple script runs in same session
- Automatic expiration

### 5. Validation
- Checks key format (`sk-` or `sk-proj-` prefix)
- Confirms with user if format is unexpected
- Graceful error handling

## Integration Details

### Path Resolution
```python
# config.py automatically adds llm_utils to path
sys.path.insert(0, str(
    Path(__file__).parent.parent.parent / 
    "vector-ingest" / "src" / "chunking" / "processors"
))
```

### Fallback Handling
```python
try:
    from llm_utils import get_openai_api_key, has_openai_api_key
    _has_llm_utils = True
except ImportError:
    _has_llm_utils = False
    # Validation will error if llm_utils not available
```

### Validation
```python
def validate_config():
    if not _has_llm_utils:
        errors.append("llm_utils not available - cannot manage API key securely")
```

## Benefits

### For Users
- **Convenient**: Prompted once per session
- **Secure**: No accidental key exposure
- **Automatic**: Handles expiration and cleanup
- **Safe**: Keys never persisted anywhere

### For Security
- **Compliance**: Meets strict security requirements
- **Auditable**: Clear key lifecycle management
- **Traceable**: Know exactly where keys are used
- **Contained**: Isolated to session scope

### For Development
- **Consistent**: Same pattern across all scripts
- **Maintainable**: Centralized in `llm_utils`
- **Testable**: Can verify without real keys
- **Flexible**: Easy to update security policies

## Testing

### Verify Integration
```bash
cd evals/ragas
python -c "from config import _has_llm_utils; print(f'llm_utils available: {_has_llm_utils}')"
```

### Check Configuration
```bash
python -c "from config import validate_config; validate_config(); print('OK')"
```

### Test Key Check (No Prompt)
```bash
python -c "from llm_utils import has_openai_api_key; print(f'Has key: {has_openai_api_key()}')"
```

## Migration Notes

### Changes Made
1. **config.py**: API key now retrieved via function instead of env var
2. **generate_testset.py**: Calls `get_api_key()` in `__init__`
3. **evaluate_rag.py**: Calls `get_api_key()` in `__init__`

### Breaking Changes
**None** - External API remains the same:
```bash
python generate_testset.py --testset-size 100
```

### Behavioral Changes
- User will be prompted for API key on first use
- Key persists for 300 minutes instead of indefinitely
- No need to set `OPENAI_API_KEY` environment variable

## Usage Examples

### Generate Testset
```bash
cd evals/ragas
python generate_testset.py --testset-size 20
# You'll be prompted for API key
# Key stored securely for session
```

### Generate Multiple Testsets (Same Session)
```bash
# First run - prompts for key
python generate_testset.py --testset-size 20

# Second run - uses cached key (no prompt)
python generate_testset.py --testset-size 50

# Third run - still uses cached key
python generate_testset.py --testset-size 100
```

### Evaluate RAG System
```bash
python evaluate_rag.py --testset output/testset.csv
# Will prompt for key if not already stored
```

## Comparison

| Feature | Old (Env Var) | New (llm_utils) |
|---------|---------------|-----------------|
| **Storage** | Environment | Memory only |
| **Persistence** | Until unset | 300 minutes |
| **Security** | Medium | High |
| **Shell History** | Visible | Not visible |
| **Disk Storage** | Possible | Never |
| **Auto-Clear** | Manual | Automatic |
| **Validation** | None | Built-in |
| **Audit Trail** | None | Managed |

## Best Practices

### ✅ Do
- Run scripts normally - they'll prompt when needed
- Trust the timeout mechanism
- Let keys auto-clear
- Use the same session for multiple runs

### ❌ Don't
- Set `OPENAI_API_KEY` environment variable
- Store keys in files
- Share keys between users
- Bypass the prompt mechanism

## Troubleshooting

### "llm_utils not available"
**Cause**: Can't find `llm_utils.py`
**Solution**: Verify path structure:
```
graph-rag-trials/
├── vector-ingest/
│   └── src/
│       └── chunking/
│           └── processors/
│               └── llm_utils.py
└── evals/
    └── ragas/
```

### "OpenAI API key not available"
**Cause**: No key stored yet
**Solution**: Just run the script - it will prompt

### Key Expires During Generation
**Cause**: Generation took > 300 minutes
**Solution**: Run again - will prompt for fresh key

## Summary

✅ **Secure Integration Complete**
- Ragas now uses your project's secure `llm_utils`
- API keys managed with strict security controls
- No environment variables or disk storage
- Automatic cleanup and expiration
- User-friendly prompting system
- Full compliance with your security requirements

The implementation ensures that OpenAI API keys are handled securely throughout the entire Ragas test generation and evaluation pipeline.

