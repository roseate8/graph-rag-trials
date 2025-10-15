"""
Utility functions for synthetic evaluation dataset generation.
Optimized for performance with compiled regex patterns and caching.
"""

import re
import string
from typing import List, Set, Tuple
from collections import Counter
from functools import lru_cache

# Pre-compiled regex patterns for better performance
DATE_PATTERNS = [
    re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),  # YYYY-MM-DD
    re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),  # MM/DD/YYYY or DD/MM/YYYY
    re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
    re.compile(r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b', re.IGNORECASE),
    re.compile(r'\b(?:Q[1-4]|FY)\s*\d{4}\b', re.IGNORECASE),
]

NUMBER_PATTERN = re.compile(r'[$€£¥]?\s*\d+[.,]?\d*\s*[kmbt%]?', re.IGNORECASE)
CURRENCY_PATTERN = re.compile(r'[$€£¥]\s*\d+[.,]?\d*\s*[kmbt]?', re.IGNORECASE)
MULTIPLIER_PATTERN = re.compile(r'([\d.]+)\s*([kmbt])', re.IGNORECASE)

# Cache for text normalization
_normalize_cache = {}


@lru_cache(maxsize=1000)
def normalize_text(text: str) -> str:
    """
    Normalize text for comparison with caching.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text (lowercase, no punctuation, stripped)
    """
    if not text:
        return ""
    
    # Lowercase and remove punctuation in one pass
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    return ' '.join(text.split())


def normalize_number(num_str: str) -> float:
    """
    Parse number strings with suffixes (K, M, B, T).
    
    Args:
        num_str: Number string (e.g., "1.2B", "$400M", "5K")
        
    Returns:
        Parsed number as float
        
    Examples:
        "1.2B" -> 1200000000.0
        "$400M" -> 400000000.0
        "5,000" -> 5000.0
    """
    # Remove currency symbols and commas
    cleaned = re.sub(r'[$€£¥,]', '', num_str.strip())
    
    # Handle suffixes
    multipliers = {
        'k': 1e3,
        'm': 1e6,
        'b': 1e9,
        't': 1e12,
    }
    
    # Check for suffix
    match = re.match(r'([\d.]+)\s*([kmbt])', cleaned.lower())
    if match:
        number = float(match.group(1))
        suffix = match.group(2)
        return number * multipliers[suffix]
    
    # Try direct conversion
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization for token-F1 calculation.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    # Normalize first
    normalized = normalize_text(text)
    
    # Split on whitespace
    return normalized.split()


def compute_token_f1(answer: str, text: str) -> float:
    """
    Compute token-level F1 score between answer and text.
    
    Args:
        answer: Expected answer
        text: Text to search in
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    answer_tokens = set(tokenize(answer))
    text_tokens = set(tokenize(text))
    
    if not answer_tokens or not text_tokens:
        return 0.0
    
    # Compute overlap
    common = answer_tokens & text_tokens
    
    if not common:
        return 0.0
    
    # Precision and recall
    precision = len(common) / len(text_tokens) if text_tokens else 0.0
    recall = len(common) / len(answer_tokens) if answer_tokens else 0.0
    
    # F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_token_f1_sentences(answer: str, text: str) -> Tuple[float, str]:
    """
    Compute max token-F1 score between answer and any sentence in text.
    
    Args:
        answer: Expected answer
        text: Text containing multiple sentences
        
    Returns:
        Tuple of (max_f1, best_matching_sentence)
    """
    sentences = extract_sentences(text)
    
    if not sentences:
        return 0.0, ""
    
    max_f1 = 0.0
    best_sentence = ""
    
    for sentence in sentences:
        f1 = compute_token_f1(answer, sentence)
        if f1 > max_f1:
            max_f1 = f1
            best_sentence = sentence
    
    return max_f1, best_sentence


def extract_sentences(text: str) -> List[str]:
    """
    Simple sentence segmentation.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Split on common sentence terminators
    sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def has_exact_match(answer: str, text: str, threshold: float = 0.9) -> bool:
    """
    Check if answer appears in text (with normalization).
    
    Args:
        answer: Answer string
        text: Text to search in
        threshold: Similarity threshold for fuzzy matching
        
    Returns:
        True if exact match found
    """
    normalized_answer = normalize_text(answer)
    normalized_text = normalize_text(text)
    
    # Exact substring match
    if normalized_answer in normalized_text:
        return True
    
    # For numbers, try numeric comparison
    if is_number(answer):
        try:
            answer_num = normalize_number(answer)
            # Find all numbers in text
            number_pattern = r'[$€£¥]?\s*\d+[.,]?\d*\s*[kmbt]?'
            text_numbers = re.findall(number_pattern, text, re.IGNORECASE)
            
            for text_num in text_numbers:
                if abs(normalize_number(text_num) - answer_num) < 1e-6:
                    return True
        except:
            pass
    
    return False


def is_number(text: str) -> bool:
    """
    Check if text represents a number.
    
    Args:
        text: Input text
        
    Returns:
        True if text is a number
    """
    # Remove common currency symbols and formatting
    cleaned = re.sub(r'[$€£¥,\s]', '', text)
    
    # Check if it matches number pattern
    number_pattern = r'^\d+[.,]?\d*[kmbt]?$'
    return bool(re.match(number_pattern, cleaned, re.IGNORECASE))


def extract_dates(text: str) -> List[str]:
    """
    Extract date patterns from text using pre-compiled regex.
    
    Args:
        text: Input text
        
    Returns:
        List of date strings
    """
    if not text:
        return []
    
    dates = []
    for pattern in DATE_PATTERNS:
        dates.extend(pattern.findall(text))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dates = []
    for date in dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)
    
    return unique_dates


def extract_numbers(text: str) -> List[str]:
    """
    Extract number patterns from text using pre-compiled regex.
    
    Args:
        text: Input text
        
    Returns:
        List of number strings
    """
    if not text:
        return []
    
    numbers = NUMBER_PATTERN.findall(text)
    
    # Filter and clean in one pass
    return [n.strip() for n in numbers if len(n.strip()) > 1]


def extract_currencies(text: str) -> List[str]:
    """
    Extract currency amounts from text using pre-compiled regex.
    
    Args:
        text: Input text
        
    Returns:
        List of currency strings
    """
    if not text:
        return []
    
    return CURRENCY_PATTERN.findall(text)


def compute_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set
        set2: Second set
        
    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if not set1 and not set2:
        return 1.0
    
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def find_answer_span(answer: str, text: str) -> Tuple[int, int]:
    """
    Find the character offset of answer in text.
    
    Args:
        answer: Answer string to find
        text: Text to search in
        
    Returns:
        Tuple of (start_offset, end_offset), or (-1, -1) if not found
    """
    # Try exact match first
    start = text.find(answer)
    if start != -1:
        return start, start + len(answer)
    
    # Try case-insensitive match
    start = text.lower().find(answer.lower())
    if start != -1:
        return start, start + len(answer)
    
    # Try normalized match
    normalized_text = normalize_text(text)
    normalized_answer = normalize_text(answer)
    
    start = normalized_text.find(normalized_answer)
    if start != -1:
        # This is approximate since normalization changes offsets
        return start, start + len(normalized_answer)
    
    return -1, -1

