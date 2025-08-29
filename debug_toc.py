#!/usr/bin/env python3
"""Debug TOC detection on the Form 10-K file."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.chunking.processors.toc_detector import TableOfContentsDetector

# Create a small sample that should definitely be detected as TOC
test_content = """## Table of Contents
|            |                                                                                                              | Page   |
|------------|--------------------------------------------------------------------------------------------------------------|--------|
| PART I     |                                                                                                              |        |
| Item 1.    | Business                                                                                                     | 5      |
| Item 1A.   | Risk Factors                                                                                                 | 16     |
| Item 1B.   | Unresolved Staff Comments                                                                                    | 49     |
| Item 1C.   | Cybersecurity                                                                                                | 49     |
| Item 2.    | Properties                                                                                                   | 50     |
| Item 3.    | Legal Proceedings                                                                                            | 50     |
"""

def debug_phases():
    detector = TableOfContentsDetector()
    lines = test_content.split('\n')
    
    print("=== DEBUG TOC DETECTOR ===")
    print(f"Input content ({len(lines)} lines):")
    for i, line in enumerate(lines):
        print(f"  {i+1}: '{line}'")
    
    print("\n=== PHASE 1: Pattern Matching ===")
    candidates = detector.phase1_pattern_matching(lines)
    print(f"Found {len(candidates)} candidates:")
    for i, candidate in enumerate(candidates):
        print(f"  Candidate {i+1}: score={candidate.score}, reason='{candidate.reason}'")
        print(f"    Lines {candidate.start_line}-{candidate.end_line}")
        print(f"    Content: {candidate.lines[:3]}...")
    
    if not candidates:
        print("Phase 1 found no candidates - investigating...")
        
        # Test keyword detection
        print("\n--- Testing keyword detection ---")
        for i, line in enumerate(lines):
            if detector.toc_keyword_regex.search(line.strip()):
                print(f"Found keyword in line {i+1}: '{line}'")
        
        # Test page number patterns
        print("\n--- Testing page number patterns ---")
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            for j, pattern in enumerate(detector.page_number_patterns):
                if pattern.match(line_stripped):
                    print(f"Pattern {j+1} matched line {i+1}: '{line_stripped}'")
                    break
            else:
                if '|' in line_stripped and any(c.isdigit() for c in line_stripped):
                    print(f"Line {i+1} has table format with numbers but no pattern match: '{line_stripped}'")
        
        return
    
    print("\n=== PHASE 2: Structural/Positional ===")
    candidates = detector.phase2_structural_positional(candidates, len(lines))
    print(f"After phase 2: {len(candidates)} candidates")
    for candidate in candidates:
        print(f"  Score: {candidate.score}, reason: {candidate.reason}")
    
    print("\n=== PHASE 3: Candidate Selection ===")
    candidates = detector.phase3_candidate_selection(candidates)
    print(f"After phase 3: {len(candidates)} candidates")
    for candidate in candidates:
        print(f"  Final score: {candidate.score}, reason: {candidate.reason}")

if __name__ == "__main__":
    debug_phases()