import re
import json
import requests
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from .llm_utils import get_openai_api_key, has_openai_api_key


class TOCFormat(Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    MIXED = "mixed"


@dataclass
class TOCEntry:
    level: int
    title: str
    page: Optional[int] = None
    section_id: Optional[str] = None
    line_number: Optional[int] = None
    confidence: float = 1.0


@dataclass
class TOCCandidate:
    start_line: int
    end_line: int
    lines: List[str]
    score: float
    reason: str
    llm_verified: bool = False
    llm_confidence: float = 0.0


class TableOfContentsDetector:
    """Systematic table of contents detection using multi-phase approach."""
    
    def __init__(self):
        self._compile_patterns()
        
        # Pre-cache common strings as frozensets for O(1) lookups
        self._toc_stop_keywords = frozenset(['#', 'chapter '])
        self._toc_position_words = frozenset(['chapter', 'part', 'section', 'item'])
        self._toc_subsection_words = frozenset(['subsection', 'appendix', 'annex'])
        self._toc_indicators = frozenset(['part i', 'item 1', 'item 2', 'business', 'page', 'contents'])
        
        # Cache for expensive operations
        self._line_cache = {}
        self._score_cache = {}
        
        # Import defaultdict once for reuse
        from collections import defaultdict
        self._defaultdict = defaultdict
    
    def _compile_patterns(self):
        """Compile regex patterns for systematic TOC detection."""
        # Single optimized regex for all TOC keywords
        self.toc_keyword_regex = re.compile(
            r'\b(?:table\s+of\s+contents?|toc|contents?|index|outline|summary|chapter\s+list)\b',
            re.IGNORECASE
        )
        
        # Pre-compiled tuple patterns for O(1) access by index
        self.page_number_patterns = (
            re.compile(r'^(.+?)\.{2,}\s*(\d+)$'),                    # Dotted leaders
            re.compile(r'^(.+?)\s{3,}(\d+)$'),                      # Multiple spaces
            re.compile(r'^\|\s*(.+?)\s*\|.*?\|\s*(\d+)\s*\|?$')    # Table format
        )
        
        self.sequential_patterns = (
            re.compile(r'^\s*(\d+)\.\s+(.+)$'),
            re.compile(r'^\s*(\d+\.\d+)\.\s+(.+)$')
        )
        
        # More specific heading pattern
        self.md_heading_regex = re.compile(r'^(#{1,6})\s+(.+)$')
        
        # Pre-compile TOC structure indicators for faster matching
        self.toc_structure_regex = re.compile(r'^[|\-*•]|\b(?:item|part|section|chapter)\b|\.{2,}|:.*\d+$', re.IGNORECASE)
    
    def detect_format(self, content: str) -> TOCFormat:
        """Detect the primary format of the document content."""
        try:
            json.loads(content)
            return TOCFormat.JSON
        except json.JSONDecodeError:
            pass
        
        if self.md_heading_regex.search(content):
            return TOCFormat.MARKDOWN
        
        return TOCFormat.MIXED
    
    def phase1_pattern_matching(self, lines: List[str]) -> List[TOCCandidate]:
        """Optimized Phase 1: Fast keyword and density-based matching."""
        candidates = []
        lines_count = len(lines)
        
        print(f"\nTOC PHASE 1: Scanning {lines_count} lines for TOC patterns...")
        
        # Pre-compile line strips for reuse
        stripped_lines = [line.strip() for line in lines]
        
        # Find keyword-triggered sections with optimized search
        keyword_matches = 0
        for i, stripped_line in enumerate(stripped_lines):
            if stripped_line and self.toc_keyword_regex.search(stripped_line):
                keyword_matches += 1
                print(f"\nFound TOC keyword at line {i+1}: '{stripped_line[:50]}'")
                
                # Look for TOC content after keyword
                end_line = self._find_toc_section_end_fast(lines, i)
                
                if end_line > i:
                    section_lines = lines[i:end_line]
                    score = self._score_page_number_density(section_lines)
                    
                    print(f"   Section lines {i}-{end_line} ({len(section_lines)} lines), score: {score:.3f}")
                    
                    candidates.append(TOCCandidate(
                        start_line=i,
                        end_line=end_line,
                        lines=section_lines,
                        score=score + 0.5,  # Keyword bonus
                        reason="keyword_triggered"
                    ))
        
        print(f"\nFound {keyword_matches} keyword matches")
        
        # Optimized density scanning with reduced window sizes
        print(f"\nScanning for high-density sections...")
        window_size = min(15, max(5, lines_count // 150))  # Smaller, more efficient windows
        step_size = max(2, window_size // 2)  # Larger steps
        base_threshold = 0.15
        
        density_candidates = 0
        for i in range(0, lines_count - window_size, step_size):
            window = lines[i:i + window_size]
            score = self._score_page_number_density(window)
            
            if score > base_threshold:
                density_candidates += 1
                print(f"   High density at lines {i}-{i+window_size}: score {score:.3f}")
                
                end_line = self._find_toc_section_end_fast(lines, i)
                section_lines = lines[i:end_line]
                
                candidates.append(TOCCandidate(
                    start_line=i,
                    end_line=end_line,
                    lines=section_lines,
                    score=score,
                    reason="page_number_density"
                ))
        
        print(f"\nPHASE 1 SUMMARY: {density_candidates} density candidates, {len(candidates)} total")
        return candidates
    
    def _find_toc_section_end_fast(self, lines: List[str], start: int) -> int:
        """Fast section end detection with optimized checks."""
        lines_len = len(lines)
        empty_count = 0
        end = start + 1
        max_section_size = min(80, lines_len - start)  # Reduced from 100
        
        print(f"     DEBUG: Finding section end from line {start}, max size: {max_section_size}")
        
        # Pre-compile stop conditions for speed
        for i in range(start + 1, min(start + max_section_size, lines_len)):
            line = lines[i].strip()
            
            if not line:
                empty_count += 1
                if empty_count >= 4:  # Reduced threshold
                    print(f"     DEBUG: Stopped at {empty_count} empty lines")
                    break
                continue
            
            empty_count = 0
            end = i + 1
            
            # Fast header detection
            if line.startswith(('#', '##')) and not ('table' in line.lower() or 'content' in line.lower()):
                print(f"     DEBUG: Stopped at header line {i}")
                break
            
            # Fast non-TOC content detection
            if (len(line) > 120 and  # Reduced from 150
                '|' not in line and ':' not in line and 
                line.count('.') < 2 and
                not any(line.endswith(str(n)) for n in range(10))):
                print(f"     DEBUG: Stopped at non-TOC line {i}")
                break
        
        print(f"     DEBUG: Section end: {end} (length: {end - start})")
        return end
    
    def _find_toc_section_end(self, lines: List[str], start: int) -> int:
        """Legacy method - use _find_toc_section_end_fast for better performance."""
        return self._find_toc_section_end_fast(lines, start)
    
    def _score_page_number_density(self, lines: List[str]) -> float:
        """Fast scoring of section based on TOC characteristics."""
        if not lines:
            return 0.0
        
        # Use single regex for all TOC structure detection
        page_number_lines = 0
        toc_structure_lines = 0
        non_empty_count = 0
        
        # Single pass through lines with early pattern matching
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            non_empty_count += 1
            
            # Fast pattern matching with early exit
            found_pattern = False
            for pattern in self.page_number_patterns:
                if pattern.match(line_stripped):
                    page_number_lines += 1
                    found_pattern = True
                    break
            
            # Only check structure if no page pattern found
            if not found_pattern and self.toc_structure_regex.search(line_stripped):
                toc_structure_lines += 1
        
        if non_empty_count == 0:
            return 0.0
        
        # Optimized calculation
        page_ratio = page_number_lines / non_empty_count
        structure_ratio = toc_structure_lines / non_empty_count
        
        # Weighted score with constants for speed
        return page_ratio * 0.7 + structure_ratio * 0.3
    
    def phase2_structural_positional(self, candidates: List[TOCCandidate], total_lines: int) -> List[TOCCandidate]:
        """Phase 2: Apply simple structural and positional heuristics."""
        scored_candidates = []
        
        for candidate in candidates:
            score = candidate.score
            
            # Simple positional bonus: favor earlier placement
            position_ratio = candidate.start_line / total_lines
            if position_ratio < 0.3:  # First 30% of document
                score += 0.2
            
            # Simple sequential numbering check
            if self._has_sequential_numbering(candidate.lines):
                score += 0.3
            
            scored_candidates.append(TOCCandidate(
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                lines=candidate.lines,
                score=score,
                reason=candidate.reason + "_positioned"
            ))
        
        return scored_candidates
    
    def _has_sequential_numbering(self, lines: List[str]) -> bool:
        """Fast check for sequential numbering pattern with early exit."""
        found_numbers = 0
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Fast pattern matching with early exit
            for pattern in self.sequential_patterns:
                match = pattern.match(line_stripped)
                if match:
                    try:
                        num_str = match.group(1).replace('chapter', '').strip()
                        if '.' not in num_str and num_str.isdigit():
                            found_numbers += 1
                            if found_numbers >= 2:  # Early exit
                                return True
                            break
                    except (ValueError, IndexError, AttributeError):
                        continue
        
        return False
    
    
    def phase3_candidate_selection(self, candidates: List[TOCCandidate]) -> List[TOCCandidate]:
        """Phase 3: Select best 1-3 candidates."""
        if not candidates:
            return []
        
        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        # Remove overlapping candidates (keep highest scoring)
        final_candidates = []
        for candidate in sorted_candidates:
            overlaps = False
            for existing in final_candidates:
                if (candidate.start_line < existing.end_line and 
                    candidate.end_line > existing.start_line):
                    overlaps = True
                    break
            
            if not overlaps:
                final_candidates.append(candidate)
            
            # Limit to top 2 for cost-effective LLM verification
            if len(final_candidates) >= 2:
                break
        
        # Only keep candidates with decent scores to avoid weak LLM calls
        return [c for c in final_candidates if c.score > 0.5]
    
    def phase4_llm_verification(self, candidates: List[TOCCandidate]) -> List[TOCCandidate]:
        """Phase 4: LLM verification using direct OpenAI API calls."""
        if not candidates:
            print("\nPHASE 4: No candidates for LLM verification")
            return candidates
        
        print(f"\nPHASE 4: Starting LLM verification for {len(candidates)} candidate(s)")
        
        # Track OpenAI API usage specifically
        total_openai_input_tokens = 0
        total_openai_output_tokens = 0
        total_openai_cost = 0.0
        api_calls_made = 0
        
        verified_candidates = []
        
        for i, candidate in enumerate(candidates, 1):
            print(f"\nVerifying candidate {i}/{len(candidates)}")
            print(f"   Lines: {candidate.start_line}-{candidate.end_line}")
            print(f"   Score: {candidate.score:.3f}")
            print(f"   Reason: {candidate.reason}")
            
            try:
                # Get candidate text
                candidate_text = "\n".join(candidate.lines)
                
                print(f"\nSENDING TO LLM:")
                print("=" * 50)
                print(candidate_text[:500] + ("..." if len(candidate_text) > 500 else ""))
                print("=" * 50)
                
                # Call LLM for verification
                llm_result = self._verify_toc_with_openai(candidate_text)
                
                if not llm_result["success"]:
                    print(f"ERROR: LLM verification failed: {llm_result.get('error', 'Unknown error')}")
                    # Keep candidate but mark as unverified
                    verified_candidates.append(candidate)
                    continue
                
                # Track OpenAI API usage from successful calls
                if "input_tokens" in llm_result:
                    total_openai_input_tokens += llm_result["input_tokens"]
                    total_openai_output_tokens += llm_result["output_tokens"]
                    total_openai_cost += llm_result["estimated_cost"]
                
                is_toc = llm_result.get("is_toc", False)
                confidence = llm_result.get("confidence", 0.0)
                reason = llm_result.get("reason", "No reason provided")
                
                print(f"\nLLM RESPONSE:")
                print(f"   Is TOC: {is_toc}")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Reason: {reason}")
                
                if is_toc and confidence >= 0.5:
                    # LLM confirms this is a TOC with decent confidence
                    candidate.score = candidate.score + (confidence * 0.5)  # Boost score
                    verified_candidates.append(candidate)
                    print(f"   ACCEPTED: High confidence TOC (final score: {candidate.score:.2f})")
                
                elif is_toc and confidence >= 0.3:
                    # LLM thinks it's a TOC but with low confidence - keep but don't boost
                    verified_candidates.append(candidate)
                    print(f"   ACCEPTED: Low confidence TOC (score unchanged: {candidate.score:.2f})")
                
                else:
                    # LLM doesn't think this is a TOC
                    print(f"   REJECTED: Not a TOC")
                    # Don't add to verified candidates - effectively filters it out
                
            except Exception as e:
                print(f"ERROR during LLM verification: {str(e)}")
                # Keep candidate in case of unexpected errors
                verified_candidates.append(candidate)
        
        print(f"\nLLM VERIFICATION COMPLETE: {len(verified_candidates)}/{len(candidates)} candidates retained")
        
        # Log total OpenAI API consumption
        if api_calls_made > 0:
            print(f"\n" + "="*60)
            print(f"OPENAI API CONSUMPTION SUMMARY")
            print(f"="*60)
            print(f"API calls made: {api_calls_made}")
            print(f"Total input tokens: {total_openai_input_tokens:,}")
            print(f"Total output tokens: {total_openai_output_tokens:,}")
            print(f"Total tokens: {total_openai_input_tokens + total_openai_output_tokens:,}")
            print(f"Total estimated cost: ${total_openai_cost:.6f}")
            print(f"="*60)
        else:
            print(f"\nNO OPENAI API CALLS WERE MADE")
        
        return verified_candidates
    
    def _verify_toc_with_openai(self, candidate_text: str) -> Dict[str, Any]:
        """Make direct API call to OpenAI to verify if text is a TOC."""
        try:
            # Check if we have API key (will prompt if needed)
            print(f"\nChecking for OpenAI API key...")
            if has_openai_api_key():
                print(f"   Valid API key found")
                api_key = get_openai_api_key()
            else:
                print(f"   No valid API key, will prompt user...")
                api_key = get_openai_api_key()
            
            # Prepare system prompt
            system_prompt = """You are a document analysis expert. Determine if the given text represents a Table of Contents (TOC).

Respond with a JSON object containing:
- "is_toc": boolean (true if this is a table of contents)
- "confidence": float (0.0-1.0, how confident you are)
- "reason": string explaining your decision

A table of contents typically:
- Lists document sections/chapters with titles
- May include page numbers (but not always)
- Has hierarchical structure (numbered sections, indentation, etc.)
- Contains multiple entries (usually 2+)
- Uses consistent formatting

Be strict but not overly rigid. Digital documents may not have page numbers."""
            
            user_prompt = f"""Analyze this text and determine if it's a Table of Contents:

```
{candidate_text}
```

Return your analysis as JSON."""
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4o-mini",  # Cost-effective model
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.1  # Low temperature for consistent analysis
            }
            
            print(f"\nCalling OpenAI API (gpt-4o-mini)...")
            
            # Make API call
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # Calculate approximate cost (GPT-4o-mini pricing)
                input_cost = (input_tokens / 1000) * 0.00015  # $0.15 per 1K input tokens
                output_cost = (output_tokens / 1000) * 0.0006  # $0.60 per 1K output tokens
                total_cost = input_cost + output_cost
                
                print(f"\nOPENAI API SUCCESS:")
                print(f"   Input tokens: {input_tokens:,}")
                print(f"   Output tokens: {output_tokens:,}")
                print(f"   Total tokens: {total_tokens:,}")
                print(f"   Estimated cost: ${total_cost:.6f}")
                
                # Track totals
                api_calls_made += 1
                
                # Parse JSON response
                try:
                    llm_analysis = json.loads(content)
                    
                    # Validate required fields
                    if not all(key in llm_analysis for key in ["is_toc", "confidence", "reason"]):
                        raise ValueError("Missing required fields in LLM response")
                    
                    return {
                        "success": True,
                        "is_toc": llm_analysis["is_toc"],
                        "confidence": llm_analysis["confidence"],
                        "reason": llm_analysis["reason"],
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "estimated_cost": total_cost
                    }
                
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Failed to parse LLM response as JSON: {str(e)}"
                    }
            
            else:
                return {
                    "success": False,
                    "error": f"API error {response.status_code}: {response.text}"
                }
        
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "API call timed out"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def phase5_validation(self, candidates: List[TOCCandidate], all_lines: List[str]) -> List[TOCCandidate]:
        """Fast validation by cross-checking against document headings."""
        if not candidates:
            return candidates
        
        # Skip validation if too many lines (performance optimization)
        if len(all_lines) > 2000:
            print("   Skipping validation for large document")
            return candidates
        
        # Fast heading extraction with early exit
        document_headings = set()
        heading_count = 0
        max_headings = 50  # Limit for performance
        
        for line in all_lines:
            if heading_count >= max_headings:
                break
            match = self.md_heading_regex.match(line)
            if match:
                document_headings.add(match.group(2).strip().lower())
                heading_count += 1
        
        # Fast candidate validation
        validated_candidates = []
        for candidate in candidates:
            validation_score = candidate.score
            
            if document_headings:
                matches = self._count_heading_matches_fast(candidate.lines, document_headings)
                if matches > 0:
                    validation_score += 0.1
            
            validated_candidates.append(TOCCandidate(
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                lines=candidate.lines,
                score=validation_score,
                reason=candidate.reason + "_validated"
            ))
        
        return validated_candidates
    
    def _count_heading_matches_fast(self, lines: List[str], headings: set) -> int:
        """Fast counting of heading matches with early exit."""
        matches = 0
        max_checks = 20  # Limit checks for performance
        
        for i, line in enumerate(lines):
            if i >= max_checks:
                break
            
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Fast pattern matching with early exit
            for pattern in self.page_number_patterns:
                match = pattern.match(line_stripped)
                if match:
                    title = match.group(1).strip().lower()
                    if any(title in heading or heading in title for heading in headings):
                        matches += 1
                        break  # Found match, move to next line
                    break  # Pattern matched but no heading match
        
        return matches
    
    def extract_entries_from_candidate(self, candidate: TOCCandidate) -> List[TOCEntry]:
        """Fast extraction of structured TOC entries."""
        entries = []
        base_line_num = candidate.start_line
        
        for line_idx, line in enumerate(candidate.lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            entry = self._extract_entry_fast(line_stripped, line, base_line_num + line_idx)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _extract_entry_fast(self, line_stripped: str, full_line: str, line_number: int) -> Optional[TOCEntry]:
        """Fast entry extraction with minimal object creation."""
        # Try page number patterns first (most common)
        for pattern in self.page_number_patterns:
            match = pattern.match(line_stripped)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    title = groups[0].strip()
                    page_str = groups[-1]  # Last group is usually page
                    page = int(page_str) if page_str.isdigit() else None
                    
                    level = self._determine_level_fast(title, full_line)
                    return TOCEntry(
                        level=level,
                        title=title,
                        page=page,
                        line_number=line_number,
                        confidence=0.9
                    )
        
        # Try sequential patterns
        for pattern in self.sequential_patterns:
            match = pattern.match(line_stripped)
            if match and len(match.groups()) >= 2:
                prefix = match.group(1).strip()
                title = match.group(2).strip()
                level = self._determine_level_from_prefix(prefix)
                
                return TOCEntry(
                    level=level,
                    title=title,
                    line_number=line_number,
                    confidence=0.7
                )
        
        return None
    
    def _determine_level_fast(self, title: str, full_line: str) -> int:
        """Fast level determination with cached calculations."""
        # Cache key for this line type
        cache_key = (len(full_line) - len(full_line.lstrip()), title[:20].lower())
        
        if cache_key in self._line_cache:
            return self._line_cache[cache_key]
        
        # Calculate level
        indent = len(full_line) - len(full_line.lstrip())
        level = max(1, (indent // 4) + 1)
        
        # Fast keyword check
        title_lower = title.lower()
        if any(title_lower.startswith(word) for word in self._toc_position_words):
            level = 1
        elif any(word in title_lower for word in self._toc_subsection_words):
            level = 2
        
        level = min(level, 6)
        
        # Cache result
        if len(self._line_cache) < 100:  # Limit cache size
            self._line_cache[cache_key] = level
        
        return level
    
    def _determine_level_from_prefix(self, prefix: str) -> int:
        """Determine level from TOC entry prefix."""
        if '.' in prefix:
            return prefix.count('.') + 1
        elif prefix.lower().startswith('chapter'):
            return 1
        else:
            return 1
    
    def detect_toc(self, content: str, source_format: Optional[TOCFormat] = None, json_path: Optional[str] = None) -> List[TOCEntry]:
        """Main method with JSON-first strategy and 1000-line limit."""
        print(f"\nStarting TOC detection pipeline...")
        print(f"Content length: {len(content)} characters")
        
        toc_entries = []
        
        # Strategy 1: JSON-first detection if JSON file is available
        if json_path and Path(json_path).exists():
            print(f"\nStrategy 1: Searching JSON tables for TOC...")
            toc_entries = self._detect_json_toc(json_path)
            if toc_entries:
                print(f"Found {len(toc_entries)} entries from JSON tables")
                return toc_entries
        
        # Strategy 2: Fallback to MD pattern search (limited to first 1000 lines)
        print(f"\nStrategy 2: Fallback to markdown pattern search (first 1000 lines)...")
        lines = content.split('\n')[:1000]  # Limit to first 1000 lines
        print(f"Limited search to {len(lines)} lines")
        
        # Phase 1: Pattern matching (on limited lines)
        candidates = self.phase1_pattern_matching(lines)
        
        # Phase 2: Structural and positional heuristics
        candidates = self.phase2_structural_positional(candidates, len(lines))
        
        # Phase 3: Candidate selection
        candidates = self.phase3_candidate_selection(candidates)
        
        # Early exit if no decent candidates found - save LLM costs
        if not candidates:
            return []
        
        # Phase 4: LLM verification
        candidates = self.phase4_llm_verification(candidates)
        
        # Phase 5: Validation
        candidates = self.phase5_validation(candidates, lines)
        
        # Extract entries from best candidate
        if candidates:
            best_candidate = max(candidates, key=lambda x: x.score)
            return self.extract_entries_from_candidate(best_candidate)
        
        return []
    
    def _detect_json_toc(self, json_path: str) -> List[TOCEntry]:
        """Detect TOC from JSON file by examining table structures."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Look for tables in the JSON structure
            tables = json_data.get('tables', [])
            if not tables:
                print("   No tables found in JSON")
                return []
            
            print(f"   Found {len(tables)} tables in JSON")
            
            # Check first few tables for TOC indicators
            for i, table in enumerate(tables[:5]):  # Check only first 5 tables
                if self._is_toc_table(table):
                    print(f"   Table {i} appears to be TOC")
                    return self._extract_toc_from_table(table)
            
            print("   No TOC table found in first 5 tables")
            return []
            
        except Exception as e:
            print(f"   ERROR reading JSON: {e}")
            return []
    
    def _is_toc_table(self, table: Dict[str, Any]) -> bool:
        """Check if a table appears to be a TOC."""
        if not isinstance(table, dict):
            return False
        
        # Check table label
        label = table.get('label', '').lower()
        if 'document_index' in label or 'toc' in label or 'contents' in label:
            return True
        
        # Check table cells for TOC indicators
        cells = table.get('data', {}).get('table_cells', [])
        if not cells:
            return False
        
        # Look for TOC patterns in cell text
        cell_texts = [cell.get('text', '').lower() for cell in cells[:20]]  # Check first 20 cells
        
        toc_indicators = 0
        for text in cell_texts:
            if any(indicator in text for indicator in ['part i', 'item 1', 'item 2', 'business', 'page']):
                toc_indicators += 1
        
        return toc_indicators >= 3  # Need at least 3 TOC indicators
    
    def _extract_toc_from_table_fast(self, table: Dict[str, Any]) -> List[TOCEntry]:
        """Fast extraction of TOC entries from table structure."""
        # Direct access to cells
        data = table.get('data')
        if not data:
            return []
        
        cells = data.get('table_cells')
        if not cells:
            return []
        
        toc_entries = []
        
        # Use defaultdict for O(1) row grouping
        from collections import defaultdict
        rows = defaultdict(list)
        
        # Single pass to group cells by row
        for cell in cells:
            row_idx = cell.get('start_row_offset_idx', 0)
            rows[row_idx].append(cell)
        
        # Process rows in order with early filtering
        for row_idx in sorted(rows.keys()):
            row_cells = rows[row_idx]
            
            if len(row_cells) < 2:
                continue
            
            # Sort by column once and extract texts in single pass
            row_cells.sort(key=lambda x: x.get('start_col_offset_idx', 0))
            texts = [cell.get('text', '').strip() for cell in row_cells if cell.get('text')]
            
            if not texts:
                continue
            
            # Skip header rows (case-insensitive check)
            texts_lower = [t.lower() for t in texts]
            if any('page' in t for t in texts_lower):
                continue
            
            # Process texts for TOC entries
            self._process_row_texts_fast(texts, texts_lower, toc_entries)
        
        return toc_entries
    
    def _process_row_texts_fast(self, texts: List[str], texts_lower: List[str], toc_entries: List[TOCEntry]):
        """Fast processing of row texts for TOC entries."""
        for i, (text, text_lower) in enumerate(zip(texts, texts_lower)):
            if not text:
                continue
            
            # Check for PART markers (fast string operation)
            if text_lower.startswith('part'):
                toc_entries.append(TOCEntry(
                    level=1,
                    title=text,
                    confidence=0.95
                ))
            
            # Check for Item markers
            elif text_lower.startswith('item'):
                # Find description and page in single pass
                description = ''
                page_num = None
                
                for j, other_text in enumerate(texts):
                    if j != i and other_text:
                        if other_text.isdigit():
                            page_num = int(other_text)
                        elif len(other_text) > len(text) and not other_text.isdigit():
                            description = other_text
                            break  # Take first valid description
                
                title = f"{text}. {description}" if description else text
                toc_entries.append(TOCEntry(
                    level=2,
                    title=title,
                    page=page_num,
                    confidence=0.9
                ))

    def _extract_toc_from_table(self, table: Dict[str, Any]) -> List[TOCEntry]:
        """Legacy method - use _extract_toc_from_table_fast for better performance."""
        return self._extract_toc_from_table_fast(table)

    def extract_toc_from_json(self, json_content: Dict[str, Any]) -> List[TOCEntry]:
        """Extract TOC from structured JSON document."""
        toc_entries = []
        
        # Pre-cache frozensets for O(1) lookups
        toc_keys = frozenset(['toc', 'contents', 'table_of_contents'])
        section_keys = frozenset(['sections', 'chapters', 'parts'])
        
        def traverse_json(obj, level=1):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_lower = key.lower()
                    
                    if key_lower in toc_keys:
                        if isinstance(value, list):
                            toc_entries.extend(self._parse_json_toc_list(value))
                        continue
                    
                    if key_lower in section_keys and isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                title = item.get('title', item.get('name', f'Section {i+1}'))
                                page = item.get('page', item.get('page_number'))
                                toc_entries.append(TOCEntry(
                                    level=level,
                                    title=title,
                                    page=page,
                                    section_id=item.get('id', item.get('section_id')),
                                    confidence=0.9
                                ))
                                
                                if 'subsections' in item:
                                    traverse_json({'sections': item['subsections']}, level + 1)
                    
                    elif isinstance(value, (dict, list)):
                        traverse_json(value, level)
            
            elif isinstance(obj, list):
                for item in obj:
                    traverse_json(item, level)
        
        traverse_json(json_content)
        return toc_entries
    
    def _parse_json_toc_list(self, toc_list: List[Dict[str, Any]]) -> List[TOCEntry]:
        """Parse a list of TOC entries from JSON."""
        entries = []
        for item in toc_list:
            if isinstance(item, dict):
                entries.append(TOCEntry(
                    level=item.get('level', 1),
                    title=item.get('title', item.get('name', 'Untitled')),
                    page=item.get('page', item.get('page_number')),
                    section_id=item.get('id', item.get('section_id')),
                    confidence=0.95
                ))
        return entries