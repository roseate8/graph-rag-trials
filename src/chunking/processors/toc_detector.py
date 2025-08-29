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
    
    def _compile_patterns(self):
        """Compile regex patterns for systematic TOC detection."""
        # Phase 1: Keyword-based matching
        self.toc_keywords = [
            r'table\s+of\s+contents?',
            r'\btoc\b',
            r'contents?',
            r'index',
            r'outline',
            r'summary',
            r'chapter\s+list'
        ]
        self.toc_keyword_regex = re.compile(
            '|'.join(f'({kw})' for kw in self.toc_keywords),
            re.IGNORECASE
        )
        
        # Simplified page number patterns - essential ones including markdown tables
        self.page_number_patterns = [
            # "Title ........ 15" (with dots or whitespace)
            re.compile(r'^(.+?)\s*\.{2,}\s*(\d+)\s*$'),
            # "Title     15" (significant whitespace)
            re.compile(r'^(.+?)\s{3,}(\d+)\s*$'),
            # Markdown table format: "| Item 1. | Title | 15 |"
            re.compile(r'^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(\d+)\s*\|?\s*$')
        ]
        
        # Simplified sequential patterns - just basic numbering
        self.sequential_patterns = [
            re.compile(r'^\s*(\d+)\.?\s+(.+)'),  # "1. Title" or "1 Title"
            re.compile(r'^\s*(\d+\.\d+)\.?\s+(.+)')  # "1.1. Title" for subsections
        ]
        
        # Markdown headings for fallback
        self.md_heading_regex = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
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
        """Phase 1: Use keyword-based matching and detect page number sequences."""
        candidates = []
        
        # Find keyword-triggered sections
        for i, line in enumerate(lines):
            if self.toc_keyword_regex.search(line.strip()):
                # Look for TOC content after keyword
                start_line = i
                end_line = self._find_toc_section_end(lines, start_line)
                
                if end_line > start_line:
                    section_lines = lines[start_line:end_line]
                    score = self._score_page_number_density(section_lines)
                    
                    if score > 0.2:  # At least 20% lines have page numbers (was 30%)
                        candidates.append(TOCCandidate(
                            start_line=start_line,
                            end_line=end_line,
                            lines=section_lines,
                            score=score + 0.5,  # Bonus for keyword match
                            reason="keyword_triggered"
                        ))
        
        # Find sections with high page number density (no keyword needed)
        window_size = 10
        for i in range(0, len(lines) - window_size, 5):
            window = lines[i:i + window_size]
            score = self._score_page_number_density(window)
            
            if score > 0.3:  # Lower threshold without keyword (was 0.5)
                # Extend window to capture full section
                start_line = i
                end_line = self._find_toc_section_end(lines, start_line)
                section_lines = lines[start_line:end_line]
                
                candidates.append(TOCCandidate(
                    start_line=start_line,
                    end_line=end_line,
                    lines=section_lines,
                    score=score,
                    reason="page_number_density"
                ))
        
        return candidates
    
    def _find_toc_section_end(self, lines: List[str], start: int) -> int:
        """Find the end of a TOC section starting from given line."""
        end = start + 1
        empty_count = 0
        
        for i in range(start + 1, len(lines)):
            line = lines[i].strip()
            
            # Stop at major section headers
            if line.startswith('#') or line.startswith('Chapter '):
                break
            
            # Count empty lines
            if not line:
                empty_count += 1
                if empty_count >= 3:  # 3 consecutive empty lines
                    break
            else:
                empty_count = 0
                end = i + 1
                
                # Stop if we hit content that doesn't look like TOC
                if len(line) > 100 and '.' not in line[-10:]:  # Long line without trailing dots/numbers
                    break
        
        return min(end, start + 50)  # Cap at reasonable size
    
    def _score_page_number_density(self, lines: List[str]) -> float:
        """Score section based on density of lines with page numbers."""
        if not lines:
            return 0.0
        
        page_number_lines = 0
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Check if line matches any page number pattern
            for pattern in self.page_number_patterns:
                if pattern.match(line_stripped):
                    page_number_lines += 1
                    break
        
        return page_number_lines / max(len([l for l in lines if l.strip()]), 1)
    
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
        """Check if lines show basic sequential numbering pattern."""
        numbers = []
        for line in lines:
            line_stripped = line.strip()
            for pattern in self.sequential_patterns:
                match = pattern.match(line_stripped)
                if match:
                    try:
                        num_str = match.group(1).replace('chapter', '').strip()
                        if '.' not in num_str:  # Skip subsections
                            num = int(num_str)
                            numbers.append(num)
                            break
                    except (ValueError, IndexError):
                        pass
        
        # Looser requirement: just need 2+ numbers
        return len(numbers) >= 2
    
    
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
            return candidates
        
        print(f"Starting LLM verification for {len(candidates)} candidate(s)")
        
        verified_candidates = []
        
        for i, candidate in enumerate(candidates, 1):
            print(f"Verifying candidate {i}/{len(candidates)}")
            
            try:
                # Get candidate text
                candidate_text = "\n".join(candidate.lines)
                
                # Call LLM for verification
                llm_result = self._verify_toc_with_openai(candidate_text)
                
                if not llm_result["success"]:
                    print(f"LLM verification failed: {llm_result.get('error', 'Unknown error')}")
                    # Keep candidate but mark as unverified
                    verified_candidates.append(candidate)
                    continue
                
                is_toc = llm_result.get("is_toc", False)
                confidence = llm_result.get("confidence", 0.0)
                reason = llm_result.get("reason", "No reason provided")
                
                if is_toc and confidence >= 0.5:
                    # LLM confirms this is a TOC with decent confidence
                    candidate.score = candidate.score + (confidence * 0.5)  # Boost score
                    verified_candidates.append(candidate)
                    print(f"LLM confirmed TOC (confidence: {confidence:.2f})")
                
                elif is_toc and confidence >= 0.3:
                    # LLM thinks it's a TOC but with low confidence - keep but don't boost
                    verified_candidates.append(candidate)
                    print(f"LLM thinks it's a TOC but uncertain (confidence: {confidence:.2f})")
                
                else:
                    # LLM doesn't think this is a TOC
                    print(f"LLM rejected as TOC (confidence: {confidence:.2f}) - {reason}")
                    # Don't add to verified candidates - effectively filters it out
                
            except Exception as e:
                print(f"Error during LLM verification: {str(e)}")
                # Keep candidate in case of unexpected errors
                verified_candidates.append(candidate)
        
        print(f"LLM verification complete: {len(verified_candidates)}/{len(candidates)} candidates retained")
        
        return verified_candidates
    
    def _verify_toc_with_openai(self, candidate_text: str) -> Dict[str, Any]:
        """Make direct API call to OpenAI to verify if text is a TOC."""
        try:
            # Check if we have API key (will prompt if needed)
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
            
            print(f"Calling OpenAI API...")
            
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
                tokens = result.get("usage", {}).get("total_tokens", 0)
                
                print(f"API call successful ({tokens} tokens)")
                
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
                        "usage_tokens": tokens
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
        """Phase 5: Optional validation by cross-checking against document headings."""
        if not candidates:
            return candidates
        
        # Extract all headings from document for validation
        document_headings = set()
        for line in all_lines:
            # Check for markdown headings
            heading_match = self.md_heading_regex.match(line)
            if heading_match:
                document_headings.add(heading_match.group(2).strip().lower())
        
        validated_candidates = []
        for candidate in candidates:
            validation_score = candidate.score
            
            if document_headings:
                # Check how many TOC entries match actual headings
                matches = 0
                total_entries = 0
                
                for line in candidate.lines:
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    
                    # Extract title from various patterns
                    title = None
                    for pattern in self.page_number_patterns:
                        match = pattern.match(line_stripped)
                        if match:
                            title = match.group(1).strip().lower()
                            break
                    
                    if not title:
                        for pattern in self.sequential_patterns:
                            match = pattern.match(line_stripped)
                            if match and len(match.groups()) >= 2:
                                title = match.group(2).strip().lower()
                                break
                    
                    if title:
                        total_entries += 1
                        if any(title in heading or heading in title for heading in document_headings):
                            matches += 1
                
                if total_entries > 0:
                    match_ratio = matches / total_entries
                    if match_ratio > 0.2:  # Lowered from 30% to 20% match
                        validation_score += 0.1
                    # Removed penalty for low matches - too harsh
            
            validated_candidates.append(TOCCandidate(
                start_line=candidate.start_line,
                end_line=candidate.end_line,
                lines=candidate.lines,
                score=validation_score,
                reason=candidate.reason + "_validated"
            ))
        
        return validated_candidates
    
    def extract_entries_from_candidate(self, candidate: TOCCandidate) -> List[TOCEntry]:
        """Extract structured TOC entries from a candidate."""
        entries = []
        
        for line_idx, line in enumerate(candidate.lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            entry = None
            
            # Try page number patterns first
            for pattern in self.page_number_patterns:
                match = pattern.match(line_stripped)
                if match:
                    # Handle different pattern formats
                    groups = match.groups()
                    if len(groups) == 3:
                        # Markdown table format: (prefix, title, page)
                        title = f"{groups[0].strip()} {groups[1].strip()}".strip()
                        page = int(groups[2]) if groups[2].isdigit() else None
                    else:
                        # Standard format: (title, page)
                        title = groups[0].strip()
                        page = int(groups[1]) if len(groups) > 1 and groups[1].isdigit() else None
                    
                    level = self._determine_level(title, line)
                    
                    entry = TOCEntry(
                        level=level,
                        title=title,
                        page=page,
                        line_number=candidate.start_line + line_idx,
                        confidence=0.9
                    )
                    break
            
            # Try sequential patterns if no page number found
            if not entry:
                for pattern in self.sequential_patterns:
                    match = pattern.match(line_stripped)
                    if match and len(match.groups()) >= 2:
                        prefix = match.group(1).strip()
                        title = match.group(2).strip()
                        level = self._determine_level_from_prefix(prefix)
                        
                        entry = TOCEntry(
                            level=level,
                            title=title,
                            line_number=candidate.start_line + line_idx,
                            confidence=0.7
                        )
                        break
            
            if entry:
                entries.append(entry)
        
        return entries
    
    def _determine_level(self, title: str, full_line: str) -> int:
        """Determine hierarchical level from title and line context."""
        # Count leading whitespace
        indent = len(full_line) - len(full_line.lstrip())
        level = max(1, (indent // 4) + 1)
        
        # Adjust based on title content
        if title.lower().startswith(('chapter', 'part', 'section')):
            level = max(level, 1)
        elif any(word in title.lower() for word in ['subsection', 'appendix']):
            level = max(level, 2)
        
        return min(level, 6)  # Cap at level 6
    
    def _determine_level_from_prefix(self, prefix: str) -> int:
        """Determine level from TOC entry prefix."""
        if '.' in prefix:
            return prefix.count('.') + 1
        elif prefix.lower().startswith('chapter'):
            return 1
        else:
            return 1
    
    def detect_toc(self, content: str, source_format: Optional[TOCFormat] = None) -> List[TOCEntry]:
        """Main method using systematic 6-phase approach."""
        if source_format is None:
            source_format = self.detect_format(content)
        
        if source_format == TOCFormat.JSON:
            try:
                json_data = json.loads(content)
                return self.extract_toc_from_json(json_data)
            except json.JSONDecodeError:
                pass
        
        # Systematic approach for text content
        lines = content.split('\n')
        
        # Phase 1: Pattern matching
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
    
    def extract_toc_from_json(self, json_content: Dict[str, Any]) -> List[TOCEntry]:
        """Extract TOC from structured JSON document."""
        toc_entries = []
        
        def traverse_json(obj, level=1):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in ['toc', 'contents', 'table_of_contents']:
                        if isinstance(value, list):
                            toc_entries.extend(self._parse_json_toc_list(value))
                        continue
                    
                    if key.lower() in ['sections', 'chapters', 'parts'] and isinstance(value, list):
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