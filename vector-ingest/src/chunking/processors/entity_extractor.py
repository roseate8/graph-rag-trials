"""Entity extraction processor for populating chunk metadata fields."""

import re
from typing import List, Dict, Any, Set
from datetime import datetime
from .base import BaseProcessor


class EntityExtractor(BaseProcessor):
    """Extract entities from text content to populate chunk metadata."""
    
    def __init__(self):
        """Initialize entity extraction patterns."""
        super().__init__()
        
        # Compile regex patterns once for O(1) access and better performance
        self.compiled_date_patterns = [
            re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', re.IGNORECASE),
            re.compile(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', re.IGNORECASE),
            re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b', re.IGNORECASE),
            re.compile(r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{2,4}\b', re.IGNORECASE),
            re.compile(r'\bQ[1-4]\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b\d{4}\b', re.IGNORECASE)
        ]
        
        self.compiled_time_period_patterns = [
            re.compile(r'\b(?:Q[1-4]|first quarter|second quarter|third quarter|fourth quarter|quarter)\b', re.IGNORECASE),
            re.compile(r'\b(?:fiscal year|FY|financial year)\s*\d{2,4}\b', re.IGNORECASE),
            re.compile(r'\byear ended?\s*(?:December|March|June|September)\s*\d{1,2},?\s*\d{2,4}\b', re.IGNORECASE),
            re.compile(r'\b(?:annually|quarterly|monthly|weekly|daily)\b', re.IGNORECASE),
            re.compile(r'\b(?:year-over-year|YoY|year-to-date|YTD)\b', re.IGNORECASE)
        ]
        
        self.compiled_metric_patterns = [
            re.compile(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|T))?\b', re.IGNORECASE),
            re.compile(r'\b\d+(?:\.\d+)?%\b', re.IGNORECASE),
            re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand|M|B|T|K)\b', re.IGNORECASE),
            re.compile(r'\b(?:revenue|income|profit|loss|earnings|sales|costs|expenses|assets|liabilities|equity)\b', re.IGNORECASE),
            re.compile(r'\b(?:EBITDA|ROI|ROE|ROA|EPS|P/E)\b', re.IGNORECASE),
            re.compile(r'\b(?:growth|increase|decrease|decline|improvement|deterioration)\b', re.IGNORECASE)
        ]
        
        self.compiled_region_patterns = [
            re.compile(r'\b(?:United States|US|USA|America|North America)\b', re.IGNORECASE),
            re.compile(r'\b(?:Europe|European Union|EU)\b', re.IGNORECASE),
            re.compile(r'\b(?:Asia|Asia-Pacific|APAC)\b', re.IGNORECASE),
            re.compile(r'\b(?:China|Japan|Korea|India|Australia)\b', re.IGNORECASE),
            re.compile(r'\b(?:Canada|Mexico|Brazil|Argentina)\b', re.IGNORECASE),
            re.compile(r'\b(?:UK|United Kingdom|Germany|France|Italy|Spain)\b', re.IGNORECASE),
            re.compile(r'\b(?:Middle East|Africa|Latin America)\b', re.IGNORECASE)
        ]
        
        # Use frozenset for O(1) lookups instead of regular set
        self.org_suffixes = frozenset({
            'Inc', 'Corp', 'Corporation', 'Company', 'Co', 'LLC', 'Ltd', 'Limited',
            'Partners', 'LP', 'LLP', 'Group', 'Holdings', 'Enterprises', 'Solutions',
            'Services', 'Systems', 'Technologies', 'Tech', 'Consulting', 'Associates'
        })
        
        self.person_titles = frozenset({
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Professor',
            'CEO', 'CFO', 'CTO', 'COO', 'CMO', 'CRO',
            'President', 'Vice President', 'VP', 'SVP', 'EVP',
            'Director', 'Managing Director', 'Executive Director',
            'Manager', 'Senior Manager', 'General Manager',
            'Chairman', 'Chair', 'Chairwoman', 'Chairperson'
        })
        
        # Pre-compile year validation regex for better performance
        self.year_pattern = re.compile(r'^\d{4}$')
        
        # Cache for common words to exclude - frozenset for O(1) lookup
        self.exclude_words = frozenset({
            'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'Or', 'But',
            'In', 'On', 'At', 'To', 'For', 'Of', 'With', 'By', 'From', 'As',
            'Company', 'Corporation', 'Inc', 'LLC', 'Ltd'
        })
    
    def process(self, content: str, **kwargs) -> Dict[str, Any]:
        """
        Extract entities from text content.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary containing extracted entities by category
        """
        if not content or not content.strip():
            return {
                'people': [],
                'organizations': [],
                'dates': [],
                'regions': [],
                'metrics': [],
                'time_periods': []
            }
        
        return {
            'people': self._extract_people(content),
            'organizations': self._extract_organizations(content),
            'dates': self._extract_dates(content),
            'regions': self._extract_regions(content),
            'metrics': self._extract_metrics(content),
            'time_periods': self._extract_time_periods(content)
        }
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract date references from text."""
        dates = set()
        
        # Use compiled patterns for better performance
        for pattern in self.compiled_date_patterns:
            for match in pattern.findall(text):
                cleaned_date = self._clean_date(match)
                if cleaned_date:
                    dates.add(cleaned_date)
        
        return sorted(dates)
    
    def _clean_date(self, date_str: str) -> str:
        """Clean and validate date string."""
        date_str = date_str.strip()
        
        # Use pre-compiled pattern for year validation
        if self.year_pattern.match(date_str):
            year = int(date_str)
            if 1900 <= year <= 2030:
                return date_str
            return None
        
        return date_str
    
    def _extract_time_periods(self, text: str) -> List[str]:
        """Extract time period references."""
        periods = set()
        
        # Use compiled patterns and generator expression for better performance
        for pattern in self.compiled_time_period_patterns:
            periods.update(match.strip() for match in pattern.findall(text))
        
        return sorted(periods)
    
    def _extract_metrics(self, text: str) -> List[str]:
        """Extract financial and business metrics."""
        metrics = set()
        
        # Use compiled patterns for better performance
        for pattern in self.compiled_metric_patterns:
            metrics.update(match.strip() for match in pattern.findall(text))
        
        return sorted(metrics)
    
    def _extract_regions(self, text: str) -> List[str]:
        """Extract geographic regions and locations."""
        regions = set()
        
        # Use compiled patterns for better performance
        for pattern in self.compiled_region_patterns:
            regions.update(match.strip() for match in pattern.findall(text))
        
        return sorted(regions)
    
    def _extract_people(self, text: str) -> List[str]:
        """Extract person names from text."""
        people = set()
        words = text.split()
        words_len = len(words)
        
        # Single pass through words with optimized bounds checking
        for i, word in enumerate(words):
            word_clean = word.rstrip('.,;:')
            
            # O(1) lookup for title check
            if word_clean in self.person_titles:
                if i + 1 < words_len:
                    next_word = words[i + 1].rstrip('.,;:')
                    if self._is_likely_name(next_word):
                        if i + 2 < words_len:
                            third_word = words[i + 2].rstrip('.,;:')
                            if self._is_likely_name(third_word):
                                people.add(f"{word_clean} {next_word} {third_word}")
                            else:
                                people.add(f"{word_clean} {next_word}")
                        else:
                            people.add(f"{word_clean} {next_word}")
            
            elif self._is_likely_name(word_clean) and i + 1 < words_len:
                next_word = words[i + 1].rstrip('.,;:')
                if self._is_likely_name(next_word):
                    people.add(f"{word_clean} {next_word}")
        
        return sorted(people)
    
    def _is_likely_name(self, word: str) -> bool:
        """Check if word is likely a person name."""
        # Early return for performance
        if not word or len(word) < 2 or not word[0].isupper() or not word.isalpha():
            return False
        
        # O(1) lookup using pre-defined frozenset
        return word not in self.exclude_words
    
    def _extract_organizations(self, text: str) -> List[str]:
        """Extract organization names from text."""
        organizations = set()
        words = text.split()
        
        for i, word in enumerate(words):
            word_clean = word.rstrip('.,;:')
            
            # O(1) lookup for organization suffixes
            if word_clean in self.org_suffixes:
                org_parts = []
                j = i - 1
                min_j = max(0, i - 4)  # Pre-calculate boundary
                
                # Optimized backward collection with pre-calculated boundary
                while j >= min_j:
                    prev_word = words[j].rstrip('.,;:')
                    if prev_word and prev_word[0].isupper() and prev_word.isalpha():
                        org_parts.insert(0, prev_word)
                        j -= 1
                    else:
                        break
                
                if org_parts:
                    organizations.add(f"{' '.join(org_parts)} {word_clean}")
        
        return sorted(organizations)