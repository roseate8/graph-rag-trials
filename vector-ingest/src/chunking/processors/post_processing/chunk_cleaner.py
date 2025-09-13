import re
from typing import List


class ChunkCleaner:
    """Clean and post-process text chunks removing artifacts and noise."""
    
    def __init__(self):
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for better performance."""
        # Navigation patterns - case insensitive, line-based
        nav_patterns = [
            r'skip\s+to\s+(main\s+)?content',
            r'skip\s+to\s+homepage', 
            r'toggle\s+navigation',
            r'breadcrumb',
            r'^\s*(home\s*[>›»]|[>›»])\s*',
        ]
        self.nav_regex = re.compile('|'.join(f'({p})' for p in nav_patterns), 
                                   re.IGNORECASE | re.MULTILINE)
        
        # Glyph artifacts - single pass cleanup
        # Unicode glyph replacement characters
        unicode_glyph_pattern = r'[\uFFFD\uf0b7\uf0a7]|[\u0000-\u0008\u000B\u000C\u000E-\u001F]'
        # PDF parsing GLYPH artifacts with font specifications
        pdf_glyph_pattern = r'GLYPH<[^>]*>'
        self.glyph_regex = re.compile(f'{unicode_glyph_pattern}|{pdf_glyph_pattern}')
        
        # Whitespace normalization
        self.multi_space_regex = re.compile(r'[ \t]{2,}')
        self.multi_newline_regex = re.compile(r'\n{3,}')
        self.trailing_space_regex = re.compile(r'[ \t]+$', re.MULTILINE)
        
        # Noise patterns for standalone lines
        noise_patterns = [
            r'^\s*[-•·▪▫◦‣⁃]+\s*$',     # Standalone bullets
            r'^\s*[─━═_]{3,}\s*$',      # Lines/underscores
            r'^\s*\.{3,}\s*$',          # Multiple dots
            r'^\s*[^\w\s]{1,2}\s*$',    # Single/double special chars
        ]
        self.noise_regex = re.compile('|'.join(noise_patterns), re.MULTILINE)
    
    def clean_chunk(self, content: str) -> str:
        """Clean a text chunk removing artifacts and noise."""
        if not content or not content.strip():
            return ""
        
        # Single-pass cleaning for efficiency  
        # 1. Remove glyph artifacts (NOTE: GLYPH artifacts are now cleaned at pipeline start)
        content = self.glyph_regex.sub(' ', content)  # Keep as safety net
        
        # 2. Remove navigation lines (but preserve content lines)
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines and navigation
            if not line_stripped:
                cleaned_lines.append('')  # Preserve paragraph breaks
                continue
                
            # Check if entire line is navigation/noise
            if (len(line_stripped) <= 30 and  # Short lines more likely to be nav
                self.nav_regex.search(line_stripped)):
                continue
                
            # Remove standalone noise lines
            if self.noise_regex.match(line):
                continue
                
            cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        # 3. Normalize whitespace efficiently
        content = self.multi_space_regex.sub(' ', content)
        content = self.trailing_space_regex.sub('', content)
        content = self.multi_newline_regex.sub('\n\n', content)
        
        return content.strip()