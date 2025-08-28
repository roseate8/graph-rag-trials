import re
from typing import List
from pathlib import Path


class TextPreprocessor:
    """Clean and preprocess document content removing artifacts and noise."""
    
    def __init__(self):
        # Compile regex patterns once for efficiency
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
        self.glyph_regex = re.compile(r'[\uFFFD\uf0b7\uf0a7]|[\u0000-\u0008\u000B\u000C\u000E-\u001F]')
        
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
    
    def clean_content(self, content: str) -> str:
        """Main preprocessing method to clean content efficiently."""
        if not content or not content.strip():
            return ""
        
        # Single-pass cleaning for efficiency
        # 1. Remove glyph artifacts
        content = self.glyph_regex.sub(' ', content)
        
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
    
    
    def preprocess_file(self, file_path: Path) -> str:
        """Read and preprocess a single file."""
        try:
            # Handle different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            
            for encoding in encodings:
                try:
                    content = file_path.read_text(encoding=encoding)
                    return self.clean_content(content)
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            return self.clean_content(content)
            
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {str(e)}")
    
    def discover_input_files(self, input_dir: Path) -> List[Path]:
        """Discover all supported files in input directory."""
        if not input_dir.exists():
            return []
        
        # Use iterdir() for better performance than multiple glob calls
        supported_exts = {'.txt', '.md', '.html', '.htm'}
        files = [f for f in input_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in supported_exts]
        
        return sorted(files)