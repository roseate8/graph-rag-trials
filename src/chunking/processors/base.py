from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path


class BaseProcessor(ABC):
    """Base class for all document processors."""
    
    @abstractmethod
    def process(self, content: str, **kwargs) -> Any:
        """Process content and return result."""
        pass


class BaseFileProcessor(ABC):
    """Base class for file-based processors."""
    
    @abstractmethod
    def process_file(self, file_path: Path) -> Any:
        """Process a file and return result."""
        pass
    
    @abstractmethod
    def supports_file_type(self, file_path: Path) -> bool:
        """Check if processor supports this file type."""
        pass