from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path


class BaseProcessor(ABC):
    """Base class for content processors."""
    
    @abstractmethod
    def process(self, content: str, **kwargs) -> Any:
        pass


class BaseFileProcessor(ABC):
    """Base class for file processors."""
    
    @abstractmethod  
    def process_file(self, file_path: Path) -> Any:
        pass
    
    @abstractmethod
    def supports_file_type(self, file_path: Path) -> bool:
        pass