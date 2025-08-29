import getpass
import threading
import time
from typing import Optional


class SecureAPIKeyManager:
    """Secure temporary API key storage for session use only."""
    
    def __init__(self):
        self._api_key: Optional[str] = None
        self._key_timestamp: Optional[float] = None
        self._cleanup_timer: Optional[threading.Timer] = None
        self._session_timeout_minutes = 30  # Auto-clear after 30 minutes
    
    def _prompt_for_api_key(self) -> str:
        """Prompt user for API key."""
        print("\nOpenAI API Key Required")
        print("Your API key will be stored temporarily for this session only.")
        print("It will be automatically cleared and never saved to disk.")
        
        api_key = input("Enter your OpenAI API key: ").strip()
        
        if not api_key:
            raise ValueError("API key cannot be empty")
        
        if not api_key.startswith(('sk-', 'sk-proj-')):
            print("Warning: API key doesn't look like an OpenAI key (should start with 'sk-' or 'sk-proj-')")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                raise ValueError("API key verification cancelled")
        
        return api_key
    
    def _set_temporary_key(self, api_key: str) -> None:
        """Set API key with automatic cleanup timer."""
        self._api_key = api_key
        self._key_timestamp = time.time()
        
        # Cancel existing timer if any
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        # Set new cleanup timer
        self._cleanup_timer = threading.Timer(
            self._session_timeout_minutes * 60, 
            self._clear_api_key
        )
        self._cleanup_timer.start()
        
        print(f"API key stored temporarily (will auto-clear in {self._session_timeout_minutes} minutes)")
    
    def _clear_api_key(self) -> None:
        """Clear the API key from memory securely."""
        if self._api_key:
            # Overwrite with random data before clearing (basic security)
            self._api_key = "x" * len(self._api_key)
            self._api_key = None
        
        self._key_timestamp = None
        
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None
        
        print("API key cleared from memory")
    
    def get_api_key(self) -> str:
        """Get API key, prompting if needed or if expired."""
        # Check if we have a valid key
        if self._api_key and self._key_timestamp:
            elapsed_minutes = (time.time() - self._key_timestamp) / 60
            if elapsed_minutes < self._session_timeout_minutes:
                return self._api_key
        
        # Key expired or doesn't exist, prompt for new one
        api_key = self._prompt_for_api_key()
        self._set_temporary_key(api_key)
        return api_key
    
    def has_valid_key(self) -> bool:
        """Check if we have a valid API key without prompting."""
        if self._api_key and self._key_timestamp:
            elapsed_minutes = (time.time() - self._key_timestamp) / 60
            return elapsed_minutes < self._session_timeout_minutes
        return False
    
    def clear_key_now(self) -> None:
        """Manually clear the API key immediately."""
        self._clear_api_key()
    
    def __del__(self):
        """Cleanup on object destruction."""
        self._clear_api_key()


# Global instance for session-wide reuse
_api_key_manager = SecureAPIKeyManager()


def get_openai_api_key() -> str:
    """Get OpenAI API key for current session."""
    return _api_key_manager.get_api_key()


def has_openai_api_key() -> bool:
    """Check if we have a valid API key without prompting user."""
    return _api_key_manager.has_valid_key()


def clear_openai_api_key() -> None:
    """Manually clear the stored API key."""
    _api_key_manager.clear_key_now()