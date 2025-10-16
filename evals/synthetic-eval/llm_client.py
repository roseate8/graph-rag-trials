"""
Shared LLM client wrapper for consistent OpenAI API interaction.

Eliminates duplicate client initialization and standardizes LLM calls.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List
import asyncio

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Centralized LLM client with both sync and async support.

    Handles:
    - Single client instance (sync and async)
    - Automatic token parameter selection (GPT-5 vs others)
    - JSON response parsing with error handling
    - Consistent error logging
    """

    def __init__(self, api_key_manager, config):
        """
        Initialize LLM client.

        Args:
            api_key_manager: SecureAPIKeyManager instance
            config: SyntheticEvalConfig instance
        """
        self.api_key_manager = api_key_manager
        self.config = config

        # Initialize clients lazily
        self._sync_client = None
        self._async_client = None

        logger.info(f"Initialized LLMClient with model: {config.model_name}")

    def _get_sync_client(self):
        """Get or create synchronous OpenAI client."""
        if self._sync_client is None:
            import openai
            api_key = self.api_key_manager.get_api_key()
            self._sync_client = openai.OpenAI(api_key=api_key)
        return self._sync_client

    async def _get_async_client(self):
        """Get or create asynchronous OpenAI client."""
        if self._async_client is None:
            import openai
            api_key = self.api_key_manager.get_api_key()
            self._async_client = openai.AsyncOpenAI(api_key=api_key)
        return self._async_client

    def _prepare_params(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Prepare LLM parameters with correct token parameter for model.

        Args:
            messages: List of message dicts
            **kwargs: Additional parameters to pass to API

        Returns:
            Complete parameters dict ready for API call
        """
        params = {
            "model": self.config.model_name,
            "messages": messages,
            **kwargs
        }

        # Handle token parameter based on model
        if "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
            if self.config.model_name.startswith("gpt-5"):
                params["max_completion_tokens"] = self.config.max_completion_tokens
            else:
                params["max_tokens"] = self.config.max_tokens

        return params

    @staticmethod
    def parse_json_response(response_text: str) -> Any:
        """
        Parse JSON response from LLM, handling markdown code blocks.

        Args:
            response_text: Raw LLM response text

        Returns:
            Parsed JSON object (dict, list, etc.)

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        # Clean markdown code block markers if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            # Remove opening ```json or ```
            cleaned = re.sub(r'^```json?\s*\n?', '', cleaned)
            # Remove closing ```
            cleaned = re.sub(r'\n?```\s*$', '', cleaned)

        return json.loads(cleaned.strip())

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Synchronous chat completion.

        Args:
            messages: List of message dicts (role, content)
            **kwargs: Additional API parameters

        Returns:
            Response text content
        """
        client = self._get_sync_client()
        params = self._prepare_params(messages, **kwargs)

        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Asynchronous chat completion.

        Args:
            messages: List of message dicts (role, content)
            **kwargs: Additional API parameters

        Returns:
            Response text content
        """
        client = await self._get_async_client()
        params = self._prepare_params(messages, **kwargs)

        response = await client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    def chat_completion_json(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Any:
        """
        Synchronous chat completion with automatic JSON parsing.

        Args:
            messages: List of message dicts (role, content)
            **kwargs: Additional API parameters

        Returns:
            Parsed JSON object

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        response_text = self.chat_completion(messages, **kwargs)
        return self.parse_json_response(response_text)

    async def chat_completion_json_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Any:
        """
        Asynchronous chat completion with automatic JSON parsing.

        Args:
            messages: List of message dicts (role, content)
            **kwargs: Additional API parameters

        Returns:
            Parsed JSON object

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        response_text = await self.chat_completion_async(messages, **kwargs)
        return self.parse_json_response(response_text)

    def handle_llm_error(
        self,
        error: Exception,
        context: str = "",
        response_text: Optional[str] = None
    ) -> None:
        """
        Standardized error handling for LLM calls.

        Args:
            error: The exception that was raised
            context: Context string for logging (e.g., "chunk_id: abc123")
            response_text: Optional response text for debugging
        """
        if isinstance(error, json.JSONDecodeError):
            logger.error(f"Failed to parse LLM response as JSON{' for ' + context if context else ''}: {error}")
            if response_text:
                logger.error(f"Response was: {response_text[:500]}")
        else:
            logger.error(f"Error in LLM call{' for ' + context if context else ''}: {error}", exc_info=True)
