import json
from abc import abstractmethod, ABC
from pathlib import Path
from typing import TypeVar, Generic, Dict, Any

from loguru import logger

from hcaptcha_challenger.tools.common import run_sync

M = TypeVar("M")


class _Reasoner(ABC, Generic[M]):

    def __init__(
        self, 
        ollama_url: str = "http://localhost:11434", 
        model: M | None = None, 
        constraint_response_schema: bool = False
    ):
        self._ollama_url: str = ollama_url
        self._model: M | None = model
        self._constraint_response_schema = constraint_response_schema
        self._response = None

    def cache_response(self, path: Path):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(self._response, 'model_dump'):
                response_data = self._response.model_dump(mode="json")
            else:
                response_data = {"response": str(self._response)}
            
            path.write_text(
                json.dumps(response_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(e)

    @abstractmethod
    async def invoke_async(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _format_options(temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """Format options for Ollama API."""
        options = {
            "temperature": temperature,
            "num_predict": kwargs.get("max_tokens", 2048),
        }
        
        # Add other Ollama-specific options if needed
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            options["top_k"] = kwargs["top_k"]
            
        return options

    # for backward compatibility
    def invoke(self, *args, **kwargs):
        return run_sync(self.invoke_async(*args, **kwargs))