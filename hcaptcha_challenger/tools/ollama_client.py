import base64
import json
import asyncio
from pathlib import Path
from typing import Union, Dict, Any, Optional
import aiohttp
from loguru import logger


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def chat(
        self,
        model: str,
        messages: list,
        images: Optional[list] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send a chat request to Ollama API."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        if images:
            payload["images"] = images
            
        if options:
            payload["options"] = options

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise

    async def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[list] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send a generate request to Ollama API."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        if images:
            payload["images"] = images
            
        if options:
            payload["options"] = options

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise