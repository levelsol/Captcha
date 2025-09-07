# hcaptcha_challenger/tools/image_classifier.py
import os
from pathlib import Path
from typing import Union

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageBinaryChallenge, DEFAULT_SCOT_MODEL
from hcaptcha_challenger.tools.common import parse_json_from_response
from hcaptcha_challenger.tools.reasoner import _Reasoner
from hcaptcha_challenger.tools.ollama_client import OllamaClient

SYSTEM_INSTRUCTION = """
You are an AI assistant that solves image-based challenges. You need to analyze a 3x3 grid of images and identify which images match the given prompt.

The grid is arranged as follows:
[0,0] [0,1] [0,2]
[1,0] [1,1] [1,2] 
[2,0] [2,1] [2,2]

You must return your answer in the following JSON format:
```json
{
  "challenge_prompt": "description of the challenge",
  "coordinates": [
    {"box_2d": [row, col]},
    {"box_2d": [row, col]}
  ]
}
```

Only return coordinates for images that match the prompt. Analyze carefully and be precise in your selections.
"""


class ImageClassifier(_Reasoner[SCoTModelType]):
    """
    A classifier that uses Ollama LLaVA models to analyze and solve image-based challenges.

    This class provides functionality to process screenshots of binary image challenges
    (typically grid-based selection challenges) and determines the correct answer coordinates.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: SCoTModelType = DEFAULT_SCOT_MODEL,
        constraint_response_schema: bool = False,
    ):
        super().__init__(ollama_url, model, constraint_response_schema)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def invoke_async(
        self,
        challenge_screenshot: Union[str, Path, os.PathLike],
        *,
        constraint_response_schema: bool | None = None,
        **kwargs,
    ) -> ImageBinaryChallenge:
        """
        Process an image challenge and return the solution coordinates.

        Args:
            challenge_screenshot: The image file containing the challenge to solve
            constraint_response_schema: Whether to use constraint response schema

        Returns:
            ImageBinaryChallenge: Object containing the solution coordinates
        """
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        async with OllamaClient(self._ollama_url) as client:
            # Encode image to base64
            image_b64 = client._encode_image(challenge_screenshot)
            
            # Prepare messages for chat format
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION
                },
                {
                    "role": "user", 
                    "content": "Please analyze this 3x3 grid image and identify which images match the challenge prompt. Return the coordinates in the specified JSON format."
                }
            ]

            # Set options for generation
            options = self._format_options(
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1024)
            )

            try:
                # Call Ollama API
                response = await client.chat(
                    model=model_to_use,
                    messages=messages,
                    images=[image_b64],
                    options=options
                )
                
                self._response = response
                response_text = response.get("message", {}).get("content", "")
                
                # Parse JSON from response
                parsed_result = parse_json_from_response(response_text)
                
                return ImageBinaryChallenge(**parsed_result)
                
            except Exception as e:
                logger.error(f"Error calling Ollama API: {str(e)}")
                # Return default response on error
                return ImageBinaryChallenge(
                    challenge_prompt="error occurred",
                    coordinates=[{"box_2d": [1, 1]}]
                )
