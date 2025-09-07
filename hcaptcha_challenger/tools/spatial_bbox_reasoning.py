import os
from pathlib import Path
from typing import Union

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageBboxChallenge, DEFAULT_SCOT_MODEL
from hcaptcha_challenger.tools.common import parse_json_from_response
from hcaptcha_challenger.tools.reasoner import _Reasoner
from hcaptcha_challenger.tools.ollama_client import OllamaClient

SYSTEM_INSTRUCTIONS = """
Analyze the input image (which includes a visible coordinate grid) and the accompanying challenge prompt text.

Tasks:
1. Interpret the challenge prompt to understand what needs to be identified
2. Identify the precise target area on the main challenge canvas
3. Determine the minimal bounding box that encloses the target
4. Output the absolute pixel coordinates based on the coordinate grid

Response format:
```json
{
    "challenge_prompt": "description of the task",
    "bounding_boxes": {
      "top_left_x": 148,
      "top_left_y": 260,
      "bottom_right_x": 235,
      "bottom_right_y": 345
    }
}
```
"""


class SpatialBboxReasoner(_Reasoner[SCoTModelType]):

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
        grid_divisions: Union[str, Path, os.PathLike],
        auxiliary_information: str | None = "",
        constraint_response_schema: bool | None = None,
        **kwargs,
    ) -> ImageBboxChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        async with OllamaClient(self._ollama_url) as client:
            # Encode both images
            challenge_b64 = client._encode_image(challenge_screenshot)
            grid_b64 = client._encode_image(grid_divisions)
            
            # Prepare prompt
            user_content = "Analyze these images to identify the target area and provide bounding box coordinates. The first image shows the challenge, the second shows the coordinate grid."
            if auxiliary_information:
                user_content += f"\n\nAdditional information: {auxiliary_information}"
            
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTIONS
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

            options = self._format_options(
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1024)
            )

            try:
                response = await client.chat(
                    model=model_to_use,
                    messages=messages,
                    images=[challenge_b64, grid_b64],
                    options=options
                )
                
                self._response = response
                response_text = response.get("message", {}).get("content", "")
                
                # Parse JSON response
                parsed_result = parse_json_from_response(response_text)
                
                # Ensure we have the required format
                if "challenge_prompt" not in parsed_result:
                    parsed_result["challenge_prompt"] = "bounding box challenge"
                
                if "bounding_boxes" not in parsed_result:
                    # Create default bounding box
                    parsed_result["bounding_boxes"] = {
                        "top_left_x": 150,
                        "top_left_y": 150,
                        "bottom_right_x": 350,
                        "bottom_right_y": 350
                    }
                
                return ImageBboxChallenge(**parsed_result)
                
            except Exception as e:
                logger.error(f"Error in spatial bbox reasoning: {str(e)}")
                return ImageBboxChallenge(
                    challenge_prompt="error occurred",
                    bounding_boxes={
                        "top_left_x": 150,
                        "top_left_y": 150,
                        "bottom_right_x": 350,
                        "bottom_right_y": 350
                    }
                )