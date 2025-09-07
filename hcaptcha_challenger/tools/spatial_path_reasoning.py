import os
from pathlib import Path
from typing import Union

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageDragDropChallenge, DEFAULT_SCOT_MODEL
from hcaptcha_challenger.tools.common import parse_json_from_response
from hcaptcha_challenger.tools.reasoner import _Reasoner
from hcaptcha_challenger.tools.ollama_client import OllamaClient

THINKING_PROMPT = """
You are an expert at solving drag-and-drop puzzle challenges. Analyze the provided images to determine where objects should be dragged.

Rules for Drag-Drop Analysis:
1. Identify what needs to be dragged (usually on the right side)
2. Identify where it should be placed (usually on the left canvas)
3. Determine start and end coordinates based on the coordinate grid
4. Consider the shape, pattern, and context clues

Response format:
```json
{
  "challenge_prompt": "Task description",
  "paths": [
    {"start_point": {"x": x1, "y": y1}, "end_point": {"x": x2, "y": y2}}
  ]
}
```

Analyze both the challenge image and coordinate grid to determine precise pixel coordinates for the drag path.
"""


class SpatialPathReasoner(_Reasoner[SCoTModelType]):

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
    ) -> ImageDragDropChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        async with OllamaClient(self._ollama_url) as client:
            # Encode both images
            challenge_b64 = client._encode_image(challenge_screenshot)
            grid_b64 = client._encode_image(grid_divisions)
            
            # Prepare prompt
            user_content = "Analyze these drag-and-drop challenge images. The first shows the challenge, the second shows the coordinate grid. Determine the drag path from source to destination."
            if auxiliary_information:
                user_content += f"\n\nAdditional information: {auxiliary_information}"
            
            messages = [
                {
                    "role": "system",
                    "content": THINKING_PROMPT
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
                    parsed_result["challenge_prompt"] = "drag and drop challenge"
                
                if "paths" not in parsed_result:
                    # Create default path
                    parsed_result["paths"] = [{
                        "start_point": {"x": 500, "y": 200},
                        "end_point": {"x": 300, "y": 300}
                    }]
                
                return ImageDragDropChallenge(**parsed_result)
                
            except Exception as e:
                logger.error(f"Error in spatial path reasoning: {str(e)}")
                return ImageDragDropChallenge(
                    challenge_prompt="error occurred",
                    paths=[{
                        "start_point": {"x": 500, "y": 200},
                        "end_point": {"x": 300, "y": 300}
                    }]
                )