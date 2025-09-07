import os
from pathlib import Path
from typing import Union

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import SCoTModelType, ImageAreaSelectChallenge, DEFAULT_SCOT_MODEL
from hcaptcha_challenger.tools.common import parse_json_from_response
from hcaptcha_challenger.tools.reasoner import _Reasoner
from hcaptcha_challenger.tools.ollama_client import OllamaClient

THINKING_PROMPT = """
You are an expert at solving visual challenges. Analyze the provided images to identify specific objects or areas based on the challenge prompt.

Rules for Visual Analysis:
- Process Global Context before Local Details
- Maintain awareness of the overall scene
- Ensure local interpretations are consistent with global context
- Use systematic, top-down analysis

Workflow:
1. Identify the challenge prompt/task from the image
2. Determine what needs to be identified and where it's located
3. Based on the coordinate system shown, determine the absolute pixel positions
4. Return coordinates in JSON format

Response format:
```json
{
  "challenge_prompt": "Task description", 
  "points": [
    {"x": x1, "y": y1},
    {"x": x2, "y": y2}
  ]
}
```

Analyze both the challenge image and the coordinate grid image to determine precise pixel coordinates.
"""


class SpatialPointReasoner(_Reasoner[SCoTModelType]):

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
    ) -> ImageAreaSelectChallenge:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        async with OllamaClient(self._ollama_url) as client:
            # Encode both images
            challenge_b64 = client._encode_image(challenge_screenshot)
            grid_b64 = client._encode_image(grid_divisions)
            
            # Prepare prompt
            user_content = "Analyze these images to solve the spatial reasoning challenge. The first image shows the challenge, the second shows the coordinate grid."
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
                    parsed_result["challenge_prompt"] = "spatial point selection"
                
                if "points" not in parsed_result:
                    # Try to extract from coordinates if present
                    if "coordinates" in parsed_result:
                        parsed_result["points"] = [
                            {"x": coord["box_2d"][0] * 100, "y": coord["box_2d"][1] * 100}
                            for coord in parsed_result["coordinates"]
                        ]
                    else:
                        parsed_result["points"] = [{"x": 200, "y": 200}]
                
                return ImageAreaSelectChallenge(**parsed_result)
                
            except Exception as e:
                logger.error(f"Error in spatial point reasoning: {str(e)}")
                return ImageAreaSelectChallenge(
                    challenge_prompt="error occurred",
                    points=[{"x": 200, "y": 200}]
                )