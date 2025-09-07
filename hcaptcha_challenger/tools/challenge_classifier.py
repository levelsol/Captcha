import os
from pathlib import Path
from typing import Union

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import (
    FastShotModelType,
    ChallengeRouterResult,
    ChallengeTypeEnum,
    DEFAULT_FAST_SHOT_MODEL,
)
from hcaptcha_challenger.tools.common import parse_json_from_response
from hcaptcha_challenger.tools.reasoner import _Reasoner
from hcaptcha_challenger.tools.ollama_client import OllamaClient

CHALLENGE_CLASSIFIER_INSTRUCTIONS = """
Your task is to classify challenge questions into one of four types:

1. `image_label_single_select`: Requires clicking on a SINGLE specific area/object of an image based on a prompt
2. `image_label_multi_select`: Requires clicking on MULTIPLE areas/objects of an image based on a prompt
3. `image_drag_single`: Requires dragging a SINGLE puzzle piece/element to a specific location on an image
4. `image_drag_multi`: Requires dragging MULTIPLE puzzle pieces/elements to specific locations on an image

Rules:
- Output ONLY one of the four classification types listed above
- For clicking/selecting tasks:
  - If the question implies selecting ONE item/area, output `image_label_single_select`
  - If the question implies selecting MULTIPLE items/areas, output `image_label_multi_select`
  - If the question implies 9grid selection, output `image_label_multi_select`
- For dragging tasks:
  - If the question implies dragging ONE item/element, output `image_drag_single`
  - If the question implies dragging MULTIPLE items/elements, output `image_drag_multi`

Examples:
- "Please click on the object that is different from the others" -> `image_label_single_select`
- "Please click on the two elements that are identical" -> `image_label_multi_select`
- "Please drag the puzzle piece to complete the image" -> `image_drag_single`
- "Arrange all the shapes by dragging them to their matching outlines" -> `image_drag_multi`
"""

CHALLENGE_ROUTER_INSTRUCTION = """
Analyze the provided challenge image and identify:
1. The challenge prompt/task description
2. The appropriate challenge type

Return your response in this JSON format:
```json
{
    "challenge_prompt": "description of what the challenge is asking",
    "challenge_type": "one of: image_label_single_select, image_label_multi_select, image_drag_single, image_drag_multi"
}
```
"""


class ChallengeClassifier(_Reasoner[FastShotModelType]):

    def __init__(self, ollama_url: str = "http://localhost:11434", model: FastShotModelType = DEFAULT_FAST_SHOT_MODEL):
        super().__init__(ollama_url, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def invoke_async(
        self, challenge_screenshot: Union[str, Path, os.PathLike], **kwargs
    ) -> ChallengeTypeEnum:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        async with OllamaClient(self._ollama_url) as client:
            image_b64 = client._encode_image(challenge_screenshot)
            
            messages = [
                {
                    "role": "system",
                    "content": CHALLENGE_CLASSIFIER_INSTRUCTIONS
                },
                {
                    "role": "user",
                    "content": "Analyze this challenge image and classify it into one of the four types."
                }
            ]

            options = self._format_options(temperature=0.0)

            try:
                response = await client.chat(
                    model=model_to_use,
                    messages=messages,
                    images=[image_b64],
                    options=options
                )
                
                self._response = response
                response_text = response.get("message", {}).get("content", "").strip()
                
                # Extract challenge type from response
                for challenge_type in ChallengeTypeEnum:
                    if challenge_type.value in response_text.lower():
                        return challenge_type
                
                # Default fallback
                return ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT
                
            except Exception as e:
                logger.error(f"Error in challenge classification: {str(e)}")
                return ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT


class ChallengeRouter(_Reasoner[FastShotModelType]):
    def __init__(self, ollama_url: str = "http://localhost:11434", model: FastShotModelType = DEFAULT_FAST_SHOT_MODEL):
        super().__init__(ollama_url, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def invoke_async(
        self, challenge_screenshot: Union[str, Path, os.PathLike], **kwargs
    ) -> ChallengeRouterResult:
        model_to_use = kwargs.pop("model", self._model)
        if model_to_use is None:
            raise ValueError("Model must be provided either at initialization or via kwargs.")

        async with OllamaClient(self._ollama_url) as client:
            image_b64 = client._encode_image(challenge_screenshot)
            
            messages = [
                {
                    "role": "system", 
                    "content": CHALLENGE_ROUTER_INSTRUCTION
                },
                {
                    "role": "user",
                    "content": "Analyze this challenge image and extract the prompt and challenge type."
                }
            ]

            options = self._format_options(temperature=0.0)

            try:
                response = await client.chat(
                    model=model_to_use,
                    messages=messages,
                    images=[image_b64],
                    options=options
                )
                
                self._response = response
                response_text = response.get("message", {}).get("content", "")
                
                # Parse JSON response
                parsed_result = parse_json_from_response(response_text)
                
                # Ensure we have required fields
                if "challenge_prompt" not in parsed_result:
                    parsed_result["challenge_prompt"] = "unknown challenge"
                    
                if "challenge_type" not in parsed_result:
                    parsed_result["challenge_type"] = "image_label_single_select"
                
                return ChallengeRouterResult(**parsed_result)
                
            except Exception as e:
                logger.error(f"Error in challenge routing: {str(e)}")
                return ChallengeRouterResult(
                    challenge_prompt="unknown challenge",
                    challenge_type=ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT
                )