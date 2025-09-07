import asyncio
import json
import re
from typing import List, Any, Coroutine, TypeVar, Dict

from loguru import logger

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine as a sync function, handling different threading scenarios.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in this thread at all:
        return asyncio.run(coro)

    # If the loop is idle, run it directly
    if not loop.is_running():
        return loop.run_until_complete(coro)

    # If we get here, the loop *is* running in this thread.
    # We schedule the work on the loop's default executor:
    def _worker():
        # Create a fresh loop in this worker thread
        worker_loop = asyncio.new_event_loop()
        try:
            return worker_loop.run_until_complete(coro)
        finally:
            worker_loop.close()

    # Schedule on the default executor (None == use loop's ThreadPoolExecutor)
    future = loop.run_in_executor(None, _worker)
    # Block until it's done and return the result/raise
    return future.result()


def extract_json_blocks(text: str) -> List[str]:
    """
    Extract the contents of JSON code blocks surrounded by ```json and ``` from the text.

    Args:
        text: String containing possible JSON code blocks

    Returns:
        Extracted JSON content list, not containing ```json and ``` tags
    """
    try:
        # Use regular expressions to match the content between ```json and ``
        pattern = r"```json\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)

        # If no match is found, record the warning and return to the empty list
        if not matches:
            logger.warning("No JSON code blocks found in the provided text")
            return []

        return matches
    except Exception as e:
        logger.error(f"Error extracting JSON blocks: {str(e)}")
        return []


def extract_first_json_block(text: str) -> Any | None:
    """
    Extract the first JSON code block contents surrounded by ```JSON and ``` from the text.

    Args:
        text: String containing possible JSON code blocks

    Returns:
        The first JSON content extracted, if not found, returns None
    """
    if blocks := extract_json_blocks(text):
        return json.loads(blocks[0])


def parse_json_from_response(text: str) -> Dict[str, Any]:
    """
    Parse JSON from LLaVA response text, handling various formats.
    """
    try:
        # First try to extract JSON from code blocks
        if json_block := extract_first_json_block(text):
            return json_block
        
        # Try to find JSON-like content in the text
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
                
        # If no JSON found, try to extract coordinates from natural language
        coord_pattern = r'\[(\d+),\s*(\d+)\]'
        coordinates = re.findall(coord_pattern, text)
        
        if coordinates:
            return {
                "challenge_prompt": "extracted from response",
                "coordinates": [{"box_2d": [int(x), int(y)]} for x, y in coordinates]
            }
            
        # Last resort: try to extract numbers that could be coordinates
        number_pattern = r'(\d+)\s*,\s*(\d+)'
        numbers = re.findall(number_pattern, text)
        
        if numbers:
            return {
                "challenge_prompt": "extracted from response", 
                "coordinates": [{"box_2d": [int(x), int(y)]} for x, y in numbers[:3]]  # Limit to 3 coordinates
            }
            
        # If nothing works, return default
        return {
            "challenge_prompt": "failed to parse",
            "coordinates": [{"box_2d": [1, 1]}]
        }
        
    except Exception as e:
        logger.error(f"Error parsing JSON from response: {str(e)}")
        return {
            "challenge_prompt": "parsing error",
            "coordinates": [{"box_2d": [1, 1]}]
        }