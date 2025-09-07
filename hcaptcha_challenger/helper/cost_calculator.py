import json
import pathlib
from collections import defaultdict
from statistics import median
from typing import TypedDict, List, Dict, Union, Optional

from pydantic import BaseModel, Field

# For Ollama models, we track usage patterns rather than costs
# These are placeholder values for compatibility
UNIT_1_M = 0.000001
UNIT_1_K = 0.001


class CostItem(TypedDict):
    model: str
    input_price: float
    output_price: float
    unit: float


# Ollama model usage tracking (local models have no cost but we track usage)
model_cost_list: List[CostItem] = [
    # Vision Models
    CostItem(model="llava:latest", input_price=0.0, output_price=0.0, unit=UNIT_1_M),
    CostItem(model="llava:7b", input_price=0.0, output_price=0.0, unit=UNIT_1_M),
    CostItem(model="llava:13b", input_price=0.0, output_price=0.0, unit=UNIT_1_M),
    CostItem(model="llava:34b", input_price=0.0, output_price=0.0, unit=UNIT_1_M),
    CostItem(model="llava-phi3:latest", input_price=0.0, output_price=0.0, unit=UNIT_1_M),
    CostItem(model="llava-llama3:latest", input_price=0.0, output_price=0.0, unit=UNIT_1_M),
    CostItem(model="minicpm-v:latest", input_price=0.0, output_price=0.0, unit=UNIT_1_M),
    CostItem(model="moondream:latest", input_price=0.0, output_price=0.0, unit=UNIT_1_M),
]

model_cost_mapping = {i['model']: i for i in model_cost_list}


class ModelUsageStats(BaseModel):
    """Statistical data model for model usage with Ollama"""

    total_files: int = Field(default=0, description="Total number of model answer files")
    total_challenges: int = Field(default=0, description="Total number of unique challenges")
    total_input_tokens: int = Field(default=0, description="Total input tokens (estimated)")
    total_output_tokens: int = Field(default=0, description="Total output tokens (estimated)")
    total_cost: float = Field(default=0.000, description="Total cost in USD (always 0 for local models)")
    average_cost_per_challenge: float = Field(default=0.000, description="Average cost per challenge (always 0)")
    median_cost_per_challenge: float = Field(default=0.000, description="Median cost per challenge (always 0)")
    model_details: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Usage details by model")
    challenge_costs: List[float] = Field(default_factory=list, description="List of costs for each challenge")

    def save_to_json(self, file_path: Union[str, pathlib.Path]) -> None:
        """Save stats to a JSON file"""
        file_path = pathlib.Path(file_path)

        # Create a serializable dict
        data = self.model_dump()

        # Format all cost values to 3 decimal places
        data["total_cost"] = round(data["total_cost"], 3)
        data["average_cost_per_challenge"] = round(data["average_cost_per_challenge"], 3)
        data["median_cost_per_challenge"] = round(data["median_cost_per_challenge"], 3)

        for model_name, model_data in data["model_details"].items():
            for cost_key in ["input_cost", "output_cost", "total_cost"]:
                if cost_key in model_data:
                    model_data[cost_key] = round(model_data[cost_key], 3)

        # Save to file
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Stats saved to {file_path}")


def calculate_model_cost(
    challenge_path: Union[str, pathlib.Path], detailed: bool = False
) -> Union[float, ModelUsageStats]:
    """
    Calculate the usage statistics for Ollama models (local models have no cost)

    Args:
        challenge_path: Path to challenge data directory
        detailed: Whether to return detailed usage breakdown

    Returns:
        If detailed=False: Returns 0.0 (no cost for local models)
        If detailed=True: Returns ModelUsageStats object with detailed statistics
    """
    challenge_root = pathlib.Path(challenge_path)

    if not challenge_root.exists():
        raise FileNotFoundError(f"Specified path does not exist: {challenge_root}")

    # Initialize stats model
    stats = ModelUsageStats()

    # Track unique challenges by parent directory
    challenge_costs = defaultdict(float)
    challenge_files = defaultdict(list)

    # Process all model answer files
    for item_file in challenge_root.rglob("*_model_answer.json"):
        try:
            stats.total_files += 1

            # Track this file under its parent challenge directory
            challenge_dir = str(item_file.parent)
            challenge_files[challenge_dir].append(item_file)

            # Try to load as JSON and extract model info
            with open(item_file, 'r', encoding='utf-8') as f:
                record = json.load(f)

            # Extract model name from response (Ollama format)
            model_name = "llava:latest"  # Default
            if isinstance(record, dict):
                if "model" in record:
                    model_name = record["model"]
                elif "response" in record and isinstance(record["response"], dict):
                    if "model" in record["response"]:
                        model_name = record["response"]["model"]

            # Estimate token usage (since Ollama doesn't always provide exact counts)
            input_tokens = 1000  # Estimated for image + text
            output_tokens = 200   # Estimated response length
            
            # Try to get actual values if available
            if isinstance(record, dict):
                if "message" in record and isinstance(record["message"], dict):
                    content = record["message"].get("content", "")
                    if content:
                        output_tokens = len(content.split()) * 1.3  # Rough token estimate

            # All costs are 0 for local models
            input_cost = 0.0
            output_cost = 0.0
            item_total_cost = 0.0

            # Update global stats
            stats.total_input_tokens += int(input_tokens)
            stats.total_output_tokens += int(output_tokens)
            stats.total_cost = 0.0  # Always 0 for local models

            # Update challenge-specific cost (always 0)
            challenge_costs[challenge_dir] = 0.0

            # Update model-specific stats if detailed reporting is requested
            if detailed:
                if model_name not in stats.model_details:
                    stats.model_details[model_name] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "input_cost": 0.0,
                        "output_cost": 0.0,
                        "total_cost": 0.0,
                        "usage_count": 0,
                    }

                stats.model_details[model_name]["input_tokens"] += int(input_tokens)
                stats.model_details[model_name]["output_tokens"] += int(output_tokens)
                stats.model_details[model_name]["input_cost"] = 0.0
                stats.model_details[model_name]["output_cost"] = 0.0
                stats.model_details[model_name]["total_cost"] = 0.0
                stats.model_details[model_name]["usage_count"] += 1

        except Exception as e:
            print(f"Error processing file {item_file}: {e}")

    # Calculate challenge statistics
    stats.total_challenges = len(challenge_costs)
    stats.challenge_costs = [0.0] * stats.total_challenges  # All costs are 0

    if stats.total_challenges > 0:
        stats.average_cost_per_challenge = 0.0
        stats.median_cost_per_challenge = 0.0

    if detailed:
        # Add total summary
        stats.model_details["Total"] = {"total_cost": 0.0}
        return stats

    return 0.0  # No cost for local models


def export_stats(
    challenge_path: Union[str, pathlib.Path], output_file: Optional[Union[str, pathlib.Path]] = None
) -> ModelUsageStats:
    """
    Calculate and export detailed statistics for Ollama model usage

    Args:
        challenge_path: Path to challenge data directory
        output_file: Path to save JSON output (optional)

    Returns:
        ModelUsageStats object with complete statistics
    """
    stats = calculate_model_cost(challenge_path, detailed=True)

    if isinstance(stats, float):
        # This shouldn't happen as we specified detailed=True
        raise ValueError("Failed to generate detailed statistics")

    # Ensure all cost values are 0 for local models
    stats.total_cost = 0.0
    stats.average_cost_per_challenge = 0.0
    stats.median_cost_per_challenge = 0.0

    for model_data in stats.model_details.values():
        for cost_key in ["input_cost", "output_cost", "total_cost"]:
            if cost_key in model_data:
                model_data[cost_key] = 0.0

    if output_file:
        stats.save_to_json(output_file)

    return stats