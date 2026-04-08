import json
import os
from typing import List, Dict

# Get the directory of the current script
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(DIR_PATH, "data", "contracts.json")

def load_contracts() -> List[Dict]:
    """Loads the contracts from the JSON file."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def get_contracts_by_difficulty(difficulty: str) -> List[Dict]:
    """Returns a list of contracts matching a given difficulty."""
    contracts = load_contracts()
    return [c for c in contracts if c.get("difficulty") == difficulty]
