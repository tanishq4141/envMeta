import os
import requests
from typing import Dict, Any, Tuple


class EnvClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:7860")

    def reset(self, task_name: str = "easy_clause_detection") -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/reset", json={"task_name": task_name}
        )
        response.raise_for_status()
        return response.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Send a step action. action should be like:
        {"action_type": "find_risk_clause", "payload": {"clause": "termination"}}
        """
        response = requests.post(
            f"{self.base_url}/step", json={"action": action}
        )
        response.raise_for_status()
        return response.json()

    def state(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
