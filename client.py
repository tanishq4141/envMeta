import os
import requests
from typing import Dict, Any

class EnvClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        
    def reset(self, task_name: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/reset", json={"task_name": task_name})
        response.raise_for_status()
        return response.json()
        
    def step(self, action: Dict[str, Any]) -> Tuple[Dict, Dict, bool, Dict]:
        # action is a dictionary matching LegalAction model
        response = requests.post(f"{self.base_url}/step", json=action)
        response.raise_for_status()
        data = response.json()
        return data["observation"], data["reward"], data["done"], data["info"]
        
    def state(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()
