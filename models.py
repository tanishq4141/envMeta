from typing import Literal, Optional, Dict, List
from pydantic import BaseModel

class LegalAction(BaseModel):
    action_type: Literal["find_risk_clause", "suggest_edit", "classify_contract", "approve", "reject"]
    payload: dict  # e.g. {"clause": "termination"} or {"contract_type": "NDA"}

class LegalObservation(BaseModel):
    document_text: str
    contract_type: Optional[str] = None
    identified_risks: List[str] = []
    suggestions: Dict[str, str] = {}
    status: str  # "in_progress" | "completed"
    feedback: str = ""  # grader feedback per step

class LegalReward(BaseModel):
    score: float  # 0.0 to 1.0

class LegalState(BaseModel):
    episode_id: str
    step_count: int
    task: str
    done: bool
