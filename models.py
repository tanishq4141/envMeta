from typing import Literal, Optional, Dict, List
from openenv.core.env_server import Action, Observation, State


class LegalAction(Action):
    """Action for the legal document review environment."""
    action_type: Literal["find_risk_clause", "suggest_edit", "classify_contract", "approve", "reject"]
    payload: dict  # e.g. {"clause": "termination"} or {"contract_type": "NDA"}


class LegalObservation(Observation):
    """Observation returned by the legal document review environment.
    Inherits done: bool and reward: float|None from Observation base class.
    """
    document_text: str = ""
    contract_type: Optional[str] = None
    identified_risks: List[str] = []
    suggestions: Dict[str, str] = {}
    status: str = "in_progress"  # "in_progress" | "completed"
    feedback: str = ""  # grader feedback per step


class LegalState(State):
    """State for the legal document review environment.
    Inherits episode_id: str and step_count: int from State base class.
    """
    task: str = ""
    done: bool = False
    cumulative_reward: float = 0.0
