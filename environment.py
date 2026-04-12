import uuid
import random
from typing import Dict, Any, Tuple, Optional

from openenv.core.env_server import Environment
from models import LegalAction, LegalObservation, LegalState
from contracts import get_contracts_by_difficulty
from graders import grade_easy, grade_medium, grade_hard

# Map task names to difficulty names defined in contracts.json
TASK_TO_DIFFICULTY = {
    "easy_clause_detection": "easy",
    "medium_risk_analysis": "medium",
    "hard_full_review": "hard"
}

DEFAULT_TASK = "easy_clause_detection"


class LegalDocumentReviewEnv(Environment):
    def __init__(self):
        super().__init__()
        self._state = LegalState()
        self._internal = {
            "current_contract": None,
            "identified_risks": [],
            "suggestions": {},
            "classification": None,
            "decision": None,
        }

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LegalObservation:
        task_name = kwargs.get("task_name", DEFAULT_TASK)
        difficulty = TASK_TO_DIFFICULTY.get(task_name)
        if not difficulty:
            raise ValueError(f"Unknown task: {task_name}")

        if seed is not None:
            random.seed(seed)

        contracts = get_contracts_by_difficulty(difficulty)
        if not contracts:
            raise RuntimeError(f"No contracts found for difficulty {difficulty}")

        contract = random.choice(contracts)

        eid = episode_id or str(uuid.uuid4())
        self._state = LegalState(
            episode_id=eid,
            step_count=0,
            task=task_name,
            done=False,
            cumulative_reward=0.0,
        )
        self._internal = {
            "current_contract": contract,
            "identified_risks": [],
            "suggestions": {},
            "classification": None,
            "decision": None,
        }

        return LegalObservation(
            document_text=contract["text"],
            status="in_progress",
            feedback="Environment initialized.",
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: LegalAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> LegalObservation:
        # Auto-reset if step is called on an uninitialized env
        # (the OpenEnv HTTP server creates a new env per request)
        if self._internal["current_contract"] is None:
            self.reset()

        self._state.step_count += 1
        reward_delta = 0.0
        feedback = ""

        action_type = action.action_type
        payload = action.payload
        contract = self._internal["current_contract"]

        if action_type == "find_risk_clause":
            clause = payload.get("clause")
            if clause in contract.get("ground_truth_risks", []):
                if clause not in self._internal["identified_risks"]:
                    self._internal["identified_risks"].append(clause)
                    reward_delta = 0.3
                    feedback = f"Correctly identified risk: {clause}"
                else:
                    reward_delta = -0.2
                    feedback = "Risk already identified."
            else:
                reward_delta = -0.1
                feedback = f"Incorrect risk identified: {clause}"

        elif action_type == "suggest_edit":
            clause = payload.get("clause")
            suggestion = payload.get("suggestion", "")
            if clause in contract.get("ground_truth_suggestions", {}):
                self._internal["suggestions"][clause] = suggestion
                reward_delta = 0.3
                feedback = f"Recorded suggestion for {clause}."
            else:
                reward_delta = -0.1
                feedback = "Provided suggestion for an invalid or non-existent risk clause."

        elif action_type == "classify_contract":
            contract_type = payload.get("contract_type")
            self._internal["classification"] = contract_type
            if contract_type == contract.get("ground_truth_classification"):
                reward_delta = 0.2
                feedback = "Correct classification."
            else:
                reward_delta = -0.1
                feedback = "Incorrect classification."

        elif action_type == "approve":
            self._internal["decision"] = "approve"
            if contract.get("ground_truth_decision") == "approve":
                reward_delta = 0.2
                feedback = "Correct decision."
            else:
                reward_delta = -0.1
                feedback = "Incorrect decision."

        elif action_type == "reject":
            self._internal["decision"] = "reject"
            if contract.get("ground_truth_decision") == "reject":
                reward_delta = 0.2
                feedback = "Correct decision."
            else:
                reward_delta = -0.1
                feedback = "Incorrect decision."
        else:
            reward_delta = -0.2
            feedback = "Unknown action type."

        # Update cumulative reward and clamp
        self._state.cumulative_reward += reward_delta
        self._state.cumulative_reward = max(0.0, min(1.0, self._state.cumulative_reward))

        # Check termination conditions
        difficulty = TASK_TO_DIFFICULTY[self._state.task]
        env_state_dict = self._build_grader_dict()

        if difficulty == "easy":
            if grade_easy(env_state_dict) >= 1.0:
                self._state.done = True
        elif difficulty == "medium":
            if (len(self._internal["identified_risks"]) >= len(contract.get("ground_truth_risks", []))
               and self._internal["classification"] is not None):
                self._state.done = True
        elif difficulty == "hard":
            if self._internal["decision"] in ["approve", "reject"]:
                self._state.done = True

        # Approve/reject always ends episode
        if action_type in ["approve", "reject"]:
            self._state.done = True

        # Timeout at 15 steps
        if self._state.step_count >= 15:
            self._state.done = True

        # Final score via graders when done
        if self._state.done:
            env_state_dict = self._build_grader_dict()
            if difficulty == "easy":
                final_score = grade_easy(env_state_dict)
            elif difficulty == "medium":
                final_score = grade_medium(env_state_dict)
            else:
                final_score = grade_hard(env_state_dict)
            self._state.cumulative_reward = max(0.0, min(1.0, final_score))

        return LegalObservation(
            document_text=contract["text"],
            contract_type=self._internal["classification"],
            identified_risks=list(self._internal["identified_risks"]),
            suggestions=dict(self._internal["suggestions"]),
            status="completed" if self._state.done else "in_progress",
            feedback=feedback,
            done=self._state.done,
            reward=self._state.cumulative_reward,
        )

    @property
    def state(self) -> LegalState:
        return self._state

    def _build_grader_dict(self) -> dict:
        """Build the dict that graders expect."""
        return {
            "current_contract": self._internal["current_contract"],
            "identified_risks": self._internal["identified_risks"],
            "suggestions": self._internal["suggestions"],
            "classification": self._internal["classification"],
            "decision": self._internal["decision"],
        }
