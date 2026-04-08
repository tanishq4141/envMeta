import uuid
import random
from typing import Dict, Any, Tuple
from .models import LegalAction, LegalObservation, LegalState, LegalReward
from .contracts import get_contracts_by_difficulty
from .graders import grade_easy, grade_medium, grade_hard

# Map task names to difficulty names defined in contracts.json
TASK_TO_DIFFICULTY = {
    "easy_clause_detection": "easy",
    "medium_risk_analysis": "medium",
    "hard_full_review": "hard"
}

class LegalDocumentReviewEnv:
    def __init__(self):
        self.state_data = {
            "episode_id": None,
            "step_count": 0,
            "task": None,
            "done": False,
            "current_contract": None,
            "identified_risks": [],
            "suggestions": {},
            "classification": None,
            "decision": None,
            "cumulative_reward": 0.0
        }

    def reset(self, task_name: str) -> LegalObservation:
        difficulty = TASK_TO_DIFFICULTY.get(task_name)
        if not difficulty:
            raise ValueError(f"Unknown task: {task_name}")

        contracts = get_contracts_by_difficulty(difficulty)
        if not contracts:
            raise RuntimeError(f"No contracts found for difficulty {difficulty}")
            
        contract = random.choice(contracts)
        
        self.state_data = {
            "episode_id": str(uuid.uuid4()),
            "step_count": 0,
            "task": task_name,
            "done": False,
            "current_contract": contract,
            "identified_risks": [],
            "suggestions": {},
            "classification": None,
            "decision": None,
            "cumulative_reward": 0.0
        }
        
        return LegalObservation(
            document_text=contract["text"],
            status="in_progress",
            feedback="Environment initialized."
        )

    def step(self, action: LegalAction) -> Tuple[LegalObservation, LegalReward, bool, Dict[str, Any]]:
        self.state_data["step_count"] += 1
        reward_delta = 0.0
        feedback = ""
        
        action_type = action.action_type
        payload = action.payload
        
        contract = self.state_data["current_contract"]
        
        if action_type == "find_risk_clause":
            clause = payload.get("clause")
            if clause in contract.get("ground_truth_risks", []):
                if clause not in self.state_data["identified_risks"]:
                    self.state_data["identified_risks"].append(clause)
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
                self.state_data["suggestions"][clause] = suggestion
                reward_delta = 0.3
                feedback = f"Recorded suggestion for {clause}."
            else:
                reward_delta = -0.1
                feedback = "Provided suggestion for an invalid or non-existent risk clause."
                
        elif action_type == "classify_contract":
            contract_type = payload.get("contract_type")
            self.state_data["classification"] = contract_type
            if contract_type == contract.get("ground_truth_classification"):
                reward_delta = 0.2
                feedback = "Correct classification."
            else:
                reward_delta = -0.1
                feedback = "Incorrect classification."
                
        elif action_type == "approve":
            self.state_data["decision"] = "approve"
            if contract.get("ground_truth_decision") == "approve":
                reward_delta = 0.2
                feedback = "Correct decision."
            else:
                reward_delta = -0.1
                feedback = "Incorrect decision."
                
        elif action_type == "reject":
            self.state_data["decision"] = "reject"
            if contract.get("ground_truth_decision") == "reject":
                reward_delta = 0.2
                feedback = "Correct decision."
            else:
                reward_delta = -0.1
                feedback = "Incorrect decision."
        else:
            reward_delta = -0.2
            feedback = "Unknown action type."

        # Add step reward to cumulative and clamp bounds
        self.state_data["cumulative_reward"] += reward_delta
        self.state_data["cumulative_reward"] = max(0.0, min(1.0, self.state_data["cumulative_reward"]))
        
        # Check termination conditions
        difficulty = TASK_TO_DIFFICULTY[self.state_data["task"]]
        if difficulty == "easy":
            if grade_easy(self.state_data) >= 1.0:
                self.state_data["done"] = True
        elif difficulty == "medium":
            if len(self.state_data["identified_risks"]) >= len(contract.get("ground_truth_risks", [])) \
               and self.state_data["classification"] is not None:
                self.state_data["done"] = True
        elif difficulty == "hard":
            if self.state_data["decision"] in ["approve", "reject"]:
                self.state_data["done"] = True
                
        # Also forcefully end if agent takes an approve/reject action (as per standard contract reviews concluding)
        if action_type in ["approve", "reject"]:
            self.state_data["done"] = True
            
        # Optional timeout logic (e.g. max 15 steps)
        if self.state_data["step_count"] >= 15:
            self.state_data["done"] = True

        # Calculate final reward if done using graders (replacing cumulative if we want strict task grading)
        # The prompt says cumulative reward is clamped [0,1]. We can just replace the final reward with the grader score for accurate scoring
        if self.state_data["done"]:
            if difficulty == "easy":
                final_score = grade_easy(self.state_data)
            elif difficulty == "medium":
                final_score = grade_medium(self.state_data)
            else:
                final_score = grade_hard(self.state_data)
            self.state_data["cumulative_reward"] = max(0.0, min(1.0, final_score))
            self.state_data["status"] = "completed"
            
        obs = LegalObservation(
            document_text=contract["text"],
            contract_type=self.state_data["classification"],
            identified_risks=list(self.state_data["identified_risks"]),
            suggestions=dict(self.state_data["suggestions"]),
            status="completed" if self.state_data["done"] else "in_progress",
            feedback=feedback
        )
        
        rew = LegalReward(score=self.state_data["cumulative_reward"])
        info = {
            "current_step": self.state_data["step_count"],
            "step_reward": reward_delta
        }
        
        return obs, rew, self.state_data["done"], info

    def state(self) -> LegalState:
        return LegalState(
            episode_id=self.state_data.get("episode_id", ""),
            step_count=self.state_data.get("step_count", 0),
            task=self.state_data.get("task", ""),
            done=self.state_data.get("done", False)
        )
