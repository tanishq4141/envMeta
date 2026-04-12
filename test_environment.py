import pytest
from models import LegalAction
from environment import LegalDocumentReviewEnv

def test_easy_task():
    env = LegalDocumentReviewEnv()
    obs = env.reset("easy_clause_detection")
    
    assert obs.status == "in_progress"
    
    # We find out which contract was loaded to do the right action
    contract_class = env.state_data["current_contract"].get("ground_truth_classification")
    gt_risks = env.state_data["current_contract"].get("ground_truth_risks", [])
    
    action = LegalAction(action_type="find_risk_clause", payload={"clause": gt_risks[0]})
    obs, rew, done, info = env.step(action)
    
    assert done is True
    assert rew.score == 1.0
    assert obs.status == "completed"

def test_reject_action_ends_env():
    env = LegalDocumentReviewEnv()
    obs = env.reset("hard_full_review")
    
    action = LegalAction(action_type="reject", payload={})
    obs, rew, done, info = env.step(action)
    
    assert done is True
    # The score might not be 1.0, but the run should end
    assert 0.0 <= rew.score <= 1.0
