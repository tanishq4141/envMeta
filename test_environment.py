"""Tests for the legal document review environment using the OpenEnv protocol."""
from models import LegalAction
from environment import LegalDocumentReviewEnv


def test_reset_default_task():
    """Test reset with default task (easy_clause_detection)."""
    env = LegalDocumentReviewEnv()
    obs = env.reset()
    assert obs.done is False
    assert obs.status == "in_progress"
    assert obs.document_text != ""
    assert obs.feedback == "Environment initialized."


def test_reset_with_task_name():
    """Test reset with explicit task name via kwargs."""
    env = LegalDocumentReviewEnv()
    obs = env.reset(task_name="easy_clause_detection")
    assert obs.done is False
    assert obs.status == "in_progress"
    assert obs.document_text != ""


def test_reset_with_seed():
    """Test reset with seed for reproducibility."""
    env = LegalDocumentReviewEnv()
    obs1 = env.reset(seed=42, task_name="easy_clause_detection")
    env2 = LegalDocumentReviewEnv()
    obs2 = env2.reset(seed=42, task_name="easy_clause_detection")
    assert obs1.document_text == obs2.document_text


def test_easy_task_correct_risk():
    """Test easy task: finding a correct risk clause ends the episode."""
    env = LegalDocumentReviewEnv()
    obs = env.reset(seed=42, task_name="easy_clause_detection")
    assert obs.status == "in_progress"

    # Get ground truth from internal state
    contract = env._internal["current_contract"]
    gt_risks = contract.get("ground_truth_risks", [])
    assert len(gt_risks) > 0, "Contract should have ground truth risks"

    action = LegalAction(action_type="find_risk_clause", payload={"clause": gt_risks[0]})
    obs = env.step(action)

    assert obs.done is True
    assert obs.reward is not None
    assert obs.reward >= 0.0
    assert obs.status == "completed"


def test_reject_action_ends_episode():
    """Test that reject action ends the episode."""
    env = LegalDocumentReviewEnv()
    env.reset(task_name="hard_full_review")

    action = LegalAction(action_type="reject", payload={})
    obs = env.step(action)

    assert obs.done is True
    assert obs.reward is not None
    assert 0.0 <= obs.reward <= 1.0


def test_approve_action_ends_episode():
    """Test that approve action ends the episode."""
    env = LegalDocumentReviewEnv()
    env.reset(task_name="hard_full_review")

    action = LegalAction(action_type="approve", payload={})
    obs = env.step(action)

    assert obs.done is True
    assert obs.reward is not None
    assert 0.0 <= obs.reward <= 1.0


def test_state_property():
    """Test that state property works correctly."""
    env = LegalDocumentReviewEnv()
    obs = env.reset(task_name="easy_clause_detection")

    state = env.state
    assert state.task == "easy_clause_detection"
    assert state.step_count == 0
    assert state.done is False
    assert state.episode_id is not None


def test_step_increments_counter():
    """Test that step count increments properly."""
    env = LegalDocumentReviewEnv()
    env.reset(task_name="easy_clause_detection")

    action = LegalAction(action_type="find_risk_clause", payload={"clause": "nonexistent"})
    env.step(action)

    assert env.state.step_count == 1


if __name__ == "__main__":
    test_reset_default_task()
    test_reset_with_task_name()
    test_reset_with_seed()
    test_easy_task_correct_risk()
    test_reject_action_ends_episode()
    test_approve_action_ends_episode()
    test_state_property()
    test_step_increments_counter()
    print("All tests passed!")
