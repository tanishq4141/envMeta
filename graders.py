def grade_easy(env_state) -> float:
    """
    Score = 1.0 if any correct risk found, else 0.0.
    """
    gt_risks = env_state["current_contract"].get("ground_truth_risks", [])
    identified_risks = env_state["identified_risks"]
    
    for risk in identified_risks:
        if risk in gt_risks:
            return 0.99
    return 0.01

def grade_medium(env_state) -> float:
    """
    Score = 0.5 * risk_detection_ratio + 0.5 * classification_correct.
    """
    gt_risks = env_state["current_contract"].get("ground_truth_risks", [])
    identified_risks = env_state["identified_risks"]
    
    if len(gt_risks) == 0:
        risk_ratio = 1.0
    else:
        correct_risks = [r for r in identified_risks if r in gt_risks]
        risk_ratio = len(correct_risks) / len(gt_risks)
        
    gt_classification = env_state["current_contract"].get("ground_truth_classification")
    classification_correct = 1.0 if env_state.get("classification") == gt_classification else 0.0
    
    score = 0.5 * risk_ratio + 0.5 * classification_correct
    return max(0.01, min(0.99, score))

def grade_hard(env_state) -> float:
    """
    Score = 0.4 * risk_detection + 0.3 * suggestion_quality (simplified boolean match) + 0.3 * decision_correct.
    """
    gt_risks = env_state["current_contract"].get("ground_truth_risks", [])
    identified_risks = env_state["identified_risks"]
    
    if len(gt_risks) == 0:
        risk_ratio = 1.0
    else:
        correct_risks = [r for r in identified_risks if r in gt_risks]
        risk_ratio = len(correct_risks) / len(gt_risks)
        
    gt_suggestions = env_state["current_contract"].get("ground_truth_suggestions", {})
    suggestions = env_state["suggestions"]
    
    if len(gt_suggestions) == 0:
        sugg_ratio = 1.0
    else:
        # In a real environment we might use an LLM for semantic similarity.
        # Since this env is fully deterministic, we award points if the agent provided *any* suggestion
        # for a valid risk clause that requires one.
        correct_suggs = 0
        for clause, sugg in suggestions.items():
            if clause in gt_suggestions and isinstance(sugg, str) and len(sugg.strip()) > 0:
                correct_suggs += 1
        sugg_ratio = correct_suggs / len(gt_suggestions)
        
    gt_decision = env_state["current_contract"].get("ground_truth_decision")
    decision_correct = 1.0 if env_state.get("decision") == gt_decision else 0.0
    
    score = 0.4 * risk_ratio + 0.3 * sugg_ratio + 0.3 * decision_correct
    return max(0.01, min(0.99, score))
