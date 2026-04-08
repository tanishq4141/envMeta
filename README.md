# Legal Document Review Environment

An **OpenEnv-compliant** environment that simulates legal contract review. An AI agent is tasked with identifying risky clauses, suggesting corrections, classifying contracts, and making final approval/rejection decisions.

## Features
- **Deterministic Environment**: No LLM used inside the environment logic. Fully reproducible. 
- **Sample Contract Corpus**: Includes 9 pre-built sample contracts (Employment, NDA, Vendor/Service) with designated risk clauses spanning easy, medium, and hard difficulties.
- **Strict Scoring**: Provides continuous rewards bounded between `[0, 1]`, encouraging precision and penalizing repeated mistakes.
- **OpenEnv Ready**: Built as a standard HTTP FastAPI application that easily connects to an OpenEnv runner.

## Quickstart

### Running the Environment Server

You can run the environment natively or using Docker.

**Native**:
```bash
pip install -r pyproject.toml # Or standard pip install .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

**Docker**:
```bash
docker build -t legal-doc-review-env -f server/Dockerfile .
docker run -p 8000:8000 legal-doc-review-env
```

### Running Inference
Start the inference loop (requires OpenAI API set, or set `OPENAI_API_BASE` for local models/vLLM):

```bash
export OPENAI_API_KEY="your-api-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

## API / Action Space

The action is submitted as JSON matching the `LegalAction` type:
- `find_risk_clause`: `{"action_type": "find_risk_clause", "payload": {"clause": "termination_without_notice"}}`
- `suggest_edit`: `{"action_type": "suggest_edit", "payload": {"clause": "liability_cap_low", "suggestion": "Increase liability..."}}`
- `classify_contract`: `{"action_type": "classify_contract", "payload": {"contract_type": "NDA"}}`
- `approve`: `{"action_type": "approve", "payload": {}}`
- `reject`: `{"action_type": "reject", "payload": {}}`

## Observations
The state return consists of `document_text`, current `status`, `identified_risks`, `suggestions`, and `feedback` from the last step.
