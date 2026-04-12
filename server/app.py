from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from environment import LegalDocumentReviewEnv
from models import LegalAction

app = FastAPI(title="Legal Document Review Environment")

@app.get("/")
def home():
    return {"status": "running"}

env_instance = None

class ResetRequest(BaseModel):
    task_name: Optional[str] = "easy_clause_detection"

@app.post("/reset")
def reset_environment(req: Optional[ResetRequest] = None):
    global env_instance
    try:
        env_instance = LegalDocumentReviewEnv()
        task_name = req.task_name if req and req.task_name else "easy_clause_detection"
        obs = env_instance.reset(task_name)
        return {
            "observation": obs.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_environment(action: LegalAction):
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    try:
        obs, rew, done, info = env_instance.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": rew.score,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    try:
        state = env_instance.state()
        return state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
