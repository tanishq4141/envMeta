from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from ..environment import LegalDocumentReviewEnv
from ..models import LegalAction

app = FastAPI(title="Legal Document Review Environment")

@app.get("/")
def home():
    return {"status": "running"}

# Instantiate a single global environment
# In a real setup, we might map episode_ids to environment instances for concurrency
env_instance = LegalDocumentReviewEnv()

class ResetRequest(BaseModel):
    task_name: str

@app.post("/reset")
def reset_environment(req: ResetRequest):
    try:
        obs = env_instance.reset(req.task_name)
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
    try:
        state = env_instance.state()
        return state.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
