from typing import Annotated, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel

from models import Action, Observation, State
from server.cloud_optimizer_environment import CloudOptimizerEnvironment


class ResetRequest(BaseModel):
    difficulty: Literal["easy", "medium", "hard"] = "easy"


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    error: Optional[str] = None
    state: State


app = FastAPI(
    title="Cloud Cost Optimizer",
    description="Simple environment API for the cloud optimization hackathon.",
    version="1.0.0",
)
env = CloudOptimizerEnvironment()


@app.get("/")
def root():
    return {
        "message": "Cloud Cost Optimizer API is running.",
        "endpoints": ["/health", "/reset", "/step", "/state"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset_environment(
    request: Annotated[Optional[ResetRequest], Body()] = None,
    difficulty: Annotated[Literal["easy", "medium", "hard"], Query()] = "easy",
):
    selected_difficulty = request.difficulty if request is not None else difficulty
    return env.reset(difficulty=selected_difficulty)


@app.get("/state", response_model=State)
def get_state():
    state = env.get_state()
    if state is None:
        raise HTTPException(status_code=400, detail="Environment has not been reset yet.")
    return state


@app.post("/step", response_model=StepResponse)
def take_step(action: Action):
    state = env.get_state()
    if state is None:
        raise HTTPException(status_code=400, detail="Reset the environment before taking a step.")

    observation, reward, done, error = env.step(action)
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        error=error,
        state=env.get_state(),
    )
