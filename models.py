# models.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class Action(BaseModel):
    action_type: str  # Options: 'terminate', 'downsize', 'keep'
    server_id: str

class Observation(BaseModel):
    active_servers: List[Dict[str, str]]
    feedback: str

class State(BaseModel):
    difficulty: str
    servers: List[Dict[str, str]]
    servers_processed: int
    total_servers: int
    mistakes_made: int