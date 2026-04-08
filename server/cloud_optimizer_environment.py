import random
from models import Action, Observation, State

class CloudOptimizerEnvironment:
    def __init__(self):
        # The 3 Difficulty Levels required by the Hackathon
        self.scenarios = {
            "easy": [
                {"id": "srv-01", "role": "abandoned_test_server", "cpu_usage": "0%", "correct_action": "terminate"}
            ],
            "medium": [
                {"id": "srv-01", "role": "production_database", "cpu_usage": "85%", "correct_action": "keep"},
                {"id": "srv-02", "role": "old_backup", "cpu_usage": "0%", "correct_action": "terminate"},
                {"id": "srv-03", "role": "oversized_cache", "cpu_usage": "5%", "correct_action": "downsize"}
            ],
            "hard": [
                {"id": "srv-01", "role": "payment_gateway", "cpu_usage": "90%", "correct_action": "keep"},
                {"id": "srv-02", "role": "dev_environment", "cpu_usage": "1%", "correct_action": "terminate"},
                {"id": "srv-03", "role": "analytics_worker", "cpu_usage": "10%", "correct_action": "downsize"},
                {"id": "srv-04", "role": "defunct_api", "cpu_usage": "0%", "correct_action": "terminate"},
                {"id": "srv-05", "role": "main_website", "cpu_usage": "75%", "correct_action": "keep"}
            ]
        }
        self.state = None

    def reset(self, difficulty="easy") -> Observation:
        # Load the right scenario
        servers = self.scenarios.get(difficulty, self.scenarios["easy"]).copy()
        
        self.state = State(
            difficulty=difficulty,
            servers=servers,
            servers_processed=0,
            total_servers=len(servers),
            mistakes_made=0
        )
        return Observation(active_servers=self.state.servers, feedback="Dashboard loaded. Please process servers.")

    def get_state(self) -> State:
        return self.state

    def step(self, action: Action):
        # 1. Find the server the AI wants to target
        target_server = next((s for s in self.state.servers if s["id"] == action.server_id), None)
        
        reward = 0.00
        error = None
        feedback = ""
        
        # 2. Programmatic Grader (0.0 to 1.0)
        if target_server:
            if action.action_type == target_server["correct_action"]:
                reward = 1.00  # Perfect action!
                feedback = f"Success: Action '{action.action_type}' applied correctly to {action.server_id}."
                self.state.servers_processed += 1
                self.state.servers.remove(target_server) # Remove processed server from dashboard
            else:
                reward = 0.00  # Wrong action
                self.state.mistakes_made += 1
                feedback = f"CRITICAL WARNING: Incorrect action '{action.action_type}' on {action.server_id}. Cost penalty applied."
                
                # If they terminate a critical server, end the game immediately (harsh penalty)
                if target_server["correct_action"] == "keep" and action.action_type == "terminate":
                    error = "FATAL: Terminated critical production server. System crashed."
        else:
            error = f"Server ID {action.server_id} not found."

        # 3. Check if all servers are processed or if fatal error occurred
        done = len(self.state.servers) == 0 or error is not None

        obs = Observation(active_servers=self.state.servers, feedback=feedback)
        return obs, reward, done, error