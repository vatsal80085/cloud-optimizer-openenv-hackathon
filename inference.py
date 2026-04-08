import os
import json
from openai import OpenAI
# If your environment file is named differently, update the import below:
from server.cloud_optimizer_environment import CloudOptimizerEnvironment
from models import Action

# 1. Read required Hackathon environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1") 
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required! Set it in your terminal.")

# 2. Initialize HF Router Client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_ai_action(observation_json: str) -> Action:
    """Asks the AI what to do based on the server dashboard."""
    prompt = f"""
    You are a Cloud Cost Optimization AI. 
    Look at this server dashboard: {observation_json}. 
    Decide what to do with ONE server. 
    Valid action_types are: 'terminate' (for 0% CPU/unused), 'downsize' (for low CPU), 'keep' (for high CPU/production).
    Respond strictly in JSON format: {{"action_type": "...", "server_id": "..."}}
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={ "type": "json_object" }
    )
    
    content = json.loads(response.choices[0].message.content)
    return Action(**content)

def run():
    env = CloudOptimizerEnvironment()
    tasks = ["easy", "medium", "hard"]
    env_name = "cloud-cost-optimizer"
    
    for task in tasks:
        obs = env.reset(difficulty=task)
        
        # EXACT Hackathon Format Requirement: [START]
        print(f"[START] task={task} env={env_name} model={MODEL_NAME}")
        
        done = False
        step_count = 0
        rewards = []
        
        while not done and step_count < 10:
            step_count += 1
            try:
                # 1. AI decides what to do
                action = get_ai_action(obs.model_dump_json())
                
                # 2. Environment reacts
                obs, reward, done, error = env.step(action)
                rewards.append(reward)
                
                error_str = error if error else "null"
                action_str = f"{action.action_type}('{action.server_id}')"
                
                # EXACT Hackathon Format Requirement: [STEP]
                print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_str}")
                
            except Exception as e:
                done = True
                print(f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e).replace(' ', '_')}")
                rewards.append(0.00)

        # EXACT Hackathon Format Requirement: [END]
        success = done and env.state.total_servers == env.state.servers_processed and env.state.mistakes_made == 0
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step_count} rewards={rewards_str}")

if __name__ == "__main__":
    run()