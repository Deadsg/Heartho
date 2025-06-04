from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np

# Initialize environment and model
vec_env = gym.make("CartPole-v1", render_mode="rgb_array")  # Use render_mode=None if not visual
ppo_agent = PPO("MlpPolicy", vec_env, verbose=0)

# Optional training step
ppo_agent.learn(total_timesteps=10000)

def ppo_planner(input_text: str) -> str:
    obs, _ = vec_env.reset()

    # Ensure obs is a NumPy array or a dict
    if isinstance(obs, dict):
        pass  # Already a dict, do nothing
    elif not isinstance(obs, np.ndarray):
        obs = np.array(obs)

    action, _ = ppo_agent.predict(obs, deterministic=True)
    return f"Recommended action: {action}"
