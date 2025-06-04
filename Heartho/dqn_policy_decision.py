from stable_baselines3 import DQN
import gymnasium as gym
import numpy as np

vec_env = gym.make("CartPole-v1")
dqn_agent = DQN("MlpPolicy", vec_env, verbose=0)

# dqn_agent.learn(total_timesteps=10000)  # Optional training

def dqn_policy_decision(input_text: str) -> str:
    obs, _ = vec_env.reset()  # ✅ Correct unpacking

    # 🔐 Ensure the observation is a NumPy array
    if isinstance(obs, tuple):
        obs = np.array(obs)
    elif not isinstance(obs, np.ndarray):
        obs = np.array(obs)

    action, _ = dqn_agent.predict(obs, deterministic=True)
    return f"Recommended action: {action}"
