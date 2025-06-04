from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
ppo_agent = PPO("MlpPolicy", env, verbose=0)
ppo_agent.learn(total_timesteps=10000)  # Uncomment if training from scratch

def ppo_planner(input_text: str) -> str:
    obs = env.reset()
    action, _ = ppo_agent.predict(obs, deterministic=True)
    return f"policy_action_{action}"
