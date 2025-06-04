from chat_interpreter import BaseModel
from chat_interpreter import CustomAlgorithmCore
from fusion import KnowledgeFusionEngine
from ppo import ppo_planner

from supervised_style_classifier import supervised_style_classifier
from dqn_policy_decision import dqn_policy_decision
from few_shot_pattern_matcher import few_shot_pattern_matcher

def main():
    model = BaseModel()
    core = CustomAlgorithmCore()
    fusion = KnowledgeFusionEngine()

    # Register all algorithms
    core.register_algorithm("PPOPlanner", ppo_planner)
    core.register_algorithm("Supervised", supervised_style_classifier)
    core.register_algorithm("DQN", dqn_policy_decision)
    core.register_algorithm("FewShot", few_shot_pattern_matcher)

    while True:
        try:
            user_input = input("🧠 You: ")
            if user_input.lower() in {"exit", "quit"}:
                break

            base_response = model.generate(user_input)
            algo_results = core.run(user_input)
            fused = fusion.fuse(algo_results)

            print("\n🔹 Base Model:", base_response)
            print("🔹 Algorithm Outputs:", algo_results)
            print("🔹 Fused Decision:", fused, "\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
