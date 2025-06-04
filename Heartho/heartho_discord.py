import discord
import asyncio
from discord.ext import commands

from chat_interpreter import BaseModel, CustomAlgorithmCore
from fusion import KnowledgeFusionEngine
from ppo import ppo_planner
from supervised_style_classifier import supervised_style_classifier
from dqn_policy_decision import dqn_policy_decision
from few_shot_pattern_matcher import few_shot_pattern_matcher

# 🧠 Core AI Components
model = BaseModel()
core = CustomAlgorithmCore()
fusion = KnowledgeFusionEngine()

# 🧩 Register algorithms
core.register_algorithm("PPOPlanner", ppo_planner)
core.register_algorithm("Supervised", supervised_style_classifier)
core.register_algorithm("DQN", dqn_policy_decision)
core.register_algorithm("FewShot", few_shot_pattern_matcher)

# 🤖 Discord Setup
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")

@bot.command(name="chat")
async def chat(ctx, *, user_input: str):
    try:
        base_response = model.generate(user_input)
        algo_results = core.run(user_input)
        fused = fusion.fuse(algo_results)

        # 📤 Build Response
        response = (
            f"**🧠 Base Model:** {base_response}\n"
            f"**🧮 Algorithm Outputs:** {algo_results}\n"
            f"**🔗 Fused Decision:** {fused}"
        )

        await ctx.send(response)

    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

# 🔑 Run the bot
if __name__ == "__main__":
    import os
    TOKEN = os.getenv("DISCORD_BOT_TOKEN")  # Secure your token via environment variable
    if not TOKEN:
        raise RuntimeError("DISCORD_BOT_TOKEN environment variable not set.")
    bot.run(TOKEN)
