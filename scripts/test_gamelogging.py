import torch

from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.training.TrainingEnums import GameLogConfig
from src.env.AgentBattler import AgentBattler
from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists

two_player_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True, logger=None)
env = FluxxEnv(two_player_fluxx, 2, render_mode="human")
agent_battler = AgentBattler(env)

actor = PPOAgent(env.game.game_config, 0)  # same architecture
actor.policy_network.load_state_dict(torch.load("../from_remote/ppo_2026-04-23_12-33-19_final.pt"))
actor.policy_network.eval()

actor2 = HeuristicAgentMKII(env.game.game_config, 1)

logconfig = GameLogConfig(
    "test_ppo_vs_heuristic",
    ["player_0", "player_1"]
)

agent_battler.run_games([actor, actor2], 3, 10000, log_games=True, log_config=logconfig)