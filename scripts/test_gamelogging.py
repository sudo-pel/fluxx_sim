from pathlib import Path

import torch

from src.agents.HeuristicAgentMKI import HeuristicAgentMKI
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.training.TrainingEnums import GameLogConfig
from src.env.AgentBattler import AgentBattler
from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists

PROJECT_ROOT = Path(__file__).resolve().parent.parent

two_player_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True, logger=None)
env = FluxxEnv(two_player_fluxx, 2, render_mode="human")
agent_battler = AgentBattler(env)

actor = HeuristicAgentMKI(env.game.game_config, 0)

actor2 = HeuristicAgentMKII(env.game.game_config, 1)

logconfig = GameLogConfig(
    "test_ppo_vs_heuristic",
    ["player_0", "player_1"]
)

agent_battler.run_games([actor, actor2], 3, 10000, log_games=True, log_config=logconfig)