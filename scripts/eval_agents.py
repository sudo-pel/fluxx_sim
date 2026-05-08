from pathlib import Path

import torch

from src.agents.DQNAgent import DQNAgent
from src.agents.HeuristicAgentMKI import HeuristicAgentMKI
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.agents.PPOAgentGeneralized import PPOAgentGeneralized
from src.agents.RandomAgent import RandomAgent
from src.agents.card_embeddings import generate_embedding_table
from src.training.TrainingEnums import GameLogConfig
from src.env.AgentBattler import AgentBattler
from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Card lists
card_lists = [
    card_lists.base_deck,
    card_lists.simple_fluxx_deck,
    card_lists.expanded_deck
]

# Prepare the agents.
agents = {
    "ppo": PPOAgent(env.game.game_config, 0),
    "ppo_general": PPOAgentGeneralized(env.game.game_config, 0),
    "dqn": DQNAgent(env.game.game_config, 0),
    "dqn_general": None,
    "ppo_general_with_reward_shaping": None,
    "random": RandomAgent(env.game.game_config, 0),
    "heuristic_agent_mki": HeuristicAgentMKI(env.game.game_config, 0),
    "heuristic_agent_mkii": HeuristicAgentMKII(env.game.game_config, 0),
}

# Load the trained state dicts.
agents["ppo"].policy_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/ppo_2026-04-28_17-11-02/final/final_model_50004751.pt"))
agents["ppo"].policy_network.eval()

# strict=False because card embeds was a part of state_dict when this code was run
agents["ppo_general"].policy_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/ppo_general_2026-05-02_09-11-14/final/final_model_50006336.pt"), strict=False)
agents["ppo_general"].policy_network.eval()

agents["dqn"].q_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/dqn_2026-04-29_08-06-30/final/final_model_50000050.pt"))
agents["dqn"].q_network.eval()

# TODO: dqn_general and ppo_general_with_reward_shaping

# (Temporary) filter out uninitialized agents
agent_names = [a for a in agents.keys()]
for agent_name in agent_names:
    if agents[agent_name] is None:
        del agents[agent_name]

results: dict[tuple[str, str], dict[str, float]] = {}

for card_list in card_lists:
    print(f"PLAYING WITH: CARD LIST: {card_list}")
    generate_embedding_table(card_list)
    two_player_fluxx = Game(2, card_list, disable_game_messages=True, logger=None)
    env = FluxxEnv(two_player_fluxx, 2, render_mode="human")
    agent_battler = AgentBattler(env)

    for agent_name, agent in agents.items():
        for other_agent_name, other_agent in agents.items():
            if agent_name == other_agent_name: continue
            print(f"RUNNING {agent_name} vs {other_agent_name}")
            agent.player_number = 0
            other_agent.player_number = 1
            results[(agent_name, other_agent_name)] = agent_battler.run_games([agent, other_agent], 10000, 10000, log_games=False)
            print(f"RESULTS: {results[(agent_name, other_agent_name)]}")
