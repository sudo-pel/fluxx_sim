import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import torch

from src.agents.Agent import Agent
from src.agents.DQNAgent import DQNAgent
from src.agents.DQNAgentGeneralized import DQNAgentGeneralized
from src.agents.HeuristicAgentMKI import HeuristicAgentMKI
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.agents.PPOAgentGeneralized import PPOAgentGeneralized
from src.agents.RandomAgent import RandomAgent
from src.agents.card_embeddings import generate_embedding_table, get_embedding_table
from src.env.AgentBattlerParallel import AgentBattler
from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists as card_list_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CARD_LISTS = {
    "simple_fluxx_deck": card_list_module.simple_fluxx_deck,
    "base_deck": card_list_module.base_deck,
    "expanded_deck": card_list_module.expanded_deck,
}

AGENT_NAMES = [
    "ppo",
    "ppo_general",
    "ppo_general_with_reward_shaping",
    "dqn",
    "random",
    "heuristic_agent_mki",
    "heuristic_agent_mkii",
]

AGENT_REGISTRY = {
    "ppo": {
        "class": PPOAgent,
        "network_attr": "policy_network",
        "state_dict": "final_experiments/ppo_2026-04-28_17-11-02/final/final_model_50004751.pt",
    },
    "ppo_general": {
        "class": PPOAgentGeneralized,
        "network_attr": "policy_network",
        "state_dict": "final_experiments/ppo_general_2026-05-02_09-11-14/final/final_model_50006336.pt",
        "strict": False,
    },
    "dqn": {
        "class": DQNAgent,
        "network_attr": "q_network",
        "state_dict": "final_experiments/dqn_2026-04-29_08-06-30/final/final_model_50000050.pt",
    },
    "dqn_general": {
        "class": DQNAgentGeneralized,
        "network_attr": "q_network",
        "state_dict": "final_experiments/dqn_general_2026-05-02_21-23-41/models/model_20000002.pt",
    },
    "ppo_general_with_reward_shaping": {
        "class": PPOAgentGeneralized,
        "network_attr": "policy_network",
        "state_dict": "final_experiments/ppo_general_with_reward_shaping_2026-05-07_20-52-04/final/final_model_50003737.pt",
    },
    "random": {
        "class": RandomAgent,
    },
    "heuristic_agent_mki": {
        "class": HeuristicAgentMKI,
    },
    "heuristic_agent_mkii": {
        "class": HeuristicAgentMKII,
    },
}


class EnvFactory:
    def __init__(self, card_list):
        self.card_list = card_list

    def __call__(self):
        return FluxxEnv(
            Game(2, self.card_list, disable_game_messages=True), 2, render_mode="human"
        )


class AgentFactory:
    def __init__(self, agent_entry, game_config):
        self.agent_entry = agent_entry
        self.game_config = game_config

        if "state_dict" in agent_entry:
            path = f"{PROJECT_ROOT}/{agent_entry['state_dict']}"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint not found: {path}")

    def __call__(self):
        agent = self.agent_entry["class"](self.game_config, 0)

        if "state_dict" in self.agent_entry:
            network = getattr(agent, self.agent_entry["network_attr"])
            network.load_state_dict(
                torch.load(f"{PROJECT_ROOT}/{self.agent_entry['state_dict']}", map_location="cpu"),
                strict=self.agent_entry.get("strict", True),
            )
            network.eval()

        return agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate agents by pitting them against each other.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("games", type=int, help="Number of games to run")
    parser.add_argument("-t", "--turn-limit", type=int, default=1000,
                        help="Maximum number of turns per game")
    parser.add_argument("-cls", "--card-lists", nargs="+", type=str,
                        default=["base_deck"],
                        choices=["base_deck", "expanded_deck", "simple_fluxx_deck"])
    return parser.parse_args()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()
    results: dict[tuple[str, str], dict] = {}

    try:
        n_workers = max(1, len(os.sched_getaffinity(0)) - 1)
    except AttributeError:
        n_workers = max(1, os.cpu_count() - 1)

    print(f"RUNNING {args.games} GAMES WITH {n_workers} WORKERS")
    for card_list_name in args.card_lists:
        print(f"PLAYING WITH: CARD LIST: {card_list_name}")

        card_list = CARD_LISTS[card_list_name]
        generate_embedding_table(card_list)
        embedding_table = get_embedding_table()

        game_config = Game(2, card_list, disable_game_messages=True).game_config
        env_factory = EnvFactory(card_list)
        agent_battler = AgentBattler()

        agent_factories = {
            name: AgentFactory(AGENT_REGISTRY[name], game_config)
            for name in AGENT_NAMES
        }

        seen_matchups: set[tuple[str, str]] = set()
        for agent_name in AGENT_NAMES:
            for other_agent_name in AGENT_NAMES:
                if agent_name == other_agent_name or (other_agent_name, agent_name) in seen_matchups:
                    continue
                seen_matchups.add((agent_name, other_agent_name))

                print(f"RUNNING {agent_name} vs {other_agent_name}")
                result = agent_battler.run_games(
                    game_count=args.games,
                    turn_limit=args.turn_limit,
                    n_workers=n_workers,
                    env_factory=env_factory,
                    agent_factories=[
                        agent_factories[agent_name],
                        agent_factories[other_agent_name],
                    ],
                    embedding_table=embedding_table
                )
                results[(agent_name, other_agent_name)] = result
                print(f"RESULTS: {result}")
    print(results)