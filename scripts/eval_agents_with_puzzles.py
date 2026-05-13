import argparse
import copy
import os

from src.agents.PPOAgentGeneralizedWithHeuristic import PPOAgentGeneralizedWithHeuristic
from src.game.FluxxEnums import GameConfig
from src.game.game_states import puzzle_a, puzzle_a2, puzzle_b, puzzle_b2, puzzle_c, puzzle_c2
from src.training.TrainingEnums import GameLogConfig

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import torch

from src.agents.DQNAgent import DQNAgent
from src.agents.DQNAgentGeneralized import DQNAgentGeneralized
from src.agents.HeuristicAgentMKI import HeuristicAgentMKI
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.agents.PPOAgentGeneralized import PPOAgentGeneralized
from src.agents.RandomAgent import RandomAgent
from src.env.AgentBattlerParallel import AgentBattler
from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CARD_LISTS = {
    "base_deck": card_lists.base_deck,
    "expanded_deck": card_lists.expanded_deck,
}

AGENT_NAMES = [
    "ppo",
    "ppo_general",
    "ppo_general_with_heuristic",
    "ppo_general_with_reward_shaping",
    "ppo_general_with_reward_shaping_and_heuristic",
    "dqn_general",
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
    },
    "ppo_general_with_heuristic": {
        "class": PPOAgentGeneralizedWithHeuristic,
        "network_attr": "policy_network",
        "state_dict": "final_experiments/ppo_general_2026-05-02_09-11-14/final/final_model_50006336.pt",
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
    "ppo_general_with_reward_shaping_and_heuristic": {
        "class": PPOAgentGeneralizedWithHeuristic,
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

PUZZLES = {
    "puzzle_a": {
        "game_state": puzzle_a,
        "card_list": "base_deck",
        "testee_player_number": 0
    },
    "puzzle_a2": {
        "game_state": puzzle_a2,
        "card_list": "expanded_deck",
        "testee_player_number": 0
    },
    "puzzle_b": {
        "game_state": puzzle_b,
        "card_list": "base_deck",
        "testee_player_number": 0
    },
    "puzzle_b2": {
        "game_state": puzzle_b2,
        "card_list": "expanded_deck",
        "testee_player_number": 0
    },
    "puzzle_c": {
        "game_state": puzzle_c,
        "card_list": "base_deck",
        "testee_player_number": 1
    },
    "puzzle_c2": {
        "game_state": puzzle_c2,
        "card_list": "expanded_deck",
        "testee_player_number": 1
    }
}

class EnvFactory:
    def __init__(self, card_list, force_game_state=None):
        self.card_list = card_list
        self.force_game_state = force_game_state

    def __call__(self):
        return FluxxEnv(
            Game(2, self.card_list, disable_game_messages=True,
                 force_game_state=copy.deepcopy(self.force_game_state)),
            2, render_mode="human"
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
                strict=False
            )
            network.eval()

        return agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate agents by pitting them against each other.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "games",
        type=int,
        help="Number of games to run"
    )
    parser.add_argument(
        "-t",
        "--turn-limit",
        type=int,
        default=1000,
        help="Maximum number of turns per game."
    )
    parser.add_argument(
        "-a",
        "--agents",
        nargs="*",
        type=str,
        default=[],
        choices=AGENT_NAMES,
        help="Agents to evaluate puzzles against. When specified, only matchups containing agents in this list will be run.",
    )
    parser.add_argument(
        "-l",
        "--log-name",
        type=str,
        default=None,
        help="Name of log file to write to. If not specified, no log file will be written.",
    )
    parser.add_argument(
        "-p",
        "--puzzles",
        nargs="*",
        type=str,
        default=[p for p in PUZZLES.keys()],
        help="Puzzles to attempt. If not specifies, all puzzles will be attempted.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=str,
        default="1",
        help="Number of workers to use for running parallel games. Default is 1. Set to 'max' to use all available cores.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    args = parse_args()
    results: dict[tuple[str, str], dict] = {}

    if args.workers == "max":
        try:
            n_workers = max(1, len(os.sched_getaffinity(0)) - 1)
        except AttributeError:
            n_workers = max(1, os.cpu_count() - 1)
        n_workers = min(args.games, n_workers)
    elif args.workers.isdigit() and int(args.workers) > 0:
        n_workers = int(args.workers)
    else:
        raise ValueError(f"Invalid value for --workers: {args.workers}. Must be 'max' or a positive integer.")

    print(f"RUNNING {args.games} GAMES WITH {n_workers} WORKERS")
    for puzzle in args.puzzles:
        print(f"PLAYING PUZZLE: {puzzle}")

        puzzle_data = PUZZLES[puzzle]
        card_list_name = puzzle_data["card_list"]
        card_list = CARD_LISTS[card_list_name]
        game_state = puzzle_data["game_state"]

        game_config = GameConfig(
            2, card_list,
        )
        env_factory = EnvFactory(card_list, force_game_state=game_state)
        agent_battler = AgentBattler()

        agent_factories = {
            name: AgentFactory(AGENT_REGISTRY[name], game_config)
            for name in AGENT_NAMES
        }

        seen_matchups: set[tuple[str, str]] = set()
        for agent_name in AGENT_NAMES:
            if args.agents and agent_name not in args.agents:
                continue

            if args.log_name is not None:
                logging_config = GameLogConfig(
                    f"{args.log_name}_{agent_name}_{puzzle}",
                   [],
                )

            if puzzle_data["testee_player_number"] == 0:
                current_agent_factories=[
                    agent_factories[agent_name],
                    agent_factories["random"],
                ]
                if args.log_name is not None:
                    logging_config.agent_names = [agent_name, "random"]
            else:
                current_agent_factories=[
                    agent_factories["random"],
                    agent_factories[agent_name],
                ]
                if args.log_name is not None:
                    logging_config.agent_names = ["random", agent_name]

            print(f"RUNNING {agent_name}")
            result = agent_battler.run_games(
                game_count=args.games,
                turn_limit=args.turn_limit,
                n_workers=n_workers,
                env_factory=env_factory,
                agent_factories=current_agent_factories,
                embedding_table_name=card_list_name,
                log_config=logging_config if args.log_name is not None else None,
                log_games=args.log_name is not None,
            )
            results[(puzzle, agent_name)] = result
            print(f"RESULTS: {result}")
    print(results)