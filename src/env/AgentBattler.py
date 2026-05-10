import copy
import multiprocessing as mp
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.Agent import Agent
from src.training.TrainingEnums import GameLogConfig
from src.env.FluxxEnv import FluxxEnv
from src.env.Logger import GameLogLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Standalone functions (callable from forked workers without pickling issues) ──

def run_games(env: FluxxEnv, agents: dict[str, Agent], game_count: int, turn_limit: int, step_limit: int, print_game_number: bool = False, log_games: bool = False, log_name: str = None) -> dict:
    wins = {f"player_{i}": 0 for i in range(env.game.player_count)}
    wins["draws"] = 0
    game_lengths = []

    for i in range(game_count):
        if print_game_number:
            print(f"game {i}")
        env.reset()
        timestep = 0

        if log_games and log_name:
            game_logger = GameLogLogger(f"{log_name}/game_{i}")
            env.game.logger = game_logger

        for agent in env.agent_iter():
            if timestep >= step_limit:
                break
            timestep += 1

            observation, _, termination, truncation, _ = env.last()

            if termination or truncation or env.game.winner is not None:
                action = None
            else:
                action, _, _ = agents[agent].act(observation)
                action = env.decode_action(action)

            env.step(action)

            if env.game.turn_count >= turn_limit:
                break

        if env.game.winner is None:
            wins["draws"] += 1
        else:
            wins[f"player_{env.game.winner}"] += 1
        game_lengths.append(env.game.turn_count)
        env.close()

    return {
        "player_wins": wins,
        "total_games": game_count,
        "average_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
    }


def parallel_worker(env, agents_list, num_games, turn_limit, step_limit, print_game_number, result_queue):
    env_copy = copy.deepcopy(env)
    agents_copy = {f"player_{i}": copy.deepcopy(a) for i, a in enumerate(agents_list)}
    result = run_games(env_copy, agents_copy, num_games, turn_limit, step_limit, print_game_number)
    result_queue.put(result)


def aggregate_results(results: list[dict]) -> dict:
    combined_wins = {}
    total_games = 0
    total_weighted_length = 0.0

    for r in results:
        for key, val in r["player_wins"].items():
            combined_wins[key] = combined_wins.get(key, 0) + val
        total_games += r["total_games"]
        total_weighted_length += r["average_game_length"] * r["total_games"]

    return {
        "player_wins": combined_wins,
        "total_games": total_games,
        "average_game_length": total_weighted_length / total_games if total_games > 0 else 0,
    }

class AgentBattler:
    def __init__(self, env: FluxxEnv):
        self.env: FluxxEnv = env

    def run_games(self, agents: list[Agent], game_count: int, turn_limit: int, log_games: bool = False, log_config: Optional[GameLogConfig] = None, step_limit: Optional[int] = None, print_game_number: bool = False, num_workers: int = 1):

        if log_games and log_config is None:
            raise ValueError("log_config must be specified if log_games is True")

        if step_limit is None:
            step_limit = turn_limit * 10

        # Sequential path allowing for game logging and not requiring results aggregation
        if num_workers <= 1:
            agents_dict = {f"player_{i}": agent for i, agent in enumerate(agents)}
            log_name = None
            if log_games:
                log_name = f"{log_config.log_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
                os.makedirs(f"{PROJECT_ROOT}/game_logs/{log_name}")
            return run_games(self.env, agents_dict, game_count, turn_limit, step_limit, print_game_number, log_games, log_name)

        # Parallel path
        if log_games:
            print("Warning: game logging is disabled in parallel mode.")

        # Distribute games evenly across workers
        chunk_size = game_count // num_workers
        remainder = game_count % num_workers

        result_queue = mp.Queue()
        processes = []

        for i in range(num_workers):
            n_games = chunk_size + (1 if i < remainder else 0)
            if n_games == 0:
                continue
            p = mp.Process(
                target=parallel_worker,
                args=(self.env, agents, n_games, turn_limit, step_limit, print_game_number, result_queue),
            )
            processes.append(p)
            p.start()

        results = [result_queue.get() for _ in processes]
        for p in processes:
            p.join()

        return aggregate_results(results)