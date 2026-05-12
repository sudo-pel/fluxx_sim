import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from src.agents.Agent import Agent
from src.agents.card_embeddings import set_embedding_table
from src.training.TrainingEnums import GameLogConfig
from src.env.FluxxEnv import FluxxEnv
from src.env.Logger import GameLogLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_game_batch(
    game_count: int,
    turn_limit: int,
    step_limit: int,
    env_factory: Callable[[], FluxxEnv],
    agent_factories: list[Callable[[], Agent]],
    player_count: int,
    log_games: bool,
    log_name: Optional[str],
    game_offset: int,
    embedding_table: dict[str, np.ndarray]
) -> dict:
    """
    Large amount of non pickleable state means that worker processes have to construct their own agents and environments
    """
    set_embedding_table(embedding_table)
    env = env_factory()
    agents = {f"player_{i}": factory() for i, factory in enumerate(agent_factories)}
    for i, agent in enumerate(agents.values()):
        agent.player_number = i

    wins = {f"player_{i}": 0 for i in range(player_count)}
    wins["draws"] = 0
    wins["errors"] = 0
    game_lengths = []
    blunders = {f"player_{i}": 0 for i in range(player_count)}

    for i in range(game_count):
        env.reset()
        timestep = 0
        global_game_index = game_offset + i
        error_occurred = False
        last_to_move = None

        if log_games:
            game_logger = GameLogLogger(f"{log_name}/game_{global_game_index}")
            env.game.logger = game_logger

        for agent in env.agent_iter():
            if env.game.winner is None:
                last_to_move = agent

            if timestep >= step_limit:
                break
            timestep += 1

            observation, _, termination, truncation, _ = env.last()

            if termination or truncation or env.game.winner is not None:
                action = None
            else:
                action, _, _ = agents[agent].act(observation)
                action = env.decode_action(action)

            try:
                env.step(action)
            except Exception:
                error_occurred = True
                break

            if env.game.turn_count >= turn_limit:
                break

        if error_occurred:
            wins["errors"] += 1
        elif env.game.winner is None:
            wins["draws"] += 1
        else:
            outcome = f"player_{env.game.winner}"
            wins[outcome] += 1
            for agent_name in agents.keys():
                if agent_name != outcome and last_to_move == agent_name:
                    blunders[agent_name] += 1

        game_lengths.append(env.game.turn_count)
        env.close()

    return {"wins": wins, "game_lengths": game_lengths, "blunders": blunders}


class AgentBattler:
    def run_games(
        self,
        game_count: int,
        turn_limit: int,
        env_factory: Callable[[], FluxxEnv],
        agent_factories: list[Callable[[], Agent]],
        log_games: bool = False,
        log_config: Optional[GameLogConfig] = None,
        step_limit: Optional[int] = None,
        n_workers: int = 1,
        embedding_table: dict[str, np.ndarray] = None
    ):
        if log_games and log_config is None:
            raise ValueError("log_config must be specified if log_games is True")

        if step_limit is None:
            step_limit = turn_limit * 10

        if embedding_table is None:
            raise Exception("embedding_table must be specified")

        player_count = len(agent_factories)

        log_name = None
        if log_games:
            log_name = f"{log_config.log_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(f"{PROJECT_ROOT}/game_logs/{log_name}", exist_ok=True)

        # Split game_count into roughly equal batches
        base, remainder = divmod(game_count, n_workers)
        batches = []
        offset = 0
        for w in range(n_workers):
            count = base + (1 if w < remainder else 0)
            if count == 0:
                continue
            batches.append((count, offset))
            offset += count

        # Aggregate
        total_wins = {f"player_{i}": 0 for i in range(player_count)}
        total_wins["draws"] = 0
        total_wins["errors"] = 0
        total_blunders = {f"player_{i}": 0 for i in range(player_count)}
        all_lengths = []

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    _run_game_batch,
                    game_count=count,
                    turn_limit=turn_limit,
                    step_limit=step_limit,
                    env_factory=env_factory,
                    agent_factories=agent_factories,
                    player_count=player_count,
                    log_games=log_games,
                    log_name=log_name,
                    game_offset=offset,
                    embedding_table=embedding_table
                )
                for count, offset in batches
            ]

            for future in as_completed(futures):
                result = future.result()
                for key in total_wins:
                    total_wins[key] += result["wins"][key]
                for key in total_blunders:
                    total_blunders[key] += result["blunders"][key]
                all_lengths.extend(result["game_lengths"])

        return {
            "player_wins": total_wins,
            "total_games": game_count,
            "average_game_length": sum(all_lengths) / len(all_lengths),
            "blunders": total_blunders
        }