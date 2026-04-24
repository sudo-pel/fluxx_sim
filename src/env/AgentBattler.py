"""
Plan

init(self, game)

run_games(self, players, game_count, turn_limit) -> dict containing [player_wins -> int, draws -> int, mean_game_length -> int, total_games -> int]


"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.agents.Agent import Agent
from src.training.TrainingEnums import GameLogConfig
from src.env.FluxxEnv import FluxxEnv
from src.env.Logger import GameLogLogger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class AgentBattler:
    def __init__(self, env: FluxxEnv):
        self.env: FluxxEnv = env

    def run_games(self, agents: list[Agent], game_count: int, turn_limit: int, log_games: bool = False, log_config: Optional[GameLogConfig] = None):
        if log_games and log_config is None:
            raise ValueError("log_name must be specified if log_games is True")

        wins = {f"player_{i}": 0 for i in range(self.env.game.player_count)}
        wins["draws"] = 0
        agents = {f"player_{i}": agent for i, agent in enumerate(agents)}

        game_lengths = []

        if log_games:
            log_name = f"{log_config.log_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(f"{PROJECT_ROOT}/game_logs/{log_name}")

        for i in range(game_count):
            self.env.reset()

            if log_games:
                game_logger = GameLogLogger(f"{log_name}/game_{i}")
                self.env.game.logger = game_logger

            for agent in self.env.agent_iter():
                observation, _, termination, truncation, _ = self.env.last()

                if termination or truncation:
                    action = None
                else:
                    action, _, _ = agents[agent].act(observation)
                    action = self.env.decode_action(action)

                self.env.step(action)

                if self.env.game.turn_count >= turn_limit:
                    break

            if self.env.game.winner is None:
                wins["draws"] += 1
            else:
                wins[f"player_{self.env.game.winner}"] += 1
            game_lengths.append(self.env.game.turn_count)
            self.env.close()

        return {
            "player_wins": wins,
            "total_games": game_count,
            "average_game_length": sum(game_lengths) / len(game_lengths)
        }