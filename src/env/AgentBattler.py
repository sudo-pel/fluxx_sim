"""
Plan

init(self, game)

run_games(self, players, game_count, turn_limit) -> dict containing [player_wins -> int, draws -> int, mean_game_length -> int, total_games -> int]


"""
from src.agents.Agent import Agent
from src.env.FluxxEnv import FluxxEnv


class AgentBattler:
    def __init__(self, env: FluxxEnv):
        self.env: FluxxEnv = env

    def run_games(self, agents: list[Agent], game_count: int, turn_limit: int):
        wins = {f"player_{i}": 0 for i in range(self.env.game.player_count)}
        agents = {f"player_{i}": agent for i, agent in enumerate(agents)}

        game_lengths = []

        for i in range(game_count):
            self.env.reset()

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