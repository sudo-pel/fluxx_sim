import numpy.typing as npt
from src.game.Game import Game

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers


def env(**kwargs):
    raw_env = FluxxEnv(**kwargs)
    raw_env = wrappers.AssertOutOfBoundsWrapper(raw_env)
    raw_env = wrappers.OrderEnforcingWrapper(raw_env)
    return raw_env

class FluxxEnv(AECEnv):

    metadata = {"render_modes": ["human"], "name": "card_game_v0"}

    # Seed unused as of yet but added for completeness
    def __init__(self, game: Game, num_players: int = 2, render_mode=None, seed: np.random.SeedSequence = None):
        super().__init__()

        self.game = game
        self.card_vector_length = len(game.card_list)
        self.card_to_index: dict[str, int] = {
            card: i for i, card in enumerate(game.card_list)
        }
        self.index_to_card = {
            i: card for i, card in enumerate(game.card_list)
        }

        self.num_players = num_players
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.terminations        = {agent: False for agent in self.agents}
        self.truncations         = {agent: False for agent in self.agents}
        self.rewards             = {agent: 0.0   for agent in self.agents}
        self._cumulative_rewards  = {agent: 0.0 for agent in self.agents}
        self.infos               = {agent: {}    for agent in self.agents}

        self.game.reset()

        # get player to move first
        acting_player = self.game.check_current_phase().acting_player
        self.agent_selection = f"player_{acting_player}"

        # Verify consistency
        phase = self.game.check_current_phase()
        assert self.game.player_turn == phase.acting_player, (
            f"Reset inconsistency: player_turn={self.game.player_turn}, "
            f"phase.acting_player={phase.acting_player}, "
            f"phase.type={phase.type.name}, "
            f"stack: {[(p.type.name, p.acting_player) for p in self.game.stack]}"
        )

        # return observation for the first player to act
        return self.observe(self.agent_selection)

    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self.game.step(action)

        # Check termination
        if self.game.winner is not None:
            for agent in self.agents:
                self.terminations[agent] = True
                self.rewards[agent] = -1.0
            winner = self.determine_winner()
            self.rewards[winner] = 1.0
        elif self.game.steps > self.game.step_limit:
            print("Game step limit reached")
            for agent in self.agents:
                self.truncations[agent] = True
                self.rewards[agent] = -1.0

        # Accumulate rewards into _cumulative_rewards
        self._accumulate_rewards()

        # Advance to the next agent (just check what's on top of the game stack)
        game_state = self.game.check_current_phase()
        self.agent_selection = f"player_{game_state.acting_player}"

    def observe(self, agent):
        return self.game.get_game_state()

    def decode_action(self, action_index: int) -> str:
        if action_index == self.card_vector_length:
            return "no_free_action"
        return self.index_to_card[action_index]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        pass

    def get_player_number(self, agent) -> int:
        """
        Convert agent identifier ('player_0') to player number (0)
        """
        return self.possible_agents.index(agent)

    def determine_winner(self):
        if self.game.winner == 0:
            return "player_0"
        elif self.game.winner == 1:
            return "player_1"
        else:
            raise Exception("determine_winner called with invalid winner")
