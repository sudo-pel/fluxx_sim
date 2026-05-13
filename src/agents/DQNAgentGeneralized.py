from __future__ import annotations

import numpy as np
import torch

from src.agents.Agent import Agent
from src.agents.utils import generalized_agent_utils
from src.agents.utils.agent_utils import (
    decision_context_vectors,
)
from src.game.FluxxEnums import GameConfig, GameState
from src.neural_networks.FluxxActorNetwork import FluxxActorNetwork
from src.agents.utils.generalized_agent_utils import BufferEntry

MAX_HAND_SIZE = 100
MAX_DISCARD_SIZE = 100
MAX_KEEPERS_PER_PLAYER = 100
MAX_OPP_KEEPERS_TOTAL = 100
MAX_GOALS_IN_PLAY = 100
MAX_RULES_IN_PLAY = 100


class DQNAgentGeneralized(Agent):
    def __init__(
        self,
        game_config: GameConfig,
        player_number: int,
        seed: np.random.SeedSequence = None,
    ):
        super(DQNAgentGeneralized, self).__init__()
        self.game_config = game_config
        self.player_number = player_number
        self.decision_context_vectors = decision_context_vectors

        action_dim = len(game_config.card_list) + 1
        self.q_network = FluxxActorNetwork(
            action_dim=action_dim,
            card_list=game_config.card_list,
        )
        self.action_dim = action_dim

        self.card_to_action_index = {c: i for i, c in enumerate(game_config.card_list)}
        self.card_to_embed_id = {c: i + 1 for i, c in enumerate(game_config.card_list)}

        self.card_to_index = self.card_to_action_index

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def act(self, game_state: "GameState", epsilon: float = 0.0) -> tuple[int, torch.Tensor, BufferEntry]:
        device = next(self.q_network.parameters()).device

        entry = generalized_agent_utils.extract_entry(game_state, self.player_number, self.game_config)
        obs_dict = generalized_agent_utils.collate([entry], device)  # batch of 1

        with torch.no_grad():
            q_values = self.q_network(obs_dict)               # (1, action_dim)
            masked_q = q_values.masked_fill(~obs_dict["action_mask"], float("-inf"))

        if self.rng.random() < epsilon:
            legal = np.flatnonzero(entry.action_mask.astype(bool))
            action = int(self.rng.choice(legal))
        else:
            action = int(masked_q.argmax(dim=-1).item())

        # Get q-value of chosen action for logging purposes
        chosen_q = masked_q[0, action].detach()

        return action, chosen_q, entry

    def encode(self, game_state: "GameState") -> BufferEntry:
        return generalized_agent_utils.extract_entry(game_state, self.player_number, self.game_config)