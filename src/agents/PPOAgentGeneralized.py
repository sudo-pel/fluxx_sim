from __future__ import annotations

import torch
from torch.distributions import Categorical

from src.agents.Agent import Agent
from src.agents.utils import generalized_agent_utils
from src.agents.utils.generalized_agent_utils import BufferEntry
from src.neural_networks.FluxxActorNetworkPPO import FluxxActorNetwork

from src.agents.utils.agent_utils import decision_context_vectors
from src.game.FluxxEnums import GameState
from src.game.FluxxEnums import GameConfig

class PPOAgentGeneralized(Agent):
    def __init__(
        self,
        game_config: GameConfig,
        player_number: int,
    ):
        super(PPOAgentGeneralized, self).__init__()
        self.game_config = game_config
        self.player_number = player_number
        self.decision_context_vectors = decision_context_vectors

        action_dim = len(game_config.card_list) + 1
        self.policy_network = FluxxActorNetwork(
            action_dim=action_dim,
            card_list=game_config.card_list,
        )
        self.action_dim = action_dim

        self.card_to_index = {c: i for i, c in enumerate(game_config.card_list)}

    def act(self, game_state: "GameState") -> tuple[int, torch.Tensor, BufferEntry]:
        """
        Returns:
            action: int (sampled action index)
            log_prob: 0-d torch.Tensor (log probability of the sampled action)
            entry: BufferEntry (to be stored in the replay buffer)
        """
        device = next(self.policy_network.parameters()).device

        entry = generalized_agent_utils.extract_entry(game_state, self.player_number, self.game_config)

        if not entry.action_mask.any():
            raise RuntimeError(
                f"All-zero action mask in phase {game_state.stack[-1].type}. "
                f"Hand size: {len(game_state.hands[self.player_number])}, "
                f"own keepers: {len(game_state.keepers[self.player_number])}"
            )

        obs_dict = generalized_agent_utils.collate([entry], device)   # batch of 1

        logits = self.policy_network(obs_dict)              # (1, action_dim)
        logits = logits.masked_fill(~obs_dict["action_mask"], float("-inf"))

        dist = Categorical(logits=logits)
        action = dist.sample()                              # (1,)
        log_prob = dist.log_prob(action)                    # (1,)

        return action.item(), log_prob.squeeze(0), entry