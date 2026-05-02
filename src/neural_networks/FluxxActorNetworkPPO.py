from __future__ import annotations
import torch
import torch.nn as nn

from src.agents.card_embeddings import CARD_EMBED_DIM, get_embedding_table

class DeepSetsPool(nn.Module):
    def __init__(
        self,
        input_dim: int = CARD_EMBED_DIM,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        card_embeddings: torch.Tensor,   # (batch, max_set_size, input_dim)
        mask: torch.Tensor,              # (batch, max_set_size), 1=real, 0=pad
    ) -> torch.Tensor:
        transformed = self.phi(card_embeddings)              # (B, S, H)
        transformed = transformed * mask.unsqueeze(-1).float()
        pooled = transformed.sum(dim=1)                      # (B, H)
        return self.rho(pooled)                              # (B, output_dim)


class FluxxStateEncoder(nn.Module):
    """
    Takes a dict of tensors resulting from PPOAgentGeneralized.collate() and returns a "proper" state encoding vector
    """
    POOL_OUTPUT_DIM = 256
    DECISION_CONTEXT_DIM = 19
    NUM_POOLS = 6  # hand, discard, own keepers, opp keepers, goals, rules

    def __init__(self):
        super().__init__()
        self.hand_pool = DeepSetsPool()
        self.discard_pool = DeepSetsPool()
        self.own_keepers_pool = DeepSetsPool()
        self.opp_keepers_pool = DeepSetsPool()
        self.goals_pool = DeepSetsPool()
        self.rules_pool = DeepSetsPool()

    @property
    def output_dim(self) -> int:
        return (
            self.DECISION_CONTEXT_DIM
            + self.NUM_POOLS * self.POOL_OUTPUT_DIM
            + 2  # draw_pile_size, opponent_hand_size
            + FluxxStateEncoder.NUM_POOLS # also encode cardinality of each pooled set
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                obs["decision_context"],
                self.hand_pool(obs["hand_embeds"], obs["hand_mask"]),
                self.discard_pool(obs["discard_embeds"], obs["discard_mask"]),
                self.own_keepers_pool(obs["own_keeper_embeds"], obs["own_keeper_mask"]),
                self.opp_keepers_pool(obs["opp_keeper_embeds"], obs["opp_keeper_mask"]),
                self.goals_pool(obs["goal_embeds"], obs["goal_mask"]),
                self.rules_pool(obs["rules_embeds"], obs["rules_mask"]),
                obs["draw_pile_size"],
                obs["opponent_hand_size"],
                obs["hand_size"],
                obs["discard_pile_size"],
                obs["own_keepers_in_play_count"],
                obs["opponent_keepers_in_play_count"],
                obs["goals_in_play_count"],
                obs["rules_in_play_count"]
            ],
            dim=-1,
        )


class FluxxActorNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int,
        card_list: list[str],
        hidden_dim: int = 256,
        embed_dim: int = CARD_EMBED_DIM,
    ):
        super().__init__()
        assert action_dim == len(card_list) + 1, (
            f"action_dim ({action_dim}) must equal len(card_list) + 1 "
            f"({len(card_list) + 1}); the +1 is the no-op slot."
        )

        self.encoder = FluxxStateEncoder()

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.query_head = nn.Linear(hidden_dim, embed_dim)
        self.no_op_head = nn.Linear(hidden_dim, 1)

        embedding_table = get_embedding_table()
        card_embeds = torch.stack(
            [
                torch.as_tensor(embedding_table[name], dtype=torch.float32)
                for name in card_list
            ],
            dim=0,
        )  # (num_cards, embed_dim)
        self.register_buffer("card_embeds", card_embeds, persistent=False)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        state_vec = self.encoder(obs)               # (B, encoder.output_dim)
        trunk_out = self.trunk(state_vec)           # (B, hidden_dim)

        query = self.query_head(trunk_out)          # (B, embed_dim)
        # (B, embed_dim) @ (embed_dim, num_cards) -> (B, num_cards)
        card_logits = query @ self.card_embeds.T

        no_op_logit = self.no_op_head(trunk_out)    # (B, 1)

        return torch.cat([card_logits, no_op_logit], dim=-1)  # (B, num_cards + 1)