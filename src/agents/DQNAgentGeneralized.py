from __future__ import annotations

import numpy as np
import torch

from src.agents.Agent import Agent
from src.agents.agent_utils import (
    convert_decision_encoding,
    decision_context_vectors,
    populate_card_vector,
)
from src.agents.card_embeddings import CARD_EMBED_DIM, get_embedding_table
from src.game.FluxxEnums import GameConfig, GamePhaseType, GameState
from src.game.cards.card_data import CARD_DATA
from src.neural_networks.FluxxActorNetworkDQN import FluxxActorNetworkDQN
from src.training.TrainingEnums import BufferEntry

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
        self.q_network = FluxxActorNetworkDQN(
            action_dim=action_dim,
            card_list=game_config.card_list,
        )
        self.action_dim = action_dim

        self.card_to_index = {c: i for i, c in enumerate(game_config.card_list)}

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def extract_entry(self, game_state: GameState) -> BufferEntry:
        current_phase = game_state.stack[-1]

        decision_context = convert_decision_encoding(
            self.decision_context_vectors[current_phase.type],
            current_phase.decisions_left,
            current_phase.counter,
            current_phase.on_complete,
        ).astype(np.float32)

        hand = list(game_state.hands[self.player_number])
        own_keepers = list(game_state.keepers[self.player_number])
        discard = list(game_state.discard_pile)
        goals = list(game_state.goals)
        rules = list(game_state.rules)

        opp_keepers = []
        for i, keeper_list in enumerate(game_state.keepers):
            if i != self.player_number:
                opp_keepers.extend(keeper_list)

        draw_pile_size = len(game_state.draw_pile)
        opponent_hand_size = sum(
            len(game_state.hands[i])
            for i in range(self.game_config.player_count)
            if i != self.player_number
        )
        hand_size = len(game_state.hands[self.player_number])
        discard_pile_size = len(game_state.discard_pile)
        own_keepers_in_play_count = len(game_state.keepers[self.player_number])
        opponent_keepers_in_play_count = sum(
            len(game_state.keepers[i])
            for i in range(self.game_config.player_count)
            if i != self.player_number
        )
        goals_in_play_count = len(game_state.goals)
        rules_in_play_count = len(game_state.rules)

        action_mask = self.build_action_mask(game_state, current_phase)

        return BufferEntry(
            decision_context=decision_context,
            hand=hand,
            discard=discard,
            own_keepers=own_keepers,
            opp_keepers=opp_keepers,
            goals=goals,
            rules=rules,
            draw_pile_size=draw_pile_size,
            opponent_hand_size=opponent_hand_size,
            action_mask=action_mask,
            hand_size=hand_size,
            discard_pile_size=discard_pile_size,
            own_keepers_in_play_count=own_keepers_in_play_count,
            opponent_keepers_in_play_count=opponent_keepers_in_play_count,
            goals_in_play_count=goals_in_play_count,
            rules_in_play_count=rules_in_play_count,
        )

    def build_action_mask(self, game_state, current_phase) -> np.ndarray:
        cards_in_hand_vec = populate_card_vector(
            self.game_config.card_list, game_state.hands[self.player_number]
        )
        keeper_vecs = [
            populate_card_vector(self.game_config.card_list, kl)
            for kl in game_state.keepers
        ]
        own_keeper_vec = keeper_vecs[self.player_number]
        other_keeper_vecs = (
            keeper_vecs[: self.player_number]
            + keeper_vecs[self.player_number + 1 :]
        )
        rules_vec = populate_card_vector(self.game_config.card_list, game_state.rules)
        goals_vec = populate_card_vector(self.game_config.card_list, game_state.goals)

        action_mask = np.zeros(len(self.game_config.card_list), dtype=np.int8)
        no_free_action_legal = False

        current_phase_type = current_phase.type
        if current_phase_type == GamePhaseType.PLAY_CARD_FOR_TURN:
            action_mask = cards_in_hand_vec
        elif current_phase_type == GamePhaseType.DISCARD_CARD_FROM_HAND:
            action_mask = cards_in_hand_vec
        elif current_phase_type == GamePhaseType.DISCARD_KEEPER:
            action_mask = own_keeper_vec
        elif current_phase_type == GamePhaseType.DISCARD_RULE_IN_PLAY:
            action_mask = rules_vec
        elif current_phase_type == GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE:
            action_mask = populate_card_vector(
                self.game_config.card_list,
                [c.name for c in current_phase.latent_space],
            )
        elif current_phase_type == GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND:
            action_mask = np.bitwise_or.reduce((*keeper_vecs, rules_vec, goals_vec))
        elif current_phase_type == GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND:
            action_mask = populate_card_vector(
                self.game_config.card_list,
                [c.name for c in current_phase.latent_space],
            )
        elif current_phase_type == GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE:
            action_mask = populate_card_vector(
                self.game_config.card_list,
                [
                    c
                    for c in game_state.discard_pile
                    if CARD_DATA[c]["card_type"] in ("RULE", "ACTION")
                ],
            )
        elif current_phase_type == GamePhaseType.DISCARD_KEEPER_IN_PLAY:
            action_mask = np.bitwise_or.reduce(keeper_vecs)
        elif current_phase_type == GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT:
            action_mask = populate_card_vector(
                self.game_config.card_list,
                [c.name for c in current_phase.latent_space],
            )
        elif current_phase_type == GamePhaseType.SELECT_KEEPER_TO_STEAL:
            action_mask = np.bitwise_or.reduce(other_keeper_vecs)
        elif current_phase_type == GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE:
            action_mask = np.bitwise_or.reduce(other_keeper_vecs)
        elif current_phase_type == GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE:
            action_mask = own_keeper_vec.copy()
            action_mask[self.card_to_index[current_phase.labelled_card.name]] = 0
        elif current_phase_type == GamePhaseType.ACTIVATE_FREE_ACTION:
            action_mask = populate_card_vector(
                self.game_config.card_list, list(game_state.available_free_actions)
            )
            no_free_action_legal = True
        elif current_phase_type == GamePhaseType.DISCARD_OWN_KEEPER_IN_PLAY:
            action_mask = own_keeper_vec
        elif current_phase_type == GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND:
            valid = [
                c
                for c in game_state.hands[self.player_number]
                if CARD_DATA[c]["card_type"] in [t.name for t in current_phase.card_types]
            ]
            action_mask = populate_card_vector(self.game_config.card_list, valid)
            no_free_action_legal = True
        elif current_phase_type == GamePhaseType.DISCARD_GOAL_IN_PLAY:
            action_mask = goals_vec
        else:
            raise Exception(f"Invalid game phase type: {current_phase_type}")

        action_mask = np.append(action_mask, 0).astype(np.int8)
        if no_free_action_legal:
            action_mask[-1] = 1
        return action_mask

    def collate(
        self,
        entries: list[BufferEntry],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        N = len(entries)

        embedding_table = get_embedding_table()

        decision_context = np.empty((N, 19), dtype=np.float32)
        hand_embeds = np.zeros((N, MAX_HAND_SIZE, CARD_EMBED_DIM), dtype=np.float32)
        hand_mask = np.zeros((N, MAX_HAND_SIZE), dtype=np.float32)
        discard_embeds = np.zeros((N, MAX_DISCARD_SIZE, CARD_EMBED_DIM), dtype=np.float32)
        discard_mask = np.zeros((N, MAX_DISCARD_SIZE), dtype=np.float32)
        own_keeper_embeds = np.zeros((N, MAX_KEEPERS_PER_PLAYER, CARD_EMBED_DIM), dtype=np.float32)
        own_keeper_mask = np.zeros((N, MAX_KEEPERS_PER_PLAYER), dtype=np.float32)
        opp_keeper_embeds = np.zeros((N, MAX_OPP_KEEPERS_TOTAL, CARD_EMBED_DIM), dtype=np.float32)
        opp_keeper_mask = np.zeros((N, MAX_OPP_KEEPERS_TOTAL), dtype=np.float32)
        goal_embeds = np.zeros((N, MAX_GOALS_IN_PLAY, CARD_EMBED_DIM), dtype=np.float32)
        goal_mask = np.zeros((N, MAX_GOALS_IN_PLAY), dtype=np.float32)
        rules_embeds = np.zeros((N, MAX_RULES_IN_PLAY, CARD_EMBED_DIM), dtype=np.float32)
        rules_mask = np.zeros((N, MAX_RULES_IN_PLAY), dtype=np.float32)
        draw_pile_size = np.empty((N, 1), dtype=np.float32)
        opponent_hand_size = np.empty((N, 1), dtype=np.float32)
        action_mask = np.empty((N, self.action_dim), dtype=np.bool_)
        hand_size = np.empty((N, 1), dtype=np.float32)
        discard_pile_size = np.empty((N, 1), dtype=np.float32)
        own_keepers_in_play_count = np.empty((N, 1), dtype=np.float32)
        opponent_keepers_in_play_count = np.empty((N, 1), dtype=np.float32)
        goals_in_play_count = np.empty((N, 1), dtype=np.float32)
        rules_in_play_count = np.empty((N, 1), dtype=np.float32)

        def fill_row(embeds_arr, mask_arr, row_idx, names, max_size, label):
            n = len(names)
            if n > max_size:
                raise ValueError(
                    f"{label} has {n} cards, exceeds MAX={max_size}. Bump the cap."
                )
            for j, name in enumerate(names):
                embeds_arr[row_idx, j] = embedding_table[name]
                mask_arr[row_idx, j] = 1.0

        for i, entry in enumerate(entries):
            decision_context[i] = entry.decision_context
            fill_row(hand_embeds, hand_mask, i, entry.hand, MAX_HAND_SIZE, "hand")
            fill_row(discard_embeds, discard_mask, i, entry.discard, MAX_DISCARD_SIZE, "discard")
            fill_row(own_keeper_embeds, own_keeper_mask, i, entry.own_keepers, MAX_KEEPERS_PER_PLAYER, "own_keepers")
            fill_row(opp_keeper_embeds, opp_keeper_mask, i, entry.opp_keepers, MAX_OPP_KEEPERS_TOTAL, "opp_keepers")
            fill_row(goal_embeds, goal_mask, i, entry.goals, MAX_GOALS_IN_PLAY, "goals")
            fill_row(rules_embeds, rules_mask, i, entry.rules, MAX_RULES_IN_PLAY, "rules")
            draw_pile_size[i, 0] = entry.draw_pile_size
            opponent_hand_size[i, 0] = entry.opponent_hand_size
            hand_size[i, 0] = entry.hand_size
            discard_pile_size[i, 0] = entry.discard_pile_size
            own_keepers_in_play_count[i, 0] = entry.own_keepers_in_play_count
            opponent_keepers_in_play_count[i, 0] = entry.opponent_keepers_in_play_count
            goals_in_play_count[i, 0] = entry.goals_in_play_count
            rules_in_play_count[i, 0] = entry.rules_in_play_count

            action_mask[i] = entry.action_mask.astype(bool)

        return {
            "decision_context": torch.from_numpy(decision_context).to(device),
            "hand_embeds": torch.from_numpy(hand_embeds).to(device),
            "hand_mask": torch.from_numpy(hand_mask).to(device),
            "discard_embeds": torch.from_numpy(discard_embeds).to(device),
            "discard_mask": torch.from_numpy(discard_mask).to(device),
            "own_keeper_embeds": torch.from_numpy(own_keeper_embeds).to(device),
            "own_keeper_mask": torch.from_numpy(own_keeper_mask).to(device),
            "opp_keeper_embeds": torch.from_numpy(opp_keeper_embeds).to(device),
            "opp_keeper_mask": torch.from_numpy(opp_keeper_mask).to(device),
            "goal_embeds": torch.from_numpy(goal_embeds).to(device),
            "goal_mask": torch.from_numpy(goal_mask).to(device),
            "rules_embeds": torch.from_numpy(rules_embeds).to(device),
            "rules_mask": torch.from_numpy(rules_mask).to(device),
            "draw_pile_size": torch.from_numpy(draw_pile_size).to(device),
            "opponent_hand_size": torch.from_numpy(opponent_hand_size).to(device),
            "action_mask": torch.from_numpy(action_mask).to(device),
            "hand_size": torch.from_numpy(hand_size).to(device),
            "discard_pile_size": torch.from_numpy(discard_pile_size).to(device),
            "own_keepers_in_play_count": torch.from_numpy(own_keepers_in_play_count).to(device),
            "opponent_keepers_in_play_count": torch.from_numpy(opponent_keepers_in_play_count).to(device),
            "goals_in_play_count": torch.from_numpy(goals_in_play_count).to(device),
            "rules_in_play_count": torch.from_numpy(rules_in_play_count).to(device),
        }

    def act(
        self,
        game_state: "GameState",
        epsilon: float = 0.0,
    ) -> tuple[int, torch.Tensor, BufferEntry]:
        device = next(self.q_network.parameters()).device

        entry = self.extract_entry(game_state)
        obs_dict = self.collate([entry], device)  # batch of 1

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
        return self.extract_entry(game_state)