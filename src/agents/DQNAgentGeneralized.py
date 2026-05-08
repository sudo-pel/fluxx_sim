from __future__ import annotations

import numpy as np
import torch

from src.agents.Agent import Agent
from src.agents.agent_utils import (
    convert_decision_encoding,
    decision_context_vectors,
    populate_card_vector,
)
from src.agents.card_embeddings import (
    CARD_EMBED_DIM,
    get_embedding_table,
    get_embedding_tensor,  # CHANGED: new import for GPU-side gather
)
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

        self.card_to_action_index = {c: i for i, c in enumerate(game_config.card_list)}
        self.card_to_embed_id = {c: i + 1 for i, c in enumerate(game_config.card_list)}

        self.card_to_index = self.card_to_action_index

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def _names_to_embed_ids(self, names: list[str]) -> np.ndarray:
        if not names:
            return np.empty(0, dtype=np.int32)
        embed_id_map = self.card_to_embed_id
        return np.fromiter(
            (embed_id_map[n] for n in names),
            dtype=np.int32,
            count=len(names),
        )

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
            hand_ids=self._names_to_embed_ids(hand),
            discard_ids=self._names_to_embed_ids(discard),
            own_keeper_ids=self._names_to_embed_ids(own_keepers),
            opp_keeper_ids=self._names_to_embed_ids(opp_keepers),
            goal_ids=self._names_to_embed_ids(goals),
            rules_ids=self._names_to_embed_ids(rules),
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
        if current_phase.type == GamePhaseType.GAME_OVER:
            action_mask = np.zeros(len(self.game_config.card_list) + 1, dtype=np.int8)
            action_mask[-1] = 1 # want to avoid strange behaviour if no actions are legal in a state
            return action_mask

        cards_in_hand_vector = populate_card_vector(self.game_config.card_list, game_state.hands[self.player_number])
        keeper_vectors = [populate_card_vector(self.game_config.card_list, kl) for kl in game_state.keepers]
        own_keeper_vector = keeper_vectors[self.player_number]
        other_keeper_vectors = (
            keeper_vectors[: self.player_number]
            + keeper_vectors[self.player_number + 1 :]
        )
        rules_vector = populate_card_vector(self.game_config.card_list, game_state.rules)
        goals_vector = populate_card_vector(self.game_config.card_list, game_state.goals)

        action_mask = np.zeros(len(self.game_config.card_list), dtype=np.int8)
        no_free_action_legal = False

        current_phase_type = current_phase.type
        if current_phase_type == GamePhaseType.PLAY_CARD_FOR_TURN:
            action_mask = cards_in_hand_vector
        elif current_phase_type == GamePhaseType.DISCARD_CARD_FROM_HAND:
            action_mask = cards_in_hand_vector
        elif current_phase_type == GamePhaseType.DISCARD_KEEPER:
            action_mask = own_keeper_vector
        elif current_phase_type == GamePhaseType.DISCARD_RULE_IN_PLAY:
            action_mask = rules_vector
        elif current_phase_type == GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE:
            action_mask = populate_card_vector(self.game_config.card_list, [c.name for c in current_phase.latent_space])
        elif current_phase_type == GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND:
            action_mask = np.bitwise_or.reduce((*keeper_vectors, rules_vector, goals_vector))
        elif current_phase_type == GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND:
            action_mask = populate_card_vector(self.game_config.card_list, [c.name for c in current_phase.latent_space])
        elif current_phase_type == GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE:
            action_mask = populate_card_vector(
                self.game_config.card_list,
                [
                    c for c in game_state.discard_pile
                    if CARD_DATA[c]["card_type"] in ("RULE", "ACTION")
                ]
            )
        elif current_phase_type == GamePhaseType.DISCARD_KEEPER_IN_PLAY:
            action_mask = np.bitwise_or.reduce(keeper_vectors)
        elif current_phase_type == GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT:
            action_mask = populate_card_vector(self.game_config.card_list, [c.name for c in current_phase.latent_space])
        elif current_phase_type == GamePhaseType.SELECT_KEEPER_TO_STEAL:
            action_mask = np.bitwise_or.reduce(other_keeper_vectors)
        elif current_phase_type == GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE:
            action_mask = np.bitwise_or.reduce(other_keeper_vectors)
        elif current_phase_type == GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE:
            action_mask = own_keeper_vector.copy()
            action_mask[self.card_to_index[current_phase.labelled_card.name]] = 0
        elif current_phase_type == GamePhaseType.ACTIVATE_FREE_ACTION:
            action_mask = populate_card_vector(self.game_config.card_list, list(game_state.available_free_actions))
            no_free_action_legal = True
        elif current_phase_type == GamePhaseType.DISCARD_OWN_KEEPER_IN_PLAY:
            action_mask = own_keeper_vector
        elif current_phase_type == GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND:
            valid = [
                c for c in game_state.hands[self.player_number]
                if CARD_DATA[c]["card_type"] in [t.name for t in current_phase.card_types]
            ]
            action_mask = populate_card_vector(self.game_config.card_list, valid)
            no_free_action_legal = True
        elif current_phase_type == GamePhaseType.DISCARD_GOAL_IN_PLAY:
            action_mask = goals_vector
        elif current_phase.type == GamePhaseType.PLAY_GOAL_FROM_DISCARD_PILE:
            valid_cards_in_discard_pile = []
            for card in game_state.discard_pile:
                if CARD_DATA[card]["card_type"] == "GOAL":
                    valid_cards_in_discard_pile.append(card)
            action_mask = populate_card_vector(self.game_config.card_list, valid_cards_in_discard_pile)
        elif current_phase.type == GamePhaseType.SELECT_KEEPER_FROM_DISCARD_PILE:
            valid_cards_in_discard_pile = []
            for card in game_state.discard_pile:
                if CARD_DATA[card]["card_type"] == "KEEPER":
                    valid_cards_in_discard_pile.append(card)
            action_mask = populate_card_vector(self.game_config.card_list, valid_cards_in_discard_pile)
        elif current_phase.type == GamePhaseType.GIVE_KEEPER_TO_OPPONENT:
            action_mask = own_keeper_vector
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

        embedding_tensor = get_embedding_tensor(device)  # (N_cards + 1, CARD_EMBED_DIM) on `device`

        hand_ids = np.zeros((N, MAX_HAND_SIZE), dtype=np.int32)
        hand_mask = np.zeros((N, MAX_HAND_SIZE), dtype=np.float32)
        discard_ids = np.zeros((N, MAX_DISCARD_SIZE), dtype=np.int32)
        discard_mask = np.zeros((N, MAX_DISCARD_SIZE), dtype=np.float32)
        own_keeper_ids = np.zeros((N, MAX_KEEPERS_PER_PLAYER), dtype=np.int32)
        own_keeper_mask = np.zeros((N, MAX_KEEPERS_PER_PLAYER), dtype=np.float32)
        opp_keeper_ids = np.zeros((N, MAX_OPP_KEEPERS_TOTAL), dtype=np.int32)
        opp_keeper_mask = np.zeros((N, MAX_OPP_KEEPERS_TOTAL), dtype=np.float32)
        goal_ids = np.zeros((N, MAX_GOALS_IN_PLAY), dtype=np.int32)
        goal_mask = np.zeros((N, MAX_GOALS_IN_PLAY), dtype=np.float32)
        rules_ids = np.zeros((N, MAX_RULES_IN_PLAY), dtype=np.int32)
        rules_mask = np.zeros((N, MAX_RULES_IN_PLAY), dtype=np.float32)

        scalars = np.empty((N, 8), dtype=np.float32)

        decision_context = np.empty((N, 19), dtype=np.float32)
        action_mask_np = np.empty((N, self.action_dim), dtype=np.bool_)

        def pad_row(ids_arr, mask_arr, row_idx, ids, max_size, label):
            n = ids.shape[0]
            if n > max_size:
                raise ValueError(
                    f"{label} has {n} cards, exceeds MAX={max_size}. Bump the cap."
                )
            if n > 0:
                ids_arr[row_idx, :n] = ids
                mask_arr[row_idx, :n] = 1.0

        for i, entry in enumerate(entries):
            decision_context[i] = entry.decision_context

            pad_row(hand_ids, hand_mask, i, entry.hand_ids, MAX_HAND_SIZE, "hand")
            pad_row(discard_ids, discard_mask, i, entry.discard_ids, MAX_DISCARD_SIZE, "discard")
            pad_row(own_keeper_ids, own_keeper_mask, i, entry.own_keeper_ids, MAX_KEEPERS_PER_PLAYER, "own_keepers")
            pad_row(opp_keeper_ids, opp_keeper_mask, i, entry.opp_keeper_ids, MAX_OPP_KEEPERS_TOTAL, "opp_keepers")
            pad_row(goal_ids, goal_mask, i, entry.goal_ids, MAX_GOALS_IN_PLAY, "goals")
            pad_row(rules_ids, rules_mask, i, entry.rules_ids, MAX_RULES_IN_PLAY, "rules")

            scalars[i, 0] = entry.draw_pile_size
            scalars[i, 1] = entry.opponent_hand_size
            scalars[i, 2] = entry.hand_size
            scalars[i, 3] = entry.discard_pile_size
            scalars[i, 4] = entry.own_keepers_in_play_count
            scalars[i, 5] = entry.opponent_keepers_in_play_count
            scalars[i, 6] = entry.goals_in_play_count
            scalars[i, 7] = entry.rules_in_play_count

            action_mask_np[i] = entry.action_mask.astype(bool)

        def to_dev(arr):
            return torch.from_numpy(arr).to(device, non_blocking=True)

        hand_ids_t = to_dev(hand_ids).long()
        discard_ids_t = to_dev(discard_ids).long()
        own_keeper_ids_t = to_dev(own_keeper_ids).long()
        opp_keeper_ids_t = to_dev(opp_keeper_ids).long()
        goal_ids_t = to_dev(goal_ids).long()
        rules_ids_t = to_dev(rules_ids).long()

        hand_embeds = embedding_tensor[hand_ids_t]
        discard_embeds = embedding_tensor[discard_ids_t]
        own_keeper_embeds = embedding_tensor[own_keeper_ids_t]
        opp_keeper_embeds = embedding_tensor[opp_keeper_ids_t]
        goal_embeds = embedding_tensor[goal_ids_t]
        rules_embeds = embedding_tensor[rules_ids_t]

        scalars_t = to_dev(scalars)

        return {
            "decision_context": to_dev(decision_context),
            "hand_embeds": hand_embeds,
            "hand_mask": to_dev(hand_mask),
            "discard_embeds": discard_embeds,
            "discard_mask": to_dev(discard_mask),
            "own_keeper_embeds": own_keeper_embeds,
            "own_keeper_mask": to_dev(own_keeper_mask),
            "opp_keeper_embeds": opp_keeper_embeds,
            "opp_keeper_mask": to_dev(opp_keeper_mask),
            "goal_embeds": goal_embeds,
            "goal_mask": to_dev(goal_mask),
            "rules_embeds": rules_embeds,
            "rules_mask": to_dev(rules_mask),
            "action_mask": to_dev(action_mask_np),
            "draw_pile_size": scalars_t[:, 0:1],
            "opponent_hand_size": scalars_t[:, 1:2],
            "hand_size": scalars_t[:, 2:3],
            "discard_pile_size": scalars_t[:, 3:4],
            "own_keepers_in_play_count": scalars_t[:, 4:5],
            "opponent_keepers_in_play_count": scalars_t[:, 5:6],
            "goals_in_play_count": scalars_t[:, 6:7],
            "rules_in_play_count": scalars_t[:, 7:8],
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