"""
PPOAgent: PPO actor that takes a raw GameState, extracts a BufferEntry,
collates batches into tensor dicts for its policy network, and selects
actions.

Key methods:
    extract_entry(game_state) -> BufferEntry
        Pulls the agent-relevant fields out of a raw game state.
    collate(entries, device) -> dict[str, Tensor]
        Converts N entries into batched, padded tensor dict for the encoder.
    act(game_state) -> (action, log_prob, entry)
        Public rollout interface. Extracts -> collates batch-of-1 ->
        runs policy -> samples masked action.

The same collate is also used by PPO during minibatching: collect a list
of N BufferEntry objects, call self.actor.collate(entries, device), and
hand the resulting dict to the actor's forward.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Categorical

from src.agents import agent_utils
from src.agents.Agent import Agent
from src.agents.HeuristicAgentMKII import GameplanSortingOptions
from src.training.TrainingEnums import BufferEntry
from src.agents.card_embeddings import CARD_EMBED_DIM, get_embedding_table
from src.neural_networks.FluxxActorNetworkPPO import FluxxActorNetwork

from src.agents.agent_utils import convert_decision_encoding, decision_context_vectors, populate_card_vector
from src.game.FluxxEnums import GameState
from src.game.FluxxEnums import GameConfig
from src.game.FluxxEnums import GamePhaseType
from src.game.cards.card_data import CARD_DATA

MAX_HAND_SIZE = 100
MAX_DISCARD_SIZE = 100
MAX_KEEPERS_PER_PLAYER = 100
MAX_OPP_KEEPERS_TOTAL = 100
MAX_GOALS_IN_PLAY = 100
MAX_RULES_IN_PLAY = 100

@dataclass
class Gameplan:
    """
    goal: goal pertinent to gameplan

    required_cards: set of cards (including goal) required to win via gameplan
    held_cards: set of cards currently held by player, in hand or in play (relevant to goal)
    missing_cards: set of cards required to play goal but not currently held by player
    cards_in_hand: subset of held_cards that are in hand
    cards_in_play: subset of held_cards that are in play

    held_count, missing_count, in_hand_count, in_play_count: all self-explanatory
    goal_in_play: whether goal is in play
    """
    goal: str
    required_cards: set[str]
    held_cards: set[str]
    missing_cards: set[str]
    cards_in_hand: set[str]
    cards_in_play: set[str]
    cards_in_discard: set[str]
    held_count: int
    missing_count: int
    in_hand_count: int
    in_play_count: int
    in_discard_count: int
    goal_in_play: bool

class PPOAgentGeneralizedWithHeuristic(Agent):
    def __init__(
        self,
        game_config: GameConfig,
        player_number: int,
    ):
        super(PPOAgentGeneralizedWithHeuristic, self).__init__()
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

    def extract_entry(self, game_state: GameState) -> BufferEntry:
        """
        Converts a specific GameState into a BufferEntry.
        """
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
                if i != self.player_number)
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


    # TODO: consider making a separate agent_utils function
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

        # TODO: implement embedding table
        embedding_table = get_embedding_table()

        # Pre-allocate batched arrays
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

        # helper function, consider moving outside
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


        # Single transfer to device per tensor
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
            "rules_in_play_count": torch.from_numpy(rules_in_play_count).to(device)
        }

    def act(self, game_state: "GameState") -> tuple[int, torch.Tensor, BufferEntry]:
        """
        Returns:
            action: int (sampled action index)
            log_prob: 0-d torch.Tensor (log probability of the sampled action)
            entry: BufferEntry (to be stored in the replay buffer)
        """
        if game_state.stack[-1].type == GamePhaseType.PLAY_CARD_FOR_TURN:
            action_mask = agent_utils.observe_hot_encoded(self, game_state, self.game_config)["action_mask"]
            cards_to_choose_from = [self.game_config.card_list[i] for i in range(len(self.game_config.card_list)) if action_mask[i] == 1]

            gameplans, card_to_gameplan = PPOAgentGeneralizedWithHeuristic.get_gameplans_from_cards(cards_to_choose_from, game_state, self.player_number, sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False)

            plays_remaining = game_state.plays_remaining[self.player_number]
            cards_drawn = game_state.cards_drawn[self.player_number]

            # Check whether there is an immediate route to victory
            for gameplan in gameplans:
                if gameplan.missing_count == 0 and gameplan.in_hand_count <= plays_remaining:
                    cards_to_play = [card for card in gameplan.required_cards if (card in cards_to_choose_from)]
                    return self.game_config.card_list.index(cards_to_play[0]), torch.tensor(1.0), self.extract_entry(game_state)

        device = next(self.policy_network.parameters()).device

        entry = self.extract_entry(game_state)

        if not entry.action_mask.any():
            raise RuntimeError(
                f"All-zero action mask in phase {game_state.stack[-1].type}. "
                f"Hand size: {len(game_state.hands[self.player_number])}, "
                f"own keepers: {len(game_state.keepers[self.player_number])}"
            )

        obs_dict = self.collate([entry], device)   # batch of 1

        logits = self.policy_network(obs_dict)              # (1, action_dim)
        logits = logits.masked_fill(~obs_dict["action_mask"], float("-inf"))

        dist = Categorical(logits=logits)
        action = dist.sample()                              # (1,)
        log_prob = dist.log_prob(action)                    # (1,)

        return action.item(), log_prob.squeeze(0), entry

    @staticmethod
    def get_gameplans_from_cards(cards: list[str], game_state: GameState, player_number: int, hand_visible: bool = True, sort_by: Optional[GameplanSortingOptions] = None, reverse: bool = False) -> tuple[list[Gameplan], dict[str, list[Gameplan]]]:
        """
        Takes a list of cards (strings) and returns:
        - list of Gameplans that can be achieved with any subset of those cards
        - dict mapping each card to a list of Gameplans that can be achieved with that card

        Lists of Gameplans (within both return values) are sorted by in ascending order of missing cards (so the gameplans closest to fruition are first)
        """

        # TODO: unique goals like 5 keepers and special keeper requirements like disallowed keepers and optional keepers
        # When a goal is seen: generate a gameplan corresponding to that goal
        # When a keeper is seen: generate a gameplan corresponding to each goal that the keeper is pertinent to
        # Avoid generating duplicate gameplans by hashing via a tuple of the goal and all required keepers (which uniquely identifies a goal)
        gameplans: dict[tuple[str], Gameplan] = {}
        for card in cards:
            card_type = CARD_DATA[card]["card_type"]
            if card_type not in {"GOAL", "KEEPER"}:
                continue
            if card_type == "GOAL":
                goal_names = [card]
            else:
                goal_names = PPOAgentGeneralizedWithHeuristic.keeper_to_goal[card]

            for goal_name in goal_names:
                # is a tuple for the sake of hashing
                required_cards = tuple(CARD_DATA[goal_name]["required_keepers"] + [goal_name])
                required_cards_set = set(required_cards)

                if required_cards in gameplans:
                    continue

                cards_in_hand = {card for card in game_state.hands[player_number] if card in required_cards} if hand_visible else set()
                cards_in_play = {card for card in required_cards if (card in game_state.keepers[player_number]) or (card in game_state.goals)}

                held_cards = cards_in_hand | cards_in_play
                missing_cards = required_cards_set - held_cards
                held_count = len(held_cards)
                missing_count = len(missing_cards)
                in_hand_count = len(cards_in_hand)
                in_play_count = len(cards_in_play)
                goal_in_play = card in game_state.goals
                cards_in_discard = {card for card in game_state.discard_pile if (card in required_cards_set)}
                in_discard_count = len(cards_in_discard)

                # TODO: add support for goals with optional or disallowed keepers (must enrich Gameplan datatype)
                gameplan = Gameplan(
                    goal_name,
                    required_cards_set,
                    held_cards,
                    missing_cards,
                    cards_in_hand,
                    cards_in_play,
                    cards_in_discard,
                    held_count,
                    missing_count,
                    in_hand_count,
                    in_play_count,
                    in_discard_count,
                    goal_in_play
                )

                gameplans[required_cards] = gameplan

        card_to_gameplan: dict[str, list[Gameplan]] = defaultdict(list)
        for required_cards, gameplan in gameplans.items():
            for card in required_cards:
                card_to_gameplan[card].append(gameplan)

        gameplan_list = [g for g in gameplans.values()]
        gameplan_list.sort(key=lambda gameplan: gameplan.goal)
        if sort_by is not None:
            if sort_by == GameplanSortingOptions.MISSING_COUNT:
                gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.missing_count, reverse=reverse)
                for _, l in card_to_gameplan.items():
                    l.sort(key=lambda gameplan: gameplan.missing_count, reverse=reverse)
            elif sort_by == GameplanSortingOptions.HELD_COUNT:
                gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.held_count, reverse=reverse)
                for _, l in card_to_gameplan.items():
                    l.sort(key=lambda gameplan: gameplan.held_count, reverse=reverse)
            elif sort_by == GameplanSortingOptions.IN_HAND_COUNT:
                gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.in_hand_count, reverse=reverse)
                for _, l in card_to_gameplan.items():
                    l.sort(key=lambda gameplan: gameplan.in_hand_count)
            elif sort_by == GameplanSortingOptions.IN_PLAY_COUNT:
                gameplan_list = sorted(gameplan_list, key=lambda gameplan: gameplan.in_play_count, reverse=reverse)
                for _, l in card_to_gameplan.items():
                    l.sort(key=lambda gameplan: gameplan.in_play_count, reverse=reverse)

        return gameplan_list, card_to_gameplan
