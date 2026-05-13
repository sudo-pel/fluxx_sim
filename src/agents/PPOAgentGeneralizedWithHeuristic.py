from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical

from src.agents.utils import agent_utils, generalized_agent_utils
from src.agents.Agent import Agent
from src.agents.HeuristicAgentMKII import GameplanSortingOptions
from src.agents.utils.generalized_agent_utils import BufferEntry, Gameplan
from src.agents.utils.card_embeddings import CARD_EMBED_DIM, get_embedding_table
from src.neural_networks.FluxxActorNetworkPPO import FluxxActorNetwork

from src.agents.utils.agent_utils import convert_decision_encoding, decision_context_vectors, populate_card_vector
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

class PPOAgentGeneralizedWithHeuristic(Agent):
    keeper_to_goal: dict[str, list[str]] = defaultdict(list[str])
    for card in CARD_DATA:
        if CARD_DATA[card]["card_type"] == "GOAL":
            for keeper in CARD_DATA[card]["required_keepers"]:
                keeper_to_goal[keeper].append(card)

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

            gameplans, card_to_gameplan = PPOAgentGeneralizedWithHeuristic.get_gameplans_from_cards(
                cards_to_choose_from,
                game_state,
                self.player_number,
                sort_by=GameplanSortingOptions.MISSING_COUNT, reverse=False
            )

            plays_remaining = game_state.plays_remaining[self.player_number]

            # Check whether there is an immediate route to victory
            for gameplan in gameplans:
                if gameplan.missing_count == 0 and gameplan.in_hand_count <= plays_remaining:
                    cards_to_play = [card for card in gameplan.required_cards if (card in cards_to_choose_from)]
                    return (self.game_config.card_list.index(
                        cards_to_play[0]),
                        torch.tensor(1.0),
                        generalized_agent_utils.extract_entry(
                            game_state,
                            self.player_number,
                            self.game_config
                        )
                    )

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
