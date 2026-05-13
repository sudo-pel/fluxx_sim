import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from src.agents.utils import agent_utils
from src.agents.Agent import Agent
from src.agents.HeuristicAgentMKII import GameplanSortingOptions
from src.agents.utils.agent_utils import rule_options, card_type, is_play_rule, is_draw_rule, is_hand_limit_rule, is_keeper_limit_rule
from src.agents.utils.generalized_agent_utils import Gameplan
from src.game.cards.card_data import CARD_DATA
from src.game.FluxxEnums import GameConfig, GameState, GamePhaseType

ASYMMETRIC_TURN_EXTENDERS = {
    "draw_2_and_use_em",
    "draw_3_play_2_of_them",
    "take_another_turn",
    "todays_special",
    "lets_do_that_again"
}

class HeuristicAgentMKI(Agent):
    keeper_to_goal: dict[str, list[str]] = defaultdict(list[str])
    for card in CARD_DATA:
        if CARD_DATA[card]["card_type"] == "GOAL":
            for keeper in CARD_DATA[card]["required_keepers"]:
                keeper_to_goal[keeper].append(card)

    def eval_play(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, self.player_number)
        opponent_gameplans, card_to_opponent_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(state.hands[1 - self.player_number], state, 1 - self.player_number, hand_visible=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        plays_remaining = state.plays_remaining[self.player_number]
        cards_drawn = state.cards_drawn[self.player_number]

        # Check whether there is an immediate route to victory
        for gameplan in gameplans:
            if gameplan.missing_count == 0 and gameplan.in_hand_count <= plays_remaining:
                return { 100: {card for card in gameplan.required_cards if (card in cards_to_eval)} }

        current_goal_opponent_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(state.goals, state, 1 - self.player_number)[0]
        current_goal_opponent_gameplan = current_goal_opponent_gameplan[0] if len(current_goal_opponent_gameplan) > 0 else None
        for card in cards_to_eval:
            # used for one of the heuristics below
            if card in card_to_opponent_gameplan:
                opponent_gameplan = card_to_opponent_gameplan[card][0]
                goal_mod = 1 if opponent_gameplan.goal_in_play else 0

            if card_type(card) == "GOAL" and current_goal_opponent_gameplan is not None and current_goal_opponent_gameplan.in_play_count > 1:
                priorities[6].add(card)
            elif card in ASYMMETRIC_TURN_EXTENDERS:
                priorities[5].add(card)
            elif is_play_rule(card):
                if rule_options(card)["play"] > plays_remaining + 1:
                    priorities[4].add(card)
                else:
                    priorities[-1].add(card)
            elif is_draw_rule(card) and rule_options(card)["draw"] > cards_drawn + 1:
                priorities[3].add(card)
            elif card_type(card) == "GOAL" and card_to_gameplan[card][0].in_play_count > 1 and card_to_gameplan[card][0].in_discard_count > 0:
                priorities[-2].add(card)
            elif card in card_to_opponent_gameplan and opponent_gameplan.missing_cards == {card}:
                priorities[-100].add(card)
            elif card in card_to_opponent_gameplan and opponent_gameplan.in_play_count > goal_mod and opponent_gameplan.in_discard_count == 0:
                priorities[-5].add(card)
            else:
                priorities[0].add(card)

        return priorities

    def eval_discard(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, self.player_number)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            if card in card_to_gameplan:
                gameplan = card_to_gameplan[card][0]
                if gameplan.in_discard_count > 0:
                    priorities[1].add(card)
                else:
                    priorities[-5].add(card)
            else:
                priorities[0].add(card)

        return priorities

    def eval_discard_rule_in_play(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            if is_play_rule(card):
                if rule_options(card)["play"] == state.cards_played[self.player_number]:
                    priorities[1].add(card)
                else:
                    priorities[-1].add(card)
            elif is_draw_rule(card):
                priorities[1].add(card)
            elif is_keeper_limit_rule(card):
                if len(state.keepers[self.player_number]) > rule_options(card)["keeper_limit"]:
                    priorities[1].add(card)
                elif len(state.keepers[1 - self.player_number]) > rule_options(card)["keeper_limit"]:
                    priorities[-1].add(card)
                else:
                    priorities[0].add(card)
            elif is_hand_limit_rule(card):
                if len(state.hands[self.player_number]) - state.plays_remaining[self.player_number] > rule_options(card)["hand_limit"]:
                    priorities[1].add(card)
                elif len(state.hands[1 - self.player_number]) > rule_options(card)["hand_limit"]:
                    priorities[-1].add(card)
                else:
                    priorities[0].add(card)
            else:
                priorities[0].add(card)

        return priorities

    def eval_add_card_in_play_to_hand(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, self.player_number)
        opponent_gameplans, card_to_opponent_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, 1 - self.player_number, hand_visible=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            if card in card_to_gameplan:
                gameplan = card_to_gameplan[card][0]
            if card in card_to_opponent_gameplan:
                opponent_gameplan = card_to_opponent_gameplan[card]
            if card in card_to_gameplan and (gameplan.missing_count > 0 or gameplan.in_hand_count > state.plays_remaining[self.player_number]):
                priorities[5].add(card)
            elif card in card_to_opponent_gameplan and card_to_opponent_gameplan[card][0].held_count > 0:
                priorities[5].add(card)
            else:
                priorities[0].add(card)

        return priorities

    def eval_share_cards_from_latent_space(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, self.player_number)
        opponent_gameplans, card_to_opponent_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, 1 - self.player_number, hand_visible=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            if card in card_to_gameplan:
                priorities[5].add(card)
            elif card in card_to_opponent_gameplan:
                priorities[5].add(card)
            else:
                priorities[0].add(card)

        return priorities

    def eval_give_keeper_to_opponent(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, self.player_number)
        opponent_gameplans, card_to_opponent_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, 1 - self.player_number, hand_visible=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        for card in cards_to_eval:
            if card in card_to_gameplan:
                priorities[-5].add(card)
            elif card in card_to_opponent_gameplan:
                priorities[-5].add(card)
            else:
                priorities[0].add(card)

        return priorities

    game_phase_to_eval_function = {
        GamePhaseType.PLAY_CARD_FOR_TURN: eval_play,
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE: eval_play,
        GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE: eval_play,
        GamePhaseType.ACTIVATE_FREE_ACTION: eval_play,

        GamePhaseType.DISCARD_CARD_FROM_HAND: eval_discard,
        GamePhaseType.DISCARD_KEEPER: eval_discard,
        GamePhaseType.DISCARD_KEEPER_IN_PLAY: eval_discard,
        GamePhaseType.DISCARD_OWN_KEEPER_IN_PLAY: eval_discard,
        GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND: eval_discard,
        GamePhaseType.DISCARD_GOAL_IN_PLAY: eval_discard,

        GamePhaseType.DISCARD_RULE_IN_PLAY: eval_discard_rule_in_play,

        GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND: eval_add_card_in_play_to_hand,

        GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND: eval_share_cards_from_latent_space,
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT: eval_share_cards_from_latent_space,

        GamePhaseType.SELECT_KEEPER_TO_STEAL: eval_share_cards_from_latent_space,
        GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE: eval_share_cards_from_latent_space,

        GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE: eval_give_keeper_to_opponent,
    }



    def __init__(self, game_config: GameConfig, player_number: int):
        self.game_config = game_config
        self.player_number = player_number

    @staticmethod
    def get_gameplans_from_cards(cards: list[str], game_state: GameState, player_number: int, hand_visible: bool = True, sort_by: Optional[GameplanSortingOptions] = None, reverse: bool = False) -> tuple[list[Gameplan], dict[str, list[Gameplan]]]:
        """
        Takes a list of cards (strings)

        Returns:
            list of Gameplans that can be achieved with any subset of those cards
            dict mapping each card to a list of Gameplans that can be achieved with that card

        Lists of Gameplans (within both return values) are sorted by in ascending order of missing cards (so the gameplans closest to fruition are first)
        """

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
                goal_names = HeuristicAgentMKI.keeper_to_goal[card]

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

    def act(self, state: GameState):
        current_phase = state.stack[-1]
        action_mask = agent_utils.observe_hot_encoded(self, state, self.game_config)["action_mask"]
        cards_to_choose_from = [self.game_config.card_list[i] for i in range(len(self.game_config.card_list)) if action_mask[i] == 1]

        if current_phase.type not in HeuristicAgentMKI.game_phase_to_eval_function:
            priorities: dict[int, set[str]] = {0: {c for c in cards_to_choose_from}}
        else:
            priorities: dict[int, set[str]] = HeuristicAgentMKI.game_phase_to_eval_function[current_phase.type](self, state, cards_to_choose_from)

        # adding "no free action" if masked in
        if action_mask[-1] == 1:
            priorities[0].add("no_free_action")

        if len(priorities) == 0:
            raise Exception("No cards to choose from")

        max_priority = max(priorities.keys())
        choice = random.choice(list(priorities[max_priority]))

        if choice == "no_free_action":
            return len(action_mask)-1, [], None
        else:
            return self.game_config.card_list.index(choice), [], None


