import random
from collections import defaultdict
from dataclasses import dataclass

from src.agents import agent_utils
from src.agents.Agent import Agent
from src.game.cards.card_data import CARD_DATA
from src.game.FluxxEnums import GameConfig, GameState, GamePhaseType
from src.game.utils.general_utils import print_game_state

"""

Implementation plan

- Function for checking whether a rule is a draw rule
- Function for checking whether a rule is a play rule


PLAY:
- Create a Gameplan object storing possible routes to victory. If any of the gameplans are "possible", promote that set of cards to the maximum, ensuring they come as a "set"
- Store a list of "asymmetric turn extender" cards
- Store a list of "symmetric turn extender" cards
- "Incomplete" gameplans should be discredited. 
- Calculate a list of goals that will cause your opponent to win the game and NEVER play them
- Similar logic for goals for which opponent has at least one keeper

DISCARD:
- Check what Gameplan the keeper/goal is a part of
    - If the gameplan has cards in the discard pile then increase the priority
    - If the gameplan has no cards in the discard pile then decrease the priority based on the number of gameplan cards missing (fewer missing = LOWER priority)
    
DISCARD RULE IN PLAY:
- Check whether rule is a "play", "draw", or "limit" rule and act accordingly

ADD CARD IN PLAY TO HAND:
- Map goals in play to gameplans and check whether the gameplan can be achieved this turn. If so, do NOT add the card in play to the hand
- ..If not, increase the priority inversely proportional to number of gameplan cards missing
- Also give priority to "limit" cards when you are over the limit

SHARE CARDS FROM LATENT SPACE:
- Map cards to gameplans and decrease priority based on number of gameplan cards missing
- Do the same for opponent, but with slightly smaller values
- Also prioritise asymmetric turn extender action cards, since they are good

STEAL CARD FROM OPPONENT:
- Map cards to gameplans and decrease priority based on number of gameplan cards missing
- Do the same for opponent, but with slightly smaller values

GIVE CARD TO OPPONENT:
- Inverse analysis to above

"""

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


ASYMMETRIC_TURN_EXTENDERS = {
    "draw_2_and_use_em",
    "draw_3_play_2_of_them",
    "take_another_turn",
    "todays_special",
    "lets_do_that_again" # TODO: note that this is not actually an asymmetric turn extender but may recur one
}


def rule_options(card_name: str) -> dict:
    """
    Return the RulesOptions dict for a rule card, or empty dict.
    """
    return CARD_DATA[card_name].get("RulesOptions", {})

def card_type(card_name: str) -> str:
    """
    Return the card_type string ('KEEPER', 'GOAL', 'RULE', 'ACTION').
    """
    return CARD_DATA[card_name]["card_type"]

def is_play_rule(card_name: str) -> bool:
    return card_type(card_name) == "RULE" and rule_options(card_name).get("play") is not None


def is_draw_rule(card_name: str) -> bool:
    return card_type(card_name) == "RULE" and rule_options(card_name).get("draw") is not None


def is_hand_limit_rule(card_name: str) -> bool:
    return card_type(card_name) == "RULE" and rule_options(card_name).get("hand_limit") is not None


def is_keeper_limit_rule(card_name: str) -> bool:
    return card_type(card_name) == "RULE" and rule_options(card_name).get("keeper_limit") is not None


def is_limit_rule(card_name: str) -> bool:
    return is_hand_limit_rule(card_name) or is_keeper_limit_rule(card_name)

class HeuristicAgentMKI(Agent):
    keeper_to_goal = {}
    for card in CARD_DATA:
        if CARD_DATA[card]["card_type"] == "GOAL":
            for keeper in CARD_DATA[card]["required_keepers"]:
                keeper_to_goal[keeper] = card

    # TODO: augment this function to take an argument of "additional cards", so that, for example, it can take into account that the card may be being ..
    # TODO: .. played from latent space (currently, card would not be in hand and so gameplan would appear incomplete even if every other card is in hand)
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
                opponent_gameplan = card_to_opponent_gameplan[card]
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
            elif card_type(card) == "GOAL" and card_to_gameplan[card].in_play_count > 1 and card_to_gameplan[card].in_discard_count > 0:
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
                gameplan = card_to_gameplan[card]
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
                gameplan = card_to_gameplan[card]
            if card in card_to_opponent_gameplan:
                opponent_gameplan = card_to_opponent_gameplan[card]
            if card in card_to_gameplan and (gameplan.missing_count > 0 or gameplan.in_hand_count > state.plays_remaining[self.player_number]):
                priorities[5].add(card)
            elif card in card_to_opponent_gameplan and card_to_opponent_gameplan[card].held_count > 0:
                priorities[5].add(card)
            else:
                priorities[0].add(card)

        return priorities

    def eval_share_cards_from_latent_space(self, state: GameState, cards_to_eval: list[str]) -> dict[int, set[str]]:
        gameplans, card_to_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, self.player_number)
        opponent_gameplans, card_to_opponent_gameplan = HeuristicAgentMKI.get_gameplans_from_cards(cards_to_eval, state, 1 - self.player_number, hand_visible=False)
        priorities: dict[int, set[str]] = defaultdict(set[str])

        # TODO: Prioritize specific gameplans by completion rate
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
            # TODO: (de-)prioritize specific gameplans by completion rate
            if card in card_to_gameplan:
                priorities[-5].add(card)
            elif card in card_to_opponent_gameplan:
                priorities[-5].add(card)
            else:
                priorities[0].add(card)

        return priorities

    # TODO: add support for "use free action" (may just leave without though)
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
    def get_gameplans_from_cards(cards: list[str], game_state: GameState, player_number: int, hand_visible: bool = True) -> tuple[list[Gameplan], dict[str, Gameplan]]:
        """
        Takes a list of cards and returns a list of Gameplans that can be achieved with any subset of those cards.
        """
        gameplans = []
        cards_seen = set()
        for card in cards:
            card_type = CARD_DATA[card]["card_type"]
            if card_type not in {"GOAL", "KEEPER"}:
                continue
            if card not in cards_seen:
                if card_type == "GOAL":
                    goal_name = card
                else:
                    goal_name = HeuristicAgentMKI.keeper_to_goal[card]
                required_cards = set(CARD_DATA[goal_name]["required_keepers"]) | {goal_name}
                cards_in_hand = {card for card in game_state.hands[player_number] if card in required_cards} if hand_visible else set()
                cards_in_play = {card for card in required_cards if (card in game_state.keepers[player_number]) or (card in game_state.goals)}

                held_cards = cards_in_hand | cards_in_play
                missing_cards = required_cards - held_cards
                held_count = len(held_cards)
                missing_count = len(missing_cards)
                in_hand_count = len(cards_in_hand)
                in_play_count = len(cards_in_play)
                goal_in_play = card in game_state.goals
                cards_in_discard = {card for card in game_state.discard_pile if (card in required_cards)}
                in_discard_count = len(cards_in_discard)

                # TODO: add support for goals with optional or disallowed keepers (must enrich Gameplan datatype)
                gameplan = Gameplan(
                    goal_name,
                    required_cards,
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

                cards_seen |= held_cards | missing_cards | {goal_name}
                gameplans.append(gameplan)

        card_to_gameplan = {}
        for i, gameplan in enumerate(gameplans):
            for card in gameplan.required_cards:
                card_to_gameplan[card] = gameplan

        return gameplans, card_to_gameplan

    def act(self, state: GameState):
        current_phase = state.stack[-1]
        action_mask = agent_utils.observe(self, state, self.game_config)["action_mask"]
        cards_to_choose_from = [self.game_config.card_list[i] for i in range(len(self.game_config.card_list)) if action_mask[i] == 1]
        priorities: dict[int, set[str]] = HeuristicAgentMKI.game_phase_to_eval_function[current_phase.type](self, state, cards_to_choose_from)

        # adding "no free action" if masked in
        if action_mask[-1] == 1:
            priorities[0].add("no_free_action")

        if len(priorities) == 0:
            print_game_state(state)
            raise Exception("No cards to choose from")

        max_priority = max(priorities.keys())
        choice = random.choice(list(priorities[max_priority]))

        if choice == "no_free_action":
            return len(action_mask)-1, [], None
        else:
            return self.game_config.card_list.index(choice), [], None


