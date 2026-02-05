from typing import TYPE_CHECKING

import random

# avoiding circular import
from fluxx.Card import CardType
if TYPE_CHECKING:
    from fluxx import Game

from fluxx.utils.card_effect_utils import *

def can_use_free_action(free_action_name: str, game_state: 'Game', player_number: int) -> bool:
    player = game_state.players[player_number]

    if free_action_name == "mystery_play":
        return "mystery_play" not in game_state.played_free_actions

    if free_action_name == "swap_plays_for_draws":
        return "swap_plays_for_draws" not in game_state.played_free_actions

    if free_action_name == "goal_mill":
        goal_cards_in_hand = [card for card in player.hand if card.card_type.name == "GOAL"] # TODO: figure out how to get this enum to work properly
        return "goal_mill" not in game_state.played_free_actions and len(goal_cards_in_hand) > 0

    if free_action_name == "get_on_with_it":
        plays_allowed = game_state.get_play_rules(player_number)
        can_play_more = (player.cards_played < plays_allowed or game_state.rule_in_play("play_all"))
        return "get_on_with_it" not in game_state.played_free_actions and can_play_more and len(player.hand) > 0

    if free_action_name == "recycling":
        return "recycling" not in game_state.played_free_actions and len(player.keepers) > 0

    raise Exception("Invalid free action")

def activate_free_action(free_action_name: str, game_state: 'Game', user_number: int):
    user = game_state.players[user_number]

    if free_action_name == "mystery_play":
        card = game_state.get_card_from_draw_pile()
        game_state.game_message(f"<< Played {card.name}! >>")
        game_state.activate_card(user_number, card)

    elif free_action_name == "swap_plays_for_draws":
        plays_allowed = game_state.get_play_rules(user_number)
        cards_played = user.cards_played

        draw_amount = max(plays_allowed - cards_played, 0)

        if game_state.rule_in_play("play_all"):
            draw_amount = len(user.hand)

        for i in range(draw_amount):
            game_state.draw(user)

        game_state.force_turn_over = True

    elif free_action_name == "goal_mill":
        cards_to_draw = 0
        while True:
            goal_cards_in_hand = [card for card in user.hand if card.card_type.name == "GOAL"]
            if len(goal_cards_in_hand) == 0:
                break
            discard_card = game_state.agents[user_number].choose_to_discard(game_state)
            if discard_card == 0:
                break
            selected_goal_location = select_card(game_state, user_number, ["hand"], [CardType.RULE, CardType.KEEPER, CardType.ACTION])
            trash_selected_card(game_state, user_number, selected_goal_location, True)
            cards_to_draw += 1

        for i in range(cards_to_draw):
            game_state.draw(user)

    elif free_action_name == "get_on_with_it":
        while len(user.hand) > 0:
            game_state.discard_pile.append(user.hand.pop())

        for i in range(3 + game_state.inflation()):
            game_state.draw(user)

        game_state.force_turn_over = True

    elif free_action_name == "recycling":
        selected_keeper_location = select_card(game_state, user_number, ["own_keepers"])
        trash_selected_card(game_state, user_number, selected_keeper_location, True)

        for i in range(3 + game_state.inflation()):
            game_state.draw(user)

    else:
        raise Exception("Invalid free action")

