from typing import TYPE_CHECKING

# avoiding circular import
if TYPE_CHECKING:
    pass

from fluxx.game.utils.card_effect_utils import trash_selected_card, select_card
from fluxx.game import game_messages
from fluxx.game.FluxxEnums import CardType, CardZone, ExtendedCardZone

# ---------------------
# FREE ACTION USABILITY CHECKS
# ---------------------

def can_use_mystery_play(game_state: 'Game', player_number: int) -> bool:
    return "mystery_play" not in game_state.played_free_actions

def can_use_swap_plays_for_draws(game_state: 'Game', player_number: int) -> bool:
    return "swap_plays_for_draws" not in game_state.played_free_actions

def can_use_goal_mill(game_state: 'Game', player_number: int) -> bool:
    player = game_state.players[player_number]

    goal_cards_in_hand = [card for card in player.hand if card.card_type == CardType.GOAL]
    return "goal_mill" not in game_state.played_free_actions and len(goal_cards_in_hand) > 0

def can_use_get_on_with_it(game_state: 'Game', player_number: int) -> bool:
    player = game_state.players[player_number]
    can_play_more = (player.cards_played < game_state.get_play_rules(player_number) or game_state.rule_in_play("play_all"))
    return "get_on_with_it" not in game_state.played_free_actions and can_play_more and len(player.hand) > 0

def can_use_recycling(game_state: 'Game', player_number: int) -> bool:
    player = game_state.players[player_number]
    return "recycling" not in game_state.played_free_actions and len(player.keepers) > 0

# ---------------------
# FREE ACTION EFFECTS
# ---------------------

def activate_mystery_play(game_state: "Game", user_number: int):
    card = game_state.get_card_from_draw_pile()
    game_messages.special_effect(f"<< Played {card.name}! >>")
    game_state.activate_card(user_number, card)

def activate_swap_plays_for_draws(game_state: "Game", user_number: int):
    user = game_state.players[user_number]

    plays_allowed = game_state.get_play_rules(user_number)
    cards_played = user.cards_played

    draw_amount = max(plays_allowed - cards_played, 0)

    if game_state.rule_in_play("play_all"):
        draw_amount = len(user.hand)

    for i in range(draw_amount):
        game_state.draw(user)

    game_state.force_turn_over = True

def activate_goal_mill(game_state: "Game", user_number: int):
    user = game_state.players[user_number]

    game_state.force_turn_over = True
    cards_to_draw = 0
    while True:
        goal_cards_in_hand = [card for card in user.hand if card.card_type == CardType.GOAL]
        if len(goal_cards_in_hand) == 0:
            break
        discard_card = game_state.agents[user_number].choose_to_discard(game_state)
        if discard_card == 0:
            break
        selected_goal_location = select_card(game_state, user_number, [CardZone.HAND],
                                             [CardType.RULE, CardType.KEEPER, CardType.ACTION])
        trash_selected_card(game_state, user_number, selected_goal_location, True)
        cards_to_draw += 1

    for i in range(cards_to_draw):
        game_state.draw(user)

def activate_get_on_with_it(game_state: "Game", user_number: int):
    user = game_state.players[user_number]

    while len(user.hand) > 0:
        game_state.discard_pile.append(user.hand.pop())

    for i in range(3 + game_state.inflation()):
        game_state.draw(user)

    game_state.force_turn_over = True

def activate_recycling(game_state: "Game", user_number: int):
    user = game_state.players[user_number]

    selected_keeper_location = select_card(game_state, user_number, [ExtendedCardZone.OWN_KEEPERS])
    trash_selected_card(game_state, user_number, selected_keeper_location, True)

    for i in range(3 + game_state.inflation()):
        game_state.draw(user)


FREE_ACTIONS = {
    "mystery_play": {
        "can_use": can_use_mystery_play,
        "activate": activate_mystery_play
    },
    "swap_plays_for_draws": {
        "can_use": can_use_swap_plays_for_draws,
        "activate": activate_swap_plays_for_draws
    },
    "goal_mill": {
        "can_use": can_use_goal_mill,
        "activate": activate_goal_mill
    },
    "get_on_with_it": {
        "can_use": can_use_get_on_with_it,
        "activate": activate_get_on_with_it
    },
    "recycling": {
        "can_use": can_use_recycling,
        "activate": activate_recycling
    }
}


def can_use_free_action(game_state: 'Game', player_number: int, free_action_name: str) -> bool:
    free_action = FREE_ACTIONS.get(free_action_name)
    if free_action is None:
        raise Exception(f"Invalid free action: {free_action_name}")

    return free_action["can_use"](game_state, player_number)

def activate_free_action(game_state: 'Game', player_number: int, free_action_name: str):
    free_action = FREE_ACTIONS.get(free_action_name)
    if free_action is None:
        raise Exception(f"Invalid free action: {free_action_name}")

    free_action["activate"](game_state, player_number)



