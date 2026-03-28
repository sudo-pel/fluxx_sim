import math
from typing import TYPE_CHECKING

import random

# avoiding circular import
from fluxx.game.FluxxEnums import CardType, CardZone, ExtendedCardZone, GamePhase, GamePhaseType
from fluxx.game.GameSchema import GameSchema
from fluxx.game.game_messages import GameMessageType

if TYPE_CHECKING:
    pass

from fluxx.game.utils.card_effect_utils import trash_selected_card, get_selected_card, select_card
from fluxx.game import game_messages


def activate_use_what_you_take(game_state: 'GameSchema', user_number: int):
    # TODO: may need multiplayer refactoring

    other_player_number = user_number ^ 1
    other_player = game_state.players[other_player_number]

    if len(other_player.hand) == 0:
        return

    card_to_take = random.randint(0, len(other_player.hand) - 1)
    card_to_play = other_player.hand[card_to_take]
    del other_player.hand[card_to_take]
    game_state.game_message(f"<< (p{user_number}): ACTIVATING {card_to_play.name} >>", GameMessageType.SPECIAL_EFFECT)
    game_state.activate_card(user_number, card_to_play)


def activate_zap_a_card(game_state: 'GameSchema', user_number: int):
    if len(game_state.get_cards_in_play_by_name()) == 0:
        return
    game_state.stack.append(
        GamePhase(GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND, user_number, decisions_left=1)
    )


def activate_trash_a_new_rule(game_state: 'GameSchema', user_number: int):
    if len(game_state.rules) == 0:
        return
    game_state.stack.append(GamePhase(GamePhaseType.DISCARD_RULE_IN_PLAY, user_number, decisions_left=1))


def activate_trash_a_keeper(game_state: 'GameSchema', user_number: int):
    if len([k for player in game_state.get_all_keepers_by_name_flat() for k in player]) == 0:
        game_state.game_message("No keepers to trash!", GameMessageType.NOTIFICATION)
        return
    game_state.stack.append(GamePhase(GamePhaseType.DISCARD_KEEPER_IN_PLAY, user_number, decisions_left=1))


def activate_trade_hands(game_state: 'GameSchema', user_number: int):
    # TODO: if multiplayer implemented, need player selection here
    user_player = game_state.players[user_number]

    temp = game_state.players[user_number ^ 1].hand
    game_state.players[user_number ^ 1].hand= user_player.hand
    user_player.hand = temp

    game_state.game_message(f"<< Player {user_number} traded hands with player {user_number ^ 1} >> ", GameMessageType.SPECIAL_EFFECT)


def activate_todays_special(game_state: 'GameSchema', user_number: int):
    latent_space = [game_state.get_card_from_draw_pile() for i in range(3+game_state.inflation())]
    latent_space = [l for l in latent_space if l is not None]
    if len(latent_space) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE,
        user_number,
        decisions_left=1+game_state.inflation(),
        latent_space=latent_space
    ))


def activate_draw_2_and_use_em(game_state: 'GameSchema', user_number: int):
    latent_space = [game_state.get_card_from_draw_pile() for i in range(2+game_state.inflation())]
    latent_space = [l for l in latent_space if l is not None]
    if len(latent_space) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE,
        user_number,
        decisions_left=2+game_state.inflation(),
        latent_space=latent_space
    ))


def activate_draw_3_play_2_of_them(game_state: 'GameSchema', user_number: int):
    latent_space = [game_state.get_card_from_draw_pile() for i in range(3 + game_state.inflation())]
    latent_space = [l for l in latent_space if l is not None]
    if len(latent_space) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE,
        user_number,
        decisions_left=2 + game_state.inflation(),
        latent_space=latent_space
    ))

def activate_steal_a_keeper(game_state: 'GameSchema', user_number: int):
    if len(game_state.players[user_number ^ 1].keepers) == 0:
        return
    game_state.stack.append(
        GamePhase(GamePhaseType.SELECT_KEEPER_TO_STEAL, user_number, decisions_left=1)
    )


def activate_share_the_wealth(game_state: 'GameSchema', user_number: int):
    all_keepers = [keeper for player in game_state.players for keeper in player.keepers]
    for player in game_state.players:
        player.keepers = []

    if len(all_keepers) == 0:
        return

    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT,
        user_number,
        decisions_left = int(math.ceil(len(all_keepers) / 2)),
        latent_space = all_keepers
    ))


def activate_rules_reset(game_state: 'GameSchema', user_number: int):
    for rule in game_state.rules:
        game_state.discard_pile.append(rule)

    game_state.rules = []


def activate_rock_paper_scissors_showdown(game_state: 'GameSchema', user_number: int):
    user_player = game_state.players[user_number]

    selected_player_number = user_number ^ 1
    selected_player = game_state.players[selected_player_number]

    coin = random.randint(0, 1)
    if coin == 0:
        winner = user_player
        loser = selected_player
    else:
        winner = selected_player
        loser = user_player

    game_state.game_message(f"<< Player {winner.id} defeated {loser.id} in RPS! >>", GameMessageType.SPECIAL_EFFECT)

    winner.hand += loser.hand
    loser.hand = []


def activate_random_tax(game_state: 'GameSchema', user_number: int):
    user_player = game_state.players[user_number]

    for _ in range(game_state.inflation() + 1):
        for player_number, player in enumerate(game_state.players):
            if len(player.hand) == 0 or player_number == user_number:
                continue

            index_to_take = random.randint(0, len(player.hand) - 1)
            card_to_take = player.hand[index_to_take]

            game_state.game_message(f"<< Player {player_number} took {card_to_take.name} from their hand >>", GameMessageType.SPECIAL_EFFECT)

            user_player.hand.append(card_to_take)
            del player.hand[index_to_take]


def activate_no_limits(game_state: 'GameSchema', user_number: int):
    new_rules = []
    for rule in game_state.rules:
        if rule.keeper_limit is None and rule.hand_limit is None:
            new_rules.append(rule)
        else:
            game_state.discard_pile.append(rule)

    game_state.rules = new_rules


def activate_lets_simplify(game_state: 'GameSchema', user_number: int):
    to_discard = (len(game_state.rules) + 1) // 2
    for i in range(to_discard):
        game_state.stack.append(GamePhase(GamePhaseType.DISCARD_RULE_IN_PLAY, user_number, decisions_left=to_discard - i))


def activate_lets_do_that_again(game_state: 'GameSchema', user_number: int):
    if len([card for card in game_state.discard_pile if card.card_type == CardType.ACTION or card.card_type == CardType.RULE]) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE,
        user_number,
        decisions_left=1
    ))


def activate_jackpot(game_state: 'GameSchema', user_number: int):
    user_player = game_state.players[user_number]

    for _ in range(3 + game_state.inflation()):
        game_state.draw(user_player)


def activate_exchange_keepers(game_state: 'GameSchema', user_number: int):
    if len(game_state.players[user_number].keepers) == 0 or len(game_state.players[user_number ^ 1].keepers) == 0:
        return
    game_state.stack.append(GamePhase(GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE, user_number, decisions_left=1))


def activate_empty_the_trash(game_state: 'GameSchema', user_number: int):
    game_state.draw_pile += game_state.discard_pile
    discard_pile_size = len(game_state.discard_pile)
    game_state.discard_pile = []
    random.shuffle(game_state.draw_pile)
    game_state.game_message(f"<< Shuffled {discard_pile_size} cards from discard pile into deck! >>", GameMessageType.SPECIAL_EFFECT)


def activate_discard_and_draw(game_state: 'GameSchema', user_number: int):
    user_player = game_state.players[user_number]

    draw_amount = len(user_player.hand)

    game_state.discard_pile.extend(user_player.hand)
    user_player.hand = []

    for _ in range(draw_amount):
        game_state.draw(user_player)

def activate_everybody_gets_1(game_state: 'GameSchema', user_number: int):
    # TODO: Will need to refactor this for multiplayer if project reaches that point
    latent_space = [game_state.get_card_from_draw_pile() for i in range((1+game_state.inflation()) * len(game_state.players))]
    latent_space = [l for l in latent_space if l is not None]
    if len(latent_space) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND,
        user_number,
        decisions_left=1+game_state.inflation(),
        latent_space=latent_space
    ))

def activate_take_another_turn(game_state: 'GameSchema', user_number: int):
    if game_state.extra_turns_taken < 2:
        game_state.extra_turns_taken += 1
        game_state.game_message("<< Extra turn! >>", GameMessageType.SPECIAL_EFFECT)
        game_state.extra_turn = True


def activate_rotate_hands(game_state: 'GameSchema', user_number: int):
    # TODO: choice of rotation direction matters if multiple players are ever added
    rotation_direction = 1
    n = len(game_state.players)

    current_index = 0
    temp = game_state.players[current_index].hand
    for _ in range(n):
        next_index = current_index + rotation_direction
        if next_index == -1:
            next_index = n - 1
        elif next_index == n:
            next_index = 0

        swap = game_state.players[next_index].hand

        game_state.players[next_index].hand = temp
        temp = swap

        current_index = (current_index + rotation_direction) % n
        if current_index == -1:
            current_index = n - 1
        elif current_index == n:
            current_index = 0


ACTION_FUNCTIONS = {
    "use_what_you_take": activate_use_what_you_take,
    "zap_a_card": activate_zap_a_card,
    "trash_a_new_rule": activate_trash_a_new_rule,
    "trash_a_keeper": activate_trash_a_keeper,
    "trade_hands": activate_trade_hands,
    "todays_special": activate_todays_special,
    "draw_2_and_use_em": activate_draw_2_and_use_em,
    "draw_3_play_2_of_them": activate_draw_3_play_2_of_them,
    "steal_a_keeper": activate_steal_a_keeper,
    "share_the_wealth": activate_share_the_wealth,
    "rules_reset": activate_rules_reset,
    "rock_paper_scissors_showdown": activate_rock_paper_scissors_showdown,
    "random_tax": activate_random_tax,
    "no_limits": activate_no_limits,
    "lets_simplify": activate_lets_simplify,
    "lets_do_that_again": activate_lets_do_that_again,
    "jackpot": activate_jackpot,
    "exchange_keepers": activate_exchange_keepers,
    "empty_the_trash": activate_empty_the_trash,
    "discard_and_draw": activate_discard_and_draw,
    "everybody_gets_1": activate_everybody_gets_1,
    "take_another_turn": activate_take_another_turn,
    "rotate_hands": activate_rotate_hands,
}


def activate_action(action_name: str, game_state: 'GameSchema', user_number: int):
    action_function = ACTION_FUNCTIONS.get(action_name)
    if action_function is None:
        raise Exception(f"Error: Action [[{action_name}]] not implemented")

    action_function(game_state, user_number)