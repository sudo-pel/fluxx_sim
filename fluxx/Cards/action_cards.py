from typing import TYPE_CHECKING

import random

# avoiding circular import
if TYPE_CHECKING:
    from fluxx import Game

def draw_and_play(game_state: 'Game', user_number: int, draw_amount: int, play_amount: int, place_remainder: str):
    """
    Draw [draw_amount] cards (not into hand) and play [play_amount] cards. Put the unplayed cards into [place_remainder]

    :param game_state: 'Game' object
    :param user_number: Index of player drawing
    :param draw_amount: Number of cards to draw
    :param play_amount: Number of cards to play
    :param place_remainder: String representing where to place unplayed cards
        'discard_pile',
        'hand'
    :return: None
    """
    latent_card_pile = [game_state.get_card_from_draw_pile() for i in range(draw_amount)]

    for i in range(play_amount):
        selected_card_index = game_state.agents[user_number].select_card_to_play(game_state, latent_card_pile)
        selected_card = latent_card_pile[selected_card_index]
        game_state.game_message(f"<< PLAYED {selected_card.name} >>")
        game_state.activate_card(user_number, selected_card)
        del latent_card_pile[selected_card_index]

    if place_remainder == "discard_pile":
        for card in latent_card_pile:
            game_state.discard_pile.append(card)

def trash_selected_card(game_state: 'Game', card_location: list):
    if card_location[0] == "rules":
        del game_state.rules[card_location[1]]
        return

    if card_location[0] == "keepers":
        del game_state.players[card_location[1]].keepers[card_location[2]]
        return

    if card_location[0] == "goals":
        del game_state.goals[card_location[1]]
        return

    raise Exception("Error: invalid card location")

def get_selected_card(game_state: 'Game', card_location: list):
    if card_location[0] == "rules":
        return game_state.rules[card_location[1]]

    if card_location[0] == "keepers":
        return game_state.players[card_location[1]].keepers[card_location[2]]

    if card_location[0] == "goals":
        return game_state.goals[card_location[1]]

    raise Exception("Error: invalid card location")

def select_card(game_state: 'Game', user_number: int, *options):
    """
    (Prompt an agent to) select a card from a range of options

    :param game_state: Game object
    :param user_number: Index of the player who is selecting a card
    :param options: 'Places' from which the card might be selected
        'rules',
        'keepers',
        'goals',
    :return: Location of the selected card
        ["rules", i],
        ["keeper", player_number, i],
        ["goals", i]
    """
    card_selection = []
    card_location = {}

    if "rules" in options:
        for i, rule in enumerate(game_state.rules):
            card_selection.append(rule.name)
            card_location[rule.name] = ["rules", i]

    if "keepers" in options:
        for player_number, player in enumerate(game_state.players):
            for i, keeper in enumerate(player.keepers):
                card_selection.append(keeper.name)
                card_location[keeper.name] = ["keepers", player_number, i]

    if "goals" in options:
        for i, goal in enumerate(game_state.goals):
            card_selection.append(goal.name)
            card_location[goal.name] = ["goals", i]

    if len(card_selection) == 0:
        game_state.game_message("There are no cards to select!")
        return None

    selected_card_name = game_state.agents[user_number].select_card(game_state, card_selection)
    return card_location[selected_card_name]

def activate_action(action_name: str, game_state: 'Game', user_number: int):
    user_player = game_state.players[user_number]
    player_agent = game_state.agents[user_number]
    other_players = []
    for player_number, player in enumerate(game_state.players):
        if player_number != user_number:
            other_players.append(player)

    if action_name == "use_what_you_take":
        players_with_cards = [p for p in other_players if len(p.hand) > 0]
        if len(players_with_cards) == 0:
            game_state.game_message("No players have any cards in hand!")
        else:
            player_to_take = random.randint(0, len(players_with_cards)-1)
            card_to_take = random.randint(0, len(players_with_cards[player_to_take].hand)-1)

            card_to_play = players_with_cards[player_to_take].hand[card_to_take]
            del players_with_cards[player_to_take].hand[card_to_take]

            game_state.game_message(f"<< ACTIVATING {card_to_play.name} >>")

            game_state.activate_card(user_number, card_to_play)

    elif action_name == "zap_a_card":
        selected_card_location = select_card(game_state, user_number, "rules", "keepers", "goals")

        if selected_card_location is None:
            return

        selected_card = get_selected_card(game_state, selected_card_location)
        user_player.hand.append(selected_card)
        trash_selected_card(game_state, selected_card_location)

    elif action_name == "trash_a_new_rule":
        selected_card_location = select_card(game_state, user_number, "rules")

        if selected_card_location is None:
            return

        trash_selected_card(game_state, selected_card_location)

    elif action_name == "trash_a_keeper":
        selected_card_location = select_card(game_state, user_number, "keepers")

        if selected_card_location is None:
            return

        trash_selected_card(game_state, selected_card_location)

    elif action_name == "trade_hands":
        selected_player = player_agent.select_player_besides_self(game_state)
        if selected_player > len(game_state.players) or selected_player == user_number:
            raise Exception("Error: invalid player selection")

        temp = game_state.players[selected_player].hand
        game_state.players[selected_player].hand = user_player.hand
        user_player.hand = temp

        game_state.game_message(f"<< Player {user_number} traded hands with player {selected_player} >> ")

    elif action_name == "todays_special":
        draw_and_play(game_state, user_number, 3, 1, "discard_pile")

    elif action_name == "draw_2_and_use_em":
        draw_and_play(game_state, user_number, 2, 2, "discard_pile")

    elif action_name == "draw_3_play_2_of_them":
        draw_and_play(game_state, user_number, 3, 2, "discard_pile")

    else:
        raise Exception(f"Error: Action [[{action_name}]] not implemented")