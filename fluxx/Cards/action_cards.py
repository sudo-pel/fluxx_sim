from typing import TYPE_CHECKING

import random

# avoiding circular import
from fluxx.Card import CardType
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

def trash_selected_card(game_state: 'Game', card_location: list, add_to_discard: bool):
    if card_location[0] == "rules":
        if add_to_discard:
            game_state.discard_pile.append(game_state.rules[card_location[1]])
        del game_state.rules[card_location[1]]
        return

    if card_location[0] == "keepers":
        if add_to_discard:
            game_state.discard_pile.append(game_state.players[card_location[1]].keepers[card_location[2]])
        del game_state.players[card_location[1]].keepers[card_location[2]]
        return

    if card_location[0] == "goals":
        if add_to_discard:
            game_state.discard_pile.append(game_state.goals[card_location[1]])
        del game_state.goals[card_location[1]]
        return

    if card_location[0] == "discard_pile":
        del game_state.discard_pile[card_location[1]]
        return

    raise Exception("Error: invalid card location")

def get_selected_card(game_state: 'Game', card_location: list):
    if card_location[0] == "rules":
        return game_state.rules[card_location[1]]

    if card_location[0] == "keepers":
        return game_state.players[card_location[1]].keepers[card_location[2]]

    if card_location[0] == "goals":
        return game_state.goals[card_location[1]]

    if card_location[0] == "discard_pile":
        return game_state.discard_pile[card_location[1]]

    raise Exception("Error: invalid card location")

def select_card(game_state: 'Game', user_number: int, select_from: list[str], exclude_types: list[CardType]=()):
    """
    (Prompt an agent to) select a card from a range of options

    :param game_state: Game object
    :param user_number: Index of the player who is selecting a card
    :param select_from: 'Places' from which the card might be selected
        'rules',
        'keepers',
        'goals',
    :param exclude_types: CardTypes to exclude from selection
    :return: Location of the selected card
        ["rules", i],
        ["keeper", player_number, i],
        ["goals", i],
        ["discard_pile", i]
    """
    select_from = set(select_from)
    exclude_types = set(exclude_types)

    card_selection = []
    card_location = {}

    if "rules" in select_from:
        for i, rule in enumerate(game_state.rules):
            card_selection.append(rule.name)
            card_location[rule.name] = ["rules", i]

    if "keepers" in select_from:
        for player_number, player in enumerate(game_state.players):
            for i, keeper in enumerate(player.keepers):
                card_selection.append(keeper.name)
                card_location[keeper.name] = ["keepers", player_number, i]

    if "goals" in select_from:
        for i, goal in enumerate(game_state.goals):
            card_selection.append(goal.name)
            card_location[goal.name] = ["goals", i]

    if "enemy_keepers" in select_from:
        for player_number, player in enumerate(game_state.players):
            if player_number == user_number:
                continue
            for i, keeper in enumerate(player.keepers):
                card_selection.append(keeper.name)
                card_location[keeper.name] = ["keepers", player_number, i]

    if "discard_pile" in select_from:
        for i, card in enumerate(game_state.discard_pile):
            if card.card_type not in exclude_types:
                card_selection.append(card.name)
                card_location[card.name] = ["discard_pile", i]

    if "own_keepers" in select_from:
        for i, keeper in enumerate(game_state.players[user_number].keepers):
            card_selection.append(keeper.name)
            card_location[keeper.name] = ["keepers", user_number, i]

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
        selected_card_location = select_card(game_state, user_number, ["rules", "keepers", "goals"])

        if selected_card_location is None:
            return

        selected_card = get_selected_card(game_state, selected_card_location)
        user_player.hand.append(selected_card)
        trash_selected_card(game_state, selected_card_location, False)

    elif action_name == "trash_a_new_rule":
        selected_card_location = select_card(game_state, user_number, ["rules"])

        if selected_card_location is None:
            return

        trash_selected_card(game_state, selected_card_location, True)

    elif action_name == "trash_a_keeper":
        selected_card_location = select_card(game_state, user_number, ["keepers"])

        if selected_card_location is None:
            return

        trash_selected_card(game_state, selected_card_location, True)

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

    elif action_name == "steal_a_keeper":
        selected_card_location = select_card(game_state, user_number, ["enemy_keepers"])

        if selected_card_location is None:
            return

        selected_card = get_selected_card(game_state, selected_card_location)
        user_player.keepers.append(selected_card)
        trash_selected_card(game_state, selected_card_location, False)

    elif action_name == "share_the_wealth":
        all_keepers = [ keeper for player in game_state.players for keeper in player.keepers ]
        for player in game_state.players:
            player.keepers = []

        for i in range(len(all_keepers)):
            game_state.players[i % len(game_state.players)].keepers.append(all_keepers[i])

        game_state.game_message("<< Keepers redistributed! >>")

    elif action_name == "rules_reset":
        for rule in game_state.rules:
            game_state.discard_pile.append(rule)

        game_state.rules = []

    elif action_name == "rock_paper_scissors_showdown":
        selected_player_number = player_agent.select_player_besides_self(game_state)
        selected_player = game_state.players[selected_player_number]

        coin = random.randint(0, 1)
        if coin == 0:
            winner = user_player
            loser = selected_player
        else:
            winner = selected_player
            loser = user_player

        game_state.game_message(f"<< Player {winner.id} defeated {loser.id} in RPS! >>")

        winner.hand += loser.hand
        loser.hand = []

    elif action_name == "random_tax":
        for player_number, player in enumerate(game_state.players):
            if len(player.hand) == 0 or player_number == user_number:
                continue

            index_to_take = random.randint(0, len(player.hand)-1)
            card_to_take = player.hand[index_to_take]

            game_state.game_message(f"<< Stolen {card_to_take.name} from Player {player_number}! >>")

            user_player.hand.append(card_to_take)
            del player.hand[index_to_take]

    elif action_name == "no_limits":
        new_rules = []
        for rule in game_state.rules:
            if rule.keeper_limit is None and rule.hand_limit is None:
                new_rules.append(rule)

        game_state.rules = new_rules

    elif action_name == "lets_simplify":
        to_discard = (len(game_state.rules)+1) // 2
        for i in range(to_discard):
            selected_card_location = select_card(game_state, user_number, ["rules"])
            trash_selected_card(game_state, selected_card_location, True)

    elif action_name == "lets_do_that_again":
        # Note: the discarded card should not be in the discard pile when it is played (in case it has its own discard interaction)
        selected_card_location = select_card(game_state, user_number, ["discard_pile"], [CardType.GOAL, CardType.KEEPER])
        selected_card = get_selected_card(game_state, selected_card_location)
        trash_selected_card(game_state, selected_card_location, False)
        game_state.activate_card(user_number, selected_card)
        game_state.discard_pile.append(selected_card)

    elif action_name == "jackpot":
        for i in range(3):
            game_state.draw(user_player)

    elif action_name == "exchange_keepers":
        keeper_to_give_location = select_card(game_state, user_number, ["own_keepers"])
        keeper_to_take_location = select_card(game_state, user_number, ["enemy_keepers"])

        if keeper_to_give_location is None or keeper_to_take_location is None:
            return

        keeper_to_give = get_selected_card(game_state, keeper_to_give_location)
        keeper_to_take = get_selected_card(game_state, keeper_to_take_location)

        trash_selected_card(game_state, keeper_to_give_location, False)
        trash_selected_card(game_state, keeper_to_take_location, False)
        user_player.keepers.append(keeper_to_take)
        game_state.players[keeper_to_take_location[1]].keepers.append(keeper_to_give)

        game_state.game_message(f"<< Swapped {keeper_to_give.name} for {keeper_to_take.name}! >>")

    elif action_name == "empty_the_trash":
        game_state.draw_pile += game_state.discard_pile
        discard_pile_size = len(game_state.discard_pile)
        game_state.discard_pile = []
        random.shuffle(game_state.draw_pile)
        game_state.game_message(f"<< Shuffled {discard_pile_size} cards from discard pile into deck! >>")

    elif action_name == "discard_and_draw":
        draw_amount = len(user_player.hand)

        game_state.discard_pile.extend(user_player.hand)
        user_player.hand = []

        for i in range(draw_amount):
            game_state.draw(user_player)

    elif action_name == "everybody_gets_1":
        latent_card_pile = [game_state.get_card_from_draw_pile() for i in range(len(game_state.players))]
        player_set = [player.id for player in game_state.players]

        for card in latent_card_pile:
            game_state.game_message(f"<< Drawn {card.name} >>")

        while player_set:
            current_card = latent_card_pile.pop()
            game_state.game_message(f"Please select a player to receive << {current_card.name} >>:")
            chosen_player_number = player_agent.select_player_from_set(game_state, player_set)

            chosen_player = game_state.players[chosen_player_number]
            chosen_player.hand.append(current_card)

            del player_set[chosen_player_number]

    elif action_name == "take_another_turn":
        game_state.game_message("<< Extra turn! >>")
        game_state.extra_turn = True

    elif action_name == "rotate_hands":
        rotation_direction = player_agent.select_player_rotation_direction(game_state)
        n = len(game_state.players)

        current_index = 0
        temp = game_state.players[current_index].hand
        for i in range(n):
            next_index = current_index + rotation_direction
            if next_index == -1:
                next_index = n-1
            elif next_index == n:
                next_index = 0

            swap = game_state.players[next_index].hand

            game_state.players[next_index].hand = temp
            temp = swap

            current_index = (current_index + rotation_direction) % n
            if current_index == -1:
                current_index = n-1
            elif current_index == n:
                current_index = 0
    else:
        raise Exception(f"Error: Action [[{action_name}]] not implemented")