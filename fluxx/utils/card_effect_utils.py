from typing import TYPE_CHECKING

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

def trash_selected_card(game_state: 'Game', user_number: int, card_location: list, add_to_discard: bool):
    if card_location[0] == "rules":
        rule_discarded = get_selected_card(game_state, card_location)
        if rule_discarded.name == "double_agenda" and len(game_state.goals) == 2:
            goal_to_discard = select_card(game_state, user_number, ["goals"])
            trash_selected_card(game_state, user_number, goal_to_discard, True)

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

    if card_location[0] == "hand":
        if add_to_discard:
            game_state.discard_pile.append(game_state.players[card_location[1]].hand[card_location[2]])
        del game_state.players[card_location[1]].hand[card_location[2]]
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

    if card_location[0] == "hand":
        return game_state.players[card_location[1]].hand[card_location[2]]

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
        ["discard_pile", i],
        ["hand", player_number (user), i],
    """
    select_from = set(select_from)
    exclude_types = set([e.name for e in exclude_types])

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
            if card.card_type.name not in exclude_types:
                card_selection.append(card.name)
                card_location[card.name] = ["discard_pile", i]

    if "own_keepers" in select_from:
        for i, keeper in enumerate(game_state.players[user_number].keepers):
            card_selection.append(keeper.name)
            card_location[keeper.name] = ["keepers", user_number, i]

    if "hand" in select_from:
        for i, card in enumerate(game_state.players[user_number].hand):
            if card.card_type.name not in exclude_types:
                card_selection.append(card.name)
                card_location[card.name] = ["hand", user_number, i]


    if len(card_selection) == 0:
        game_state.game_message("There are no cards to select!")
        return None

    selected_card_name = game_state.agents[user_number].select_card(game_state, card_selection)
    return card_location[selected_card_name]
