from typing import Optional, Union

from fluxx.game import GameSchema, game_messages
from fluxx.game.FluxxEnums import CardType, CardZone, ExtendedCardZone, AnyCardZone, GamePhase, GamePhaseType
from dataclasses import dataclass

@dataclass
class CardLocation:
    """
    Class (kinda like a complex enum) for encoding where cards are for passing between functions
    """
    zone: CardZone
    index: int
    zone_index: Optional[int] = None

def trash_selected_card(game: GameSchema, user_number: int, card_location: CardLocation, add_to_discard: bool):
    if card_location.zone == CardZone.RULES:
        rule_discarded = get_selected_card(game, card_location)
        if rule_discarded.name == "double_agenda" and len(game.goals) == 2:
            game.stack.append(GamePhase(GamePhaseType.DISCARD_GOAL_IN_PLAY, user_number, decisions_left=1))
        if add_to_discard:
            game.discard_pile.append(game.rules[card_location.index])
        del game.rules[card_location.index]
        return

    if card_location.zone == CardZone.KEEPERS:
        if add_to_discard:
            game.discard_pile.append(game.players[card_location.zone_index].keepers[card_location.index])
        del game.players[card_location.zone_index].keepers[card_location.index]
        return

    if card_location.zone == CardZone.GOALS:
        if add_to_discard:
            game.discard_pile.append(game.goals[card_location.index])
        del game.goals[card_location.index]
        return

    if card_location.zone == CardZone.DISCARD_PILE:
        del game.discard_pile[card_location.index]
        return

    if card_location.zone == CardZone.HAND:
        if add_to_discard:
            game.discard_pile.append(game.players[card_location[1]].hand[card_location[2]])
        del game.players[card_location[1]].hand[card_location[2]]
        return

    raise Exception("Error: invalid card location")

def get_selected_card(game: GameSchema, card_location: CardLocation):
    if card_location.zone == CardZone.RULES:
        return game.rules[card_location.index]

    if card_location.zone == CardZone.KEEPERS:
        return game.players[card_location.zone_index].keepers[card_location.index]

    if card_location.zone == CardZone.GOALS:
        return game.goals[card_location.index]

    if card_location.zone == CardZone.DISCARD_PILE:
        return game.discard_pile[card_location.index]

    if card_location.zone == CardZone.HAND:
        return game.players[card_location.zone_index].hand[card_location.index]

    raise Exception("Error: invalid card location")

def find_card_in_play_by_name(game: GameSchema, card_name: str) -> Optional[CardLocation]:
    for i, rule in enumerate(game.rules):
        if rule.name == card_name:
            return CardLocation(CardZone.RULES, i)
    for i, player in enumerate(game.players):
        for j, keeper in enumerate(player.keepers):
            if keeper.name == card_name:
                return CardLocation(CardZone.KEEPERS, j, i)
    for i, goal in enumerate(game.goals):
        if goal.name == card_name:
            return CardLocation(CardZone.GOALS, i)
    for i, card in enumerate(game.discard_pile):
        if card.name == card_name:
            return CardLocation(CardZone.DISCARD_PILE, i)
    return None

def select_card(game: GameSchema, user_number: int, select_from: list[AnyCardZone], exclude_types: list[CardType]=()):
    """
    (Prompt an agent to) select a card from a range of options

    :param game: Game object
    :param user_number: Index of the player who is selecting a card
    :param select_from: 'Places' from which the card might be selected
        'rules',
        'keepers',
        'goals',
    :param exclude_types: CardTypes to exclude from selection
    :return: CardLocation of the selected card
    """
    select_from = set(select_from)
    exclude_types = set([e.name for e in exclude_types])

    card_selection = []
    card_location = {}

    if CardZone.RULES in select_from:
        for i, rule in enumerate(game.rules):
            card_selection.append(rule.name)
            card_location[rule.name] = CardLocation(CardZone.RULES, i)

    if CardZone.KEEPERS in select_from:
        for player_number, player in enumerate(game.players):
            for i, keeper in enumerate(player.keepers):
                card_selection.append(keeper.name)
                card_location[keeper.name] = CardLocation(CardZone.KEEPERS, i)

    if CardZone.GOALS in select_from:
        for i, goal in enumerate(game.goals):
            card_selection.append(goal.name)
            card_location[goal.name] = CardLocation(CardZone.GOALS, i)

    if ExtendedCardZone.ENEMY_KEEPERS in select_from:
        for player_number, player in enumerate(game.players):
            if player_number == user_number:
                continue
            for i, keeper in enumerate(player.keepers):
                card_selection.append(keeper.name)
                card_location[keeper.name] = CardLocation(CardZone.KEEPERS, i, player_number)

    if CardZone.DISCARD_PILE in select_from:
        for i, card in enumerate(game.discard_pile):
            if card.card_type.name not in exclude_types:
                card_selection.append(card.name)
                card_location[card.name] = CardLocation(CardZone.DISCARD_PILE, i)

    if ExtendedCardZone.OWN_KEEPERS in select_from:
        for i, keeper in enumerate(game.players[user_number].keepers):
            card_selection.append(keeper.name)
            card_location[keeper.name] = CardLocation(CardZone.KEEPERS, i, user_number)

    if CardZone.HAND in select_from:
        for i, card in enumerate(game.players[user_number].hand):
            if card.card_type.name not in exclude_types:
                card_selection.append(card.name)
                card_location[card.name] = CardLocation(CardZone.HAND, i, user_number)


    if len(card_selection) == 0:
        game_messages.notification("There are no cards to select!")
        return None

    selected_card_name = game.agents[user_number].select_card(game, card_selection)
    return card_location[selected_card_name]
