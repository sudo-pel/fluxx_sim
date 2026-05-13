def index_of_card(list_of_cards, card_name):
    for i, card in enumerate(list_of_cards):
        if card.name == card_name:
            return i
    return -1