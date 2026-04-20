class Player:
    def __init__(self, id: int):
        self.id = id
        self.cards_played = 0
        self.cards_drawn = 0
        self.hand = []
        self.keepers = []