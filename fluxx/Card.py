from enum import Enum

class CardType(Enum):
    RULE = 1
    KEEPER = 2
    GOAL = 3
    ACTION = 4

# self-note: maybe worth adding something extra for Goals here
class Card:
    def __init__(self, name: str, card_type: CardType):
        self.name = name
        self.card_type = card_type

"""
List of possible rules
draw x (int)
play x (int or all-1)
keeper_limit x (int)
hand_limit x (int)
special? (bool)
    For Rule cards with complex text.
"""
class RulesOptions:
    def __init__(self, draw=None, play=None, keeper_limit=None, hand_limit=None, free_action=False):
        self.draw = draw
        self.play = play
        self.keeper_limit = keeper_limit
        self.hand_limit = hand_limit
        self.free_action = free_action

class Rule(Card):
    def __init__(self, name: str, rules_options: RulesOptions):
        super().__init__(name, CardType.RULE)
        self.draw = rules_options.draw
        self.play = rules_options.play
        self.keeper_limit = rules_options.keeper_limit
        self.hand_limit = rules_options.hand_limit
        self.free_action = rules_options.free_action

    def __getitem__(self, key):
        if key == "draw":
            return self.draw
        elif key == "play":
            return self.play
        elif key == "keeper_limit":
            return self.keeper_limit
        elif key == "hand_limit":
            return self.hand_limit
        elif key == "free_action":
            return self.free_action
        else:
            raise Exception("Invalid key")

class Keeper(Card):
    def __init__(self, name: str):
        super().__init__(name, CardType.KEEPER)

class Goal(Card):
    def __init__(self, name: str, required_keepers: list[str], disallowed_keepers: list[str]=None, optional_keepers: list[list[str]]=None):
        super().__init__(name, CardType.GOAL)
        self.required_keepers = required_keepers

        if disallowed_keepers is None:
            self.disallowed_keepers = []
        else:
            self.disallowed_keepers = disallowed_keepers

        if optional_keepers is None:
            self.optional_keepers = []
        else:
            self.optional_keepers = optional_keepers

class Action(Card):
    def __init__(self, name: str):
        super().__init__(name, CardType.ACTION)