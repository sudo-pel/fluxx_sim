from Agents.Agent import Agent
from itertools import chain

max_row_size = 30
ui_size = 94
col_size = 30

class PlayerControlledAgent(Agent):
    def play_card(self, game_state):
        self.printout_state(game_state)
        return int(input(f"(PLAYER {self.player_number}) Please choose a card to play >> "))

    def discard_keeper(self, game_state):
        self.printout_state(game_state)
        return int(input(f"(PLAYER {self.player_number}) Please choose a KEEPER to discard >>"))

    def discard_from_hand(self, game_state):
        self.printout_state(game_state)
        return int(input(f"(PLAYER {self.player_number}) Please choose a CARD to discard from HAND >>"))

    def select_card(self, game_state, selection):
        for card_name in selection:
            print(card_name)

        return input(f"(PLAYER {self.player_number}) Please choose a CARD (type the name) from the following above >>")

    def select_player_besides_self(self, game_state):
        player_numbers = [i for i in range(len(game_state.players))]
        del player_numbers[self.player_number]

        return int(input(f"(PLAYER {self.player_number}) Please choose a PLAYER NUMBER from the following: {player_numbers}>>"))

    def select_player_from_set(self, game_state, player_numbers: list[int]):
        return int(input(f"(PLAYER {self.player_number}) Please choose a PLAYER NUMBER from the following: {player_numbers}"))

    def select_card_to_play(self, game_state, selection):
        for i, card in enumerate(selection):
            print(f"({i}) - {card.name}")

        return int(input(f"(PLAYER {self.player_number}) Please choose a CARD NUMBER from the following above >>"))

    def select_player_rotation_direction(self, game_state):
        return int(input(f"(PLAYER {self.player_number}) Please select a ROTATION DIRECTION (-1 = left, 1 = right) >>"))

    def play_free_action(self, game_state, available_free_actions):
        for i, action in enumerate(available_free_actions):
            print(f"({i}) - {action}")
        print(f"({len(available_free_actions)}) - No free action")

        choice = int(input(f"(PLAYER {self.player_number}) Please choose a FREE ACTION to play (or none)"))

        if choice == len(available_free_actions):
            return None
        else:
            return choice

    def choose_to_discard(self, game_state):
        return int(input(f"(PLAYER {self.player_number}) Choose to discard? [[0: NO, 1: YES]]>>"))

    def printout_state(self, game_state):
        """Print out the game state"""

        goals = [g.name for g in game_state["goals"]]

        rules = [r.name for r in game_state["rules"]]
        player_keepers = [[k.name for k in player.keepers] for player in game_state["players"]]
        hand = [f"({i}) - {c.name}" for i, c in enumerate(game_state["players"][self.player_number].hand)]

        enemy_keepers = []
        for i, player in enumerate(game_state["players"]):
            if i == self.player_number:
                continue
            enemy_keepers.append(f" ***PLAYER {i} KEEPERS***")
            for keeper in player.keepers:
                enemy_keepers.append(keeper.name)

        output_string: list[str] = []

        output_string.append("-" * ui_size)
        output_string.append("|" + " " * (ui_size - 2) + "|")

        def make_row(column: int, content: str):
            if column == 0:
                new_row = "|"
            else:
                new_row = ("|" + " " * col_size) * column
            new_row += content
            output_string.append(new_row)

        def add_to_row(column: int, row_data: list[str]):
            """Only call once per column."""
            for i, row in enumerate(row_data):
                if i > len(output_string) - 3:
                    make_row(column, (row + " " * (max_row_size - len(row)) + "|"))
                else:
                    row_to_append_to = output_string[i+2]
                    output_string[i+2] = row_to_append_to + (row + " " * (max_row_size - len(row)) + "|")

            # pad unfilled columns
            for i in range(len(row_data) + 2, len(output_string)):
                row_to_append_to = output_string[i]
                output_string[i] = row_to_append_to + (" " * max_row_size + "|")

        add_to_row(0, [" ***HAND***"] + hand + ["", " ***KEEPERS***"] + player_keepers[self.player_number])
        add_to_row(1, enemy_keepers)
        add_to_row(2, [" ***GOAL***"] + goals + ["", " ***RULES***"] + rules)

        output_string.append("-" * ui_size)

        for row in output_string:
            print(row)


