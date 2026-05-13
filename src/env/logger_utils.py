max_row_size = 30
ui_size = 94
col_size = 30

def printout_state(player_number, game_state):
    """Print out the game state"""
    hand = [f"({i}) - {c}" for i, c in enumerate(game_state.hands[player_number])]

    enemy_keepers = []
    for i, keepers in enumerate(game_state.keepers):
        if i == player_number:
            continue
        enemy_keepers.append(f" ***PLAYER {i} KEEPERS***")
        for keeper in keepers:
            enemy_keepers.append(keeper)

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

    add_to_row(0, [" ***HAND***"] + hand + ["", " ***KEEPERS***"] + game_state.keepers[player_number])
    add_to_row(1, enemy_keepers)
    add_to_row(2, [" ***GOAL***"] + game_state.goals + ["", " ***rules***"] + game_state.rules)

    output_string.append("-" * ui_size)
    return output_string