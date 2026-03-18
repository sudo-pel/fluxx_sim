from agents.RandomAgent import RandomAgent
from fluxx.env.FluxxEnv import FluxxEnv
from fluxx.game.Cards import card_lists
from fluxx.game.Game import Game

agents = {
    "player_0": RandomAgent(),
    "player_1": RandomAgent()
}

def main():
    two_player_simple_fluxx = Game(2, card_lists.simple_fluxx_deck)
    env = FluxxEnv(two_player_simple_fluxx, 2, render_mode="human")
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = agents[agent].act(observation)
            action = env.decode_action(action)
            print(f"Agent {agent} took action {action}")
            printout_state(env.get_player_number(agent), env.game.get_game_state())
            input("Press enter to continue...")

        env.step(action)

    print(env.game.winner)
    env.close()

max_row_size = 30
ui_size = 94
col_size = 30

def printout_state(player_number, game_state):
    """Print out the game state"""

    goals = [g.name for g in game_state["goals"]]

    rules = [r.name for r in game_state["rules"]]
    player_keepers = [[k.name for k in player.keepers] for player in game_state["players"]]
    hand = [f"({i}) - {c.name}" for i, c in enumerate(game_state["players"][player_number].hand)]

    enemy_keepers = []
    for i, player in enumerate(game_state["players"]):
        if i == player_number:
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

    add_to_row(0, [" ***HAND***"] + hand + ["", " ***KEEPERS***"] + player_keepers[player_number])
    add_to_row(1, enemy_keepers)
    add_to_row(2, [" ***GOAL***"] + goals + ["", " ***RULES***"] + rules)

    output_string.append("-" * ui_size)

    for row in output_string:
        print(row)

if __name__ == '__main__':
    main()