import torch

from agents.FeedForwardNN import FeedForwardNN
from agents.RandomAgent import RandomAgent
from fluxx.env.FluxxEnv import FluxxEnv
from fluxx.game.Cards import card_lists
from fluxx.game.Game import Game
from fluxx.game_states import two_player_p0_one_turn_win, two_player_p0_two_turn_win

actor = FeedForwardNN(374, 186)  # same architecture
actor.load_state_dict(torch.load("actor.pt"))
actor.eval()

agents = {
    "player_0": actor,
    "player_1": RandomAgent()
}

def main(one_turn_win_simple_fluxx=None):
    two_player_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True)
    one_turn_win_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True, force_game_state=two_player_p0_one_turn_win)
    two_turn_win_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True, force_game_state=two_player_p0_two_turn_win)

    env = FluxxEnv(two_player_simple_fluxx, 2, render_mode="human")

    victories = {
        "player_0": 0,
        "player_1": 0
    }

    GAME_COUNT = 1000

    for i in range(5):

        round_victories = {
            "player_0": 0,
            "player_1": 0
        }

        for i in range(GAME_COUNT):
            env.reset()

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    action = None
                else:
                    # this is where you would insert your policy
                    action, _= agents[agent].act(observation)
                    action = env.decode_action(action)
                    #input("Press enter to continue...")

                #print(f"Agent {agent} took action {action}")
                env.step(action)
                #printout_state(env.get_player_number(agent), env.game.get_game_state())
                #print(env.game.stack)

            #print(env.game.winner)
            round_victories[f"player_{env.game.winner}"] += 1
            env.close()


        victories["player_0"] += round_victories["player_0"]
        victories["player_1"] += round_victories["player_1"]

        print(f"Round Victories: {round_victories}")
        print(f"Victories: {victories}")

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