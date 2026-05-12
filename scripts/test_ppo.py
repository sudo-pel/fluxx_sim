import faulthandler, signal

faulthandler.enable()
faulthandler.register(signal.SIGUSR1)
faulthandler.dump_traceback_later(600, repeat=True, exit=False)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from src.training.ppo.ppo import PPO
from src.env.FluxxEnv import FluxxEnv
from src.game.cards import card_lists
from src.game.Game import Game
from src.game.game_states import puzzle_a, puzzle_a2, puzzle_b, puzzle_b2, puzzle_c, puzzle_c2

PUZZLES = {
    "puzzle_a": {
        "game_state": puzzle_a,
        "card_list": card_lists.base_deck,
        "testee_player_number": 0
    },
    "puzzle_a2": {
        "game_state": puzzle_a2,
        "card_list": card_lists.expanded_deck,
        "testee_player_number": 0
    },
    "puzzle_b": {
        "game_state": puzzle_b,
        "card_list": card_lists.base_deck,
        "testee_player_number": 0
    },
    "puzzle_b2": {
        "game_state": puzzle_b2,
        "card_list": card_lists.expanded_deck,
        "testee_player_number": 0
    },
    "puzzle_c": {
        "game_state": puzzle_c,
        "card_list": card_lists.base_deck,
        "testee_player_number": 1
    },
    "puzzle_c2": {
        "game_state": puzzle_c2,
        "card_list": card_lists.expanded_deck,
        "testee_player_number": 1
    }
}

GAME_COUNT = 1000
for puzzle_name, puzzle_data in PUZZLES.items():
    fluxx_game = Game(2, puzzle_data["card_list"], disable_game_messages=True, force_game_state=puzzle_data["game_state"])
    env = FluxxEnv(fluxx_game, 2, render_mode="human")

    for i in range(GAME_COUNT):
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action, _, _ = agents[agent].act(observation)
                action = env.decode_action(action)
                # input("Press enter to continue...")

            # print(f"Agent {agent} took action {action}")
            env.step(action)
            # print("\n".join(printout_state(env.get_player_number(agent), env.game.get_game_state())))
            # print(env.game.stack)

        # print(env.game.winner)
        if env.game.winner is None:
            round_victories["draws"] += 1
        else:
            round_victories[f"player_{env.game.winner}"] += 1
        env.close()