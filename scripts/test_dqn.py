import os

from src.agents.training.DQN import DQN

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from src.env.FluxxEnv import FluxxEnv
from src.game.cards import card_lists
from src.game.Game import Game
from src.game.game_states import two_player_p0_one_turn_win, two_player_p0_two_turn_win

two_player_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True)
two_player_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True)
one_turn_win_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True, force_game_state=two_player_p0_one_turn_win)
two_turn_win_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True, force_game_state=two_player_p0_two_turn_win)

env = FluxxEnv(two_player_fluxx, 2, render_mode="human")

model = DQN(env, ["player_0", "player_1"])
model.learn(10000000)
