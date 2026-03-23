from agents.PPOAgent import PPO
from fluxx.env.FluxxEnv import FluxxEnv
from fluxx.game.Cards import card_lists
from fluxx.game.FluxxEnums import GameState
from fluxx.game.Game import Game
from fluxx.game_states import two_player_p0_one_turn_win, two_player_p0_two_turn_win

two_player_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True)
one_turn_win_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True, force_game_state=two_player_p0_one_turn_win)
two_turn_win_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True, force_game_state=two_player_p0_two_turn_win)

env = FluxxEnv(two_player_simple_fluxx, 2, render_mode="human")

model = PPO(env, ["player_0", "player_1"])
model.learn(100000)
