from agents.PPOAgent import PPO
from fluxx.env.FluxxEnv import FluxxEnv
from fluxx.game.Cards import card_lists
from fluxx.game.Game import Game

two_player_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True)
env = FluxxEnv(two_player_simple_fluxx, 2, render_mode="human")

model = PPO(env, ["player_0", "player_1"])
model.learn(100000)
