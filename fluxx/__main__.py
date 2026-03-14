from fluxx.game.Game import Game, test_deck
from Agents.PlayerControlled import PlayerControlledAgent

new_game = Game([PlayerControlledAgent(), PlayerControlledAgent()], test_deck)
new_game.run_game()