# TODO: make this an abstract class
import os
from pathlib import Path

from scripts import debug_utils
from src.game.FluxxEnums import GameState
from abc import ABC, abstractmethod

from src.game.game_messages import GameMessageType

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class Logger(ABC):
    @abstractmethod
    def game_message(self, message, message_type):
        raise NotImplementedError

    @abstractmethod
    def game_stepped(self, game_state: GameState):
        raise NotImplementedError

class GameLogLogger(Logger):
    def __init__(self, filepath: str):
        self.filepath = f"{PROJECT_ROOT}/game_logs/{filepath}"
        f = open(self.filepath, "w+")
        f.close()

    def game_message(self, message: str, message_type: GameMessageType):
        with open(self.filepath, "a") as f:
            f.write(message + "\n")

    def game_stepped(self, game_state: GameState):
        with open(self.filepath, "a") as f:
            output = []
            for i in range(game_state.player_count):
                output.append(f"==========PLAYER_{i} STATE==========\n")
                output.extend(debug_utils.printout_state(i, game_state))
            f.write("\n".join(output) + "\n")

    def game_over(self, winner: int, game_state: GameState):
        with open(self.filepath, "r") as f:
            current_file = f.read()
        with open(self.filepath, "w+") as f:
            f.write(f"GAME TURNS: {game_state.turn_count}\nWINNER: PLAYER {winner}\nGAME LOGS BELOW\n--------------------\n\n")
            f.write(current_file)
            f.write(f"GAME OVER: PLAYER {winner} WINS\n")
        self.game_stepped(game_state)
