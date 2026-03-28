from enum import Enum


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    YELLOW_TEXT = '\x1b[93m'
    CYAN_TEXT = '\x1b[96m'
    GREEN_TEXT = '\x1b[92m'
    RED_TEXT = '\x1b[91m'

class GameMessageType(Enum):
    SPECIAL_EFFECT = 0,
    NOTIFICATION = 1,
    DRAWN_CARD = 2,
    TURN_START = 3,
    GAME_OVER = 4

def special_effect(message: str):
    print(f"{bcolors.YELLOW_TEXT}{message}{bcolors.ENDC}")

def notification(message: str):
    print(f"{bcolors.CYAN_TEXT}{message}{bcolors.ENDC}")

def drawn_card(message: str):
    print(f"{bcolors.GREEN_TEXT}{message}{bcolors.ENDC}")

def turn_start(message: str):
    print(f"{bcolors.UNDERLINE}{bcolors.BOLD}{message}{bcolors.ENDC}")

def game_over(message: str):
    print(f"{bcolors.RED_TEXT}{message}{bcolors.ENDC}")