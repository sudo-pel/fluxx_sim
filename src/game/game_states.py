from src.game.FluxxEnums import GameState

# ---------------
# PUZZLES
# ---------------

# PUZZLE A: p0 must play "the_sun" to win via "day dreams", or else the opponent will play "time" and win via "time_is_money".
puzzle_a = GameState(
    0,
    2,
    [
        ["the_sun", "the_party", "the_brain", "sleep", "chocolate"],
        ["time"]
    ],
    [
        ["dreams"],
        ["money"]
    ],
    ["day_dreams", "time_is_money"],
    [],
    [],
    [],
    [],
    [],
    False,
    0
)

# PUZZLE A2: p0 must play "sheep" to win via "sheepdog", or else the opponent will play "excalibur" and win via "sword_in_the_stone".
puzzle_a2 = GameState(
    0,
    2,
    [
        ["sheep", "the_party", "the_brain", "sleep", "chocolate"],
        ["excalibur"]
    ],
    [
        ["dog"],
        []
    ],
    ["sheepdog", "sword_in_the_stone"],
    [],
    [],
    [],
    [],
    [],
    False,
    0
)

# p0 must play day_dreams, and then the_sun to win
# p1 will draw into "time" and play it, and then play "time_is_money"
# p1 will have the keepers required and so will win as soon as "time_is_money" is played: p0 cannot interact with their gameplan and must play two perfect turns

puzzle_b = GameState(
    0,
    2,
    [
        ["the_sun", "day_dreams", "the_party", "the_brain"],
        []
    ],
    [
        ["dreams"],
        ["money"]
    ],
    [],
    [],
    ["time_is_money", "keeper_limit_4", "time", "keeper_limit_4"],
    [],
    [],
    [],
    False,
    0
)