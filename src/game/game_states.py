from src.game.FluxxEnums import GameState

# ---------------
# PUZZLES
# ---------------

# PUZZLE A: p0 must play "the_sun" to win via "day dreams", or else the opponent will play "time" and win via "time_is_money".
puzzle_a = GameState(
    turn_count=0,
    player_count=2,
    hands=[
        ["the_sun", "the_party", "the_brain", "sleep", "chocolate"],
        ["time"]
    ],
    keepers=[
        ["dreams"],
        ["money"]
    ],
    goals=["day_dreams", "time_is_money"],
    discard_pile=[],
    draw_pile=[],
    rules=[],
    stack=[],
)

# PUZZLE A2: p0 must play "sheep" to win via "sheepdog", or else the opponent will play "excalibur" and win via "sword_in_the_stone".
puzzle_a2 = GameState(
    turn_count=0,
    player_count=2,
    hands=[
        ["sheep", "nothingness", "all_the_animals", "curious_monkey", "skymap"],
        ["excalibur"]
    ],
    keepers=[
        ["dog"],
        []
    ],
    goals=["sheepdog", "sword_in_the_stone"],
    discard_pile=[],
    draw_pile=[],
    rules=[],
    stack=[],
)

# PUZZLE B:
"""
p0 must play "day_dreams" to replace "time_is_money", because p1 will play "time" next turn with "money" in play
p0 must then play "the_sun" to win via "day_dreams", because p1 will draw into "time_is_money" again and play it with "time" and "money" in play
"""
puzzle_b = GameState(
    turn_count=0,
    player_count=2,
    hands=[
        ["the_sun", "day_dreams", "the_party", "the_brain"],
        []
    ],
    keepers=[
        ["dreams"],
        ["money"]
    ],
    goals=["time_is_money"],
    discard_pile=[],
    draw_pile=["time", "the_moon", "time_is_money"],
    rules=[],
    stack=[],
)

# PUZZLE B2:
"""
p0 must play "rainbow" to replace "sheepdog", because p1 will play "sheep" next turn with "dog" in play
p0 must then play "sunshine" to win via "rainbow", because p1 will draw into "sheepdog" again and play it with "sheep" and "dog" in play
"""
puzzle_b2 = GameState(
    turn_count=0,
    player_count=2,
    hands=[
        ["monkey", "rainbow", "book", "key"],
        []
    ],
    keepers=[
        ["rain"],
        ["dog"]
    ],
    goals=["sheepdog"],
    discard_pile=[],
    draw_pile=["sheep", "map", "sheepdog"],
    rules=[],
    stack=[],
)

# PUZZLE C:
"""
TESTEE IS PLAYER 1

p0 plays "hand_limit_3". p1 must keep "the_sun", "dreams" and "day_dreams", a set of coherent Goals and Keepers.
p1 must discard "5_keepers" using "goal_mill". Then, they must play the drawn "play_3" rule card and play the aforementioned winning gameplan.

If they don't the battle handler will truncate the game state, resulting in a draw.
"""
puzzle_c = GameState(
    turn_count=0,
    player_count=2,
    hands=[
        ["hand_limit_3"],
        ["the_sun", "dreams", "day_dreams", "cookies", "time", "the_eye"]
    ],
    keepers=[
        [],
        []
    ],
    goals=["sheepdog"],
    discard_pile=[],
    draw_pile=["5_keepers", "play_3", "toast", "winning_the_lottery"],
    rules=["goal_mill"],
    stack=[],
)

# PUZZLE C2:
"""
TESTEE IS PLAYER 1

p0 plays "hand_limit_3". p1 must keep "sheepdog", "sheep" and "dog", a set of coherent Goals and Keepers.
p1 must play "space_jackpot", drawing "play_3", "sunshine" and "fire". Then, they must play "play_4", "sheep", "dog" and "sheepdog" in order to win the game.

If they don't the battle handler will truncate the game state, resulting in a draw.
"""
puzzle_c2 = GameState(
    turn_count=0,
    player_count=2,
    hands=[
        ["hand_limit_3"],
        ["sheep", "dog", "sheepdog", "air", "excalibur", "the_void"]
    ],
    keepers=[
        [],
        []
    ],
    goals=["sheepdog"],
    discard_pile=[],
    draw_pile=["space_jackpot", "play_4", "sunshine", "fire"],
    rules=["goal_mill"],
    stack=[],
)
