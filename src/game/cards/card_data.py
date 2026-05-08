"""
EFFECT PARAMETER AXES

0: card draw for self
1: card draw for opponent
2: cards played for self
3: cards played for opponent
4: randomness
5: backfire potential (e.g losing your hand to rock-paper-scissors, or being forced to play a card that wins the opponent the game)
6: turn extension (e.g getting to play more cards or getting another turn)
7: rule removal
8: own keeper removal
9: enemy keeper removal
10: goal removal
11: opponent hand disruption
12: overall impact on game state
13: card selection (how much choice there is in the card impact)
14: card advantage (how many cards this card generates)
15: virtual card advantage (how many cards you get to "choose from", on average)

"""

CARD_DATA = {

    # -----------------------------------------------
    # EXPANDED SET CARDS
    # -----------------------------------------------
    # *Note that some of these cards were invented solely for this project.

    # Actions
    "pandoras_box": {"card_type": "ACTION", "card_effect": "draw cards from the draw pile one at a time put any new rule cards into play and discard the rest until you have played three new rules", "effect_parameters": [0, 0, 0, 0, 9, 4, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0]},
    "rewind": {"card_type": "ACTION", "card_effect": "search the discard pile choose any goal card and play it again", "effect_parameters": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 6, 8, 0, 0]},
    "robin_hood": {"card_type": "ACTION", "card_effect": "move a keeper from the player with the most keepers to the player with the fewest if everyone has the same number of keepers move one from yourself to your opponent", "effect_parameters": [0, 0, 0, 0, 2, 4, 0, 0, 4, 4, 0, 0, 6, 3, 0, 0]},
    "time_vortex": {"card_type": "ACTION", "card_effect": "gather all cards in players hands shuffle them together and deal them back out evenly at random starting with yourself", "effect_parameters": [3, 3, 0, 0, 9, 7, 0, 0, 0, 0, 0, 9, 7, 0, 0, 0]},
    "gift_giveaway": {"card_type": "ACTION", "card_effect": "all players pick up one of their keepers and give it to another player gifts are given in turn order starting with the player to your left and players without keepers receive gifts first", "effect_parameters": [0, 0, 0, 0, 4, 7, 0, 0, 5, 5, 0, 0, 7, 4, 0, 0]},
    "dig_up_the_past": {"card_type": "ACTION", "card_effect": "take any keeper out of the discard pile and add it to your hand", "effect_parameters": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 1, 6]},
    "space_jackpot": {"card_type": "ACTION", "card_effect": "draw five extra cards add them to your hand then discard two cards", "effect_parameters": [9, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 5, 5, 7, 9]},
    "supernova": {"card_type": "ACTION", "card_effect": "discard all keepers in play except up to three named keepers then shuffle the discard pile back into the draw pile", "effect_parameters": [0, 0, 0, 0, 5, 5, 0, 0, 9, 9, 0, 0, 10, 2, 0, 0]},
    "destroy_all_keepers": {"card_type": "ACTION", "card_effect": "destroy all keepers currently in play", "effect_parameters": [0, 0, 0, 0, 0, 4, 0, 0, 9, 9, 0, 0, 9, 0, 0, 0]},
    "roll_for_it": {"card_type": "ACTION", "card_effect": "roll a six sided die one draw one card and play it two draw two cards three draw three cards four trash a new rule five steal a keeper six take another turn", "effect_parameters": [4, 0, 1, 0, 10, 3, 2, 1, 0, 1, 0, 0, 6, 0, 3, 3]},
    "close_enough": {"card_type": "ACTION", "card_effect": "if you have one of the keepers required by the current goal you win otherwise nothing happens", "effect_parameters": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0]},
    "rough_seas": {"card_type": "ACTION", "card_effect": "all players must discard down to three cards in hand and two keepers in play", "effect_parameters": [0, 0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 7, 8, 5, 0, 0]},
    "brain_drain": {"card_type": "ACTION", "card_effect": "each player must discard one card from their hand", "effect_parameters": [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 4, 2, 0, 0]},
    "oops": {"card_type": "ACTION", "card_effect": "discard your entire hand", "effect_parameters": [0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0]},

    # Free Action Rules
    "card_transfusion": {"card_type": "RULE", "RulesOptions": {"free_action": True}, "card_effect": "once per turn you may discard as many cards from your hand as you choose and draw the same number of cards", "effect_parameters": [4, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 5, 5, 0, 5]},
    "plunder": {"card_type": "RULE", "RulesOptions": {"free_action": True}, "card_effect": "once per turn you may steal any keeper another player has in play and put it in front of you", "effect_parameters": [0, 0, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 9, 8, 1, 0]},

    # Keepers
    "cat": {"card_type": "KEEPER"},
    "dog": {"card_type": "KEEPER"},
    "monkey": {"card_type": "KEEPER"},
    "horse": {"card_type": "KEEPER"},
    "cow": {"card_type": "KEEPER"},
    "sheep": {"card_type": "KEEPER"},
    "farmer": {"card_type": "KEEPER"},
    "grass": {"card_type": "KEEPER"},
    "field": {"card_type": "KEEPER"},
    "fire": {"card_type": "KEEPER"},
    "wood": {"card_type": "KEEPER"},
    "rain": {"card_type": "KEEPER"},
    "sunshine": {"card_type": "KEEPER"},
    "book": {"card_type": "KEEPER"},
    "map": {"card_type": "KEEPER"},
    "key": {"card_type": "KEEPER"},
    "excalibur": {"card_type": "KEEPER"},
    "air": {"card_type": "KEEPER"},
    "the_void": {"card_type": "KEEPER"},

    # Goals
    "the_farm": {
        "card_type": "GOAL",
        "required_keepers": ["farmer", "field"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "tending_the_flock": {
        "card_type": "GOAL",
        "required_keepers": ["farmer", "sheep"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "dairy_farm": {
        "card_type": "GOAL",
        "required_keepers": ["farmer", "cow"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "plowing": {
        "card_type": "GOAL",
        "required_keepers": ["farmer", "horse"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "pasture": {
        "card_type": "GOAL",
        "required_keepers": ["field", "grass"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "cows_in_the_field": {
        "card_type": "GOAL",
        "required_keepers": ["field", "cow"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "open_range": {
        "card_type": "GOAL",
        "required_keepers": ["field", "horse"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "grazing_sheep": {
        "card_type": "GOAL",
        "required_keepers": ["grass", "sheep"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "growing_grass": {
        "card_type": "GOAL",
        "required_keepers": ["grass", "rain"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "key_lost_in_the_grass": {
        "card_type": "GOAL",
        "required_keepers": ["grass", "key"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "sheepdog": {
        "card_type": "GOAL",
        "required_keepers": ["sheep", "dog"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "milking_in_the_sun": {
        "card_type": "GOAL",
        "required_keepers": ["cow", "sunshine"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "galloping_on_air": {
        "card_type": "GOAL",
        "required_keepers": ["horse", "air"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "rainbow": {
        "card_type": "GOAL",
        "required_keepers": ["rain", "sunshine"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "cat_in_the_rain": {
        "card_type": "GOAL",
        "required_keepers": ["rain", "cat"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "sunbathing_cat": {
        "card_type": "GOAL",
        "required_keepers": ["sunshine", "cat"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "reading_dog": {
        "card_type": "GOAL",
        "required_keepers": ["book", "dog"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "treasure_map": {
        "card_type": "GOAL",
        "required_keepers": ["book", "map"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "reading_by_fire": {
        "card_type": "GOAL",
        "required_keepers": ["book", "fire"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "curious_monkey": {
        "card_type": "GOAL",
        "required_keepers": ["book", "monkey"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "cartographer": {
        "card_type": "GOAL",
        "required_keepers": ["map", "key"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "skymap": {
        "card_type": "GOAL",
        "required_keepers": ["map", "air"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "campfire": {
        "card_type": "GOAL",
        "required_keepers": ["wood", "fire"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "tree_climber": {
        "card_type": "GOAL",
        "required_keepers": ["wood", "monkey"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "sword_in_the_stone": {
        "card_type": "GOAL",
        "required_keepers": ["excalibur"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "farmer_and_an_animal": {
        "card_type": "GOAL",
        "required_keepers": ["farmer"],
        "disallowed_keepers": [],
        "optional_keepers": [["cat", "dog", "monkey", "horse", "cow"]]
    },
    "all_the_animals": {
        "card_type": "GOAL",
        "required_keepers": ["cat", "dog", "monkey", "horse", "cow"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "nothingness": {
        "card_type": "GOAL",
        "required_keepers": ["the_void"],
        "disallowed_keepers": ["cat", "dog", "monkey", "horse", "cow", "sheep", "farmer", "grass", "field", "fire", "wood", "rain", "sunshine", "book", "map", "key", "excalibur", "air"],
        "optional_keepers": []
    },

    # Actions


    # Rules
    "triple_agenda": {"card_type": "RULE", "RulesOptions": {}, "card_effect": "three goal cards may be in play at the same time and a player wins if they meet either goal", "effect_parameters": [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0]},

    # -----------------------------------------------
    # BASE SET CARDS
    # -----------------------------------------------

    # Rules - Draw/Play/Limits
    "hand_limit_2": {"card_type": "RULE", "RulesOptions": {"hand_limit": 2}, "card_effect": "at the end of your turn discard down to two cards in hand other players must also discard down to two between turns", "effect_parameters": [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 7, 5, 0, 0, 0]},
    "hand_limit_1": {"card_type": "RULE", "RulesOptions": {"hand_limit": 1}, "card_effect": "at the end of your turn discard down to one card in hand other players must also discard down to one between turns", "effect_parameters": [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 9, 6, 0, 0, 0]},
    "hand_limit_0": {"card_type": "RULE", "RulesOptions": {"hand_limit": 0}, "card_effect": "at the end of your turn discard your entire hand other players must also discard their hands between turns", "effect_parameters": [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 10, 8, 0, 0, 0]},
    "keeper_limit_4": {"card_type": "RULE", "RulesOptions": {"keeper_limit": 4}, "card_effect": "no player may have more than four keepers in play discard down to four at the end of your turn", "effect_parameters": [0, 0, 0, 0, 0, 1, 0, 0, 2, 2, 0, 0, 4, 0, 0, 0]},
    "keeper_limit_3": {"card_type": "RULE", "RulesOptions": {"keeper_limit": 3}, "card_effect": "no player may have more than three keepers in play discard down to three at the end of your turn", "effect_parameters": [0, 0, 0, 0, 0, 2, 0, 0, 4, 4, 0, 0, 5, 0, 0, 0]},
    "keeper_limit_2": {"card_type": "RULE", "RulesOptions": {"keeper_limit": 2}, "card_effect": "no player may have more than two keepers in play discard down to two at the end of your turn", "effect_parameters": [0, 0, 0, 0, 0, 3, 0, 0, 6, 6, 0, 0, 7, 0, 0, 0]},
    "draw_5": {"card_type": "RULE", "RulesOptions": {"draw": 5}, "card_effect": "draw five cards at the start of each turn instead of one", "effect_parameters": [9, 9, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 8, 0, 9, 9]},
    "draw_4": {"card_type": "RULE", "RulesOptions": {"draw": 4}, "card_effect": "draw four cards at the start of each turn instead of one", "effect_parameters": [7, 7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 7, 0, 7, 7]},
    "draw_2": {"card_type": "RULE", "RulesOptions": {"draw": 2}, "card_effect": "draw two cards at the start of each turn instead of one", "effect_parameters": [3, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 5, 0, 3, 3]},
    "play_2": {"card_type": "RULE", "RulesOptions": {"play": 2}, "card_effect": "play two cards from your hand each turn instead of one", "effect_parameters": [0, 0, 3, 3, 0, 0, 3, 0, 0, 0, 0, 0, 5, 0, 0, 0]},
    "draw_3": {"card_type": "RULE", "RulesOptions": {"draw": 3}, "card_effect": "draw three cards at the start of each turn instead of one", "effect_parameters": [5, 5, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 6, 0, 5, 5]},
    "play_4": {"card_type": "RULE", "RulesOptions": {"play": 4}, "card_effect": "play four cards from your hand each turn instead of one", "effect_parameters": [0, 0, 7, 7, 0, 0, 6, 0, 0, 0, 0, 0, 8, 0, 0, 0]},
    "play_3": {"card_type": "RULE", "RulesOptions": {"play": 3}, "card_effect": "play three cards from your hand each turn instead of one", "effect_parameters": [0, 0, 5, 5, 0, 0, 5, 0, 0, 0, 0, 0, 7, 0, 0, 0]},

    # Keepers
    "the_sun": {"card_type": "KEEPER"},
    "the_party": {"card_type": "KEEPER"},
    "music": {"card_type": "KEEPER"},
    "dreams": {"card_type": "KEEPER"},
    "love": {"card_type": "KEEPER"},
    "peace": {"card_type": "KEEPER"},
    "sleep": {"card_type": "KEEPER"},
    "the_brain": {"card_type": "KEEPER"},
    "bread": {"card_type": "KEEPER"},
    "chocolate": {"card_type": "KEEPER"},
    "cookies": {"card_type": "KEEPER"},
    "milk": {"card_type": "KEEPER"},
    "time": {"card_type": "KEEPER"},
    "money": {"card_type": "KEEPER"},
    "the_eye": {"card_type": "KEEPER"},
    "the_moon": {"card_type": "KEEPER"},
    "the_rocket": {"card_type": "KEEPER"},
    "the_toaster": {"card_type": "KEEPER"},
    "television": {"card_type": "KEEPER"},

    # Goals
    "the_appliances": {
        "card_type": "GOAL",
        "required_keepers": ["the_toaster", "television"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "baked_goods": {
        "card_type": "GOAL",
        "required_keepers": ["bread", "cookies"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "bed_time": {
        "card_type": "GOAL",
        "required_keepers": ["sleep", "time"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "bread_and_chocolate": {
        "card_type": "GOAL",
        "required_keepers": ["bread", "chocolate"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "cant_buy_me_love": {
        "card_type": "GOAL",
        "required_keepers": ["money", "love"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "chocolate_cookies": {
        "card_type": "GOAL",
        "required_keepers": ["chocolate", "cookies"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "chocolate_milk": {
        "card_type": "GOAL",
        "required_keepers": ["chocolate", "milk"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "day_dreams": {
        "card_type": "GOAL",
        "required_keepers": ["the_sun", "dreams"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "dreamland": {
        "card_type": "GOAL",
        "required_keepers": ["sleep", "dreams"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "the_eye_of_the_beholder": {
        "card_type": "GOAL",
        "required_keepers": ["the_eye", "love"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "great_theme_song": {
        "card_type": "GOAL",
        "required_keepers": ["music", "television"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "hearts_and_minds": {
        "card_type": "GOAL",
        "required_keepers": ["love", "the_brain"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "hippyism": {
        "card_type": "GOAL",
        "required_keepers": ["peace", "love"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "lullaby": {
        "card_type": "GOAL",
        "required_keepers": ["sleep", "music"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "milk_and_cookies": {
        "card_type": "GOAL",
        "required_keepers": ["milk", "cookies"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "the_minds_eye": {
        "card_type": "GOAL",
        "required_keepers": ["the_brain", "the_eye"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "night_and_day": {
        "card_type": "GOAL",
        "required_keepers": ["the_sun", "the_moon"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "party_time": {
        "card_type": "GOAL",
        "required_keepers": ["the_party", "time"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "rocket_science": {
        "card_type": "GOAL",
        "required_keepers": ["the_rocket", "the_brain"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "rocket_to_the_moon": {
        "card_type": "GOAL",
        "required_keepers": ["the_rocket", "the_moon"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "squishy_chocolate": {
        "card_type": "GOAL",
        "required_keepers": ["chocolate", "the_sun"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "time_is_money": {
        "card_type": "GOAL",
        "required_keepers": ["time", "money"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "toast": {
        "card_type": "GOAL",
        "required_keepers": ["bread", "the_toaster"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "turn_it_up": {
        "card_type": "GOAL",
        "required_keepers": ["music", "the_party"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "winning_the_lottery": {
        "card_type": "GOAL",
        "required_keepers": ["dreams", "money"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "world_peace": {
        "card_type": "GOAL",
        "required_keepers": ["dreams", "peace"],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "5_keepers": {
        "card_type": "GOAL",
        "required_keepers": [],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "10_cards_in_hand": {
        "card_type": "GOAL",
        "required_keepers": [],
        "disallowed_keepers": [],
        "optional_keepers": []
    },
    "the_brain_no_tv": {
        "card_type": "GOAL",
        "required_keepers": ["the_brain"],
        "disallowed_keepers": ["television"],
        "optional_keepers": []
    },
    "party_snacks": {
        "card_type": "GOAL",
        "required_keepers": ["the_party"],
        "disallowed_keepers": [],
        "optional_keepers": [["milk", "cookies", "chocolate", "bread"]]
    },

    # Actions
    "use_what_you_take": {"card_type": "ACTION", "card_effect": "take a random card from another player's hand and play it", "effect_parameters": [0, 0, 1, 0, 7, 5, 0, 0, 0, 0, 0, 6, 5, 1, 0, 1]},
    "zap_a_card": {"card_type": "ACTION", "card_effect": "remove any card in play and put it into your hand", "effect_parameters": [1, 0, 0, 0, 0, 1, 0, 7, 4, 7, 7, 0, 8, 9, 1, 1]},
    "trash_a_new_rule": {"card_type": "ACTION", "card_effect": "discard any new rule card currently in play", "effect_parameters": [0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 6, 7, 0, 0]},
    "trash_a_keeper": {"card_type": "ACTION", "card_effect": "choose any keeper in play and discard it", "effect_parameters": [0, 0, 0, 0, 0, 1, 0, 0, 4, 8, 0, 0, 7, 8, 0, 0]},
    "trade_hands": {"card_type": "ACTION", "card_effect": "trade your entire hand of cards with another player", "effect_parameters": [0, 0, 0, 0, 5, 7, 0, 0, 0, 0, 0, 9, 7, 1, 0, 0]},
    "todays_special": {"card_type": "ACTION", "card_effect": "remove a keeper of your choice from play and discard it", "effect_parameters": [0, 0, 0, 0, 0, 1, 0, 0, 4, 8, 0, 0, 7, 8, 0, 0]},
    "draw_2_and_use_em": {"card_type": "ACTION", "card_effect": "set your hand aside draw two cards play them in either order then pick up your hand", "effect_parameters": [3, 0, 2, 0, 6, 2, 4, 0, 0, 0, 0, 0, 5, 5, 2, 2]},
    "draw_3_play_2_of_them": {"card_type": "ACTION", "card_effect": "set your hand aside draw three cards play two of them and discard the third then pick up your hand", "effect_parameters": [3, 0, 2, 0, 6, 2, 5, 0, 0, 0, 0, 0, 6, 7, 2, 3]},
    "steal_a_keeper": {"card_type": "ACTION", "card_effect": "steal a keeper from another player and put it in front of you", "effect_parameters": [0, 0, 0, 0, 0, 1, 0, 0, 0, 8, 0, 0, 8, 8, 1, 0]},
    "share_the_wealth": {"card_type": "ACTION", "card_effect": "gather all keepers in play shuffle them and redeal them evenly to all players", "effect_parameters": [0, 0, 0, 0, 9, 8, 0, 0, 6, 6, 0, 0, 8, 0, 0, 0]},
    "rules_reset": {"card_type": "ACTION", "card_effect": "discard all new rules currently in play returning the game to the basic rules", "effect_parameters": [0, 0, 0, 0, 0, 5, 0, 10, 0, 0, 0, 0, 9, 1, 0, 0]},
    "rock_paper_scissors_showdown": {"card_type": "ACTION", "card_effect": "challenge another player to rock paper scissors winner takes a keeper from the loser", "effect_parameters": [0, 0, 0, 0, 9, 6, 0, 0, 3, 5, 0, 0, 6, 4, 1, 0]},
    "random_tax": {"card_type": "ACTION", "card_effect": "take one card at random from each other player's hand", "effect_parameters": [4, 0, 0, 0, 7, 1, 1, 0, 0, 0, 0, 7, 5, 1, 4, 4]},
    "no_limits": {"card_type": "ACTION", "card_effect": "discard all hand limit and keeper limit rules currently in play", "effect_parameters": [0, 0, 0, 0, 0, 2, 1, 5, 0, 0, 0, 0, 5, 2, 0, 0]},
    "lets_simplify": {"card_type": "ACTION", "card_effect": "discard up to half of the new rules in play rounded up", "effect_parameters": [0, 0, 0, 0, 0, 3, 0, 7, 0, 0, 0, 0, 7, 8, 0, 0]},
    "lets_do_that_again": {"card_type": "ACTION", "card_effect": "search the discard pile choose an action or new rule card and play it immediately", "effect_parameters": [2, 0, 1, 0, 0, 2, 3, 0, 0, 0, 0, 0, 6, 9, 1, 4]},
    "jackpot": {"card_type": "ACTION", "card_effect": "draw three extra cards from the deck", "effect_parameters": [9, 0, 0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 4, 0, 9, 9]},
    "exchange_keepers": {"card_type": "ACTION", "card_effect": "swap one of your keepers with a keeper belonging to another player", "effect_parameters": [0, 0, 0, 0, 0, 2, 0, 0, 4, 4, 0, 0, 6, 7, 0, 0]},
    "empty_the_trash": {"card_type": "ACTION", "card_effect": "shuffle the discard pile back into the draw pile", "effect_parameters": [0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]},
    "discard_and_draw": {"card_type": "ACTION", "card_effect": "discard your entire hand and draw the same number of cards", "effect_parameters": [5, 0, 0, 0, 8, 3, 1, 0, 0, 0, 0, 0, 4, 1, 0, 5]},
    "take_another_turn": {"card_type": "ACTION", "card_effect": "after your current turn ends take an additional turn", "effect_parameters": [4, 0, 5, 0, 1, 0, 10, 0, 0, 0, 0, 0, 8, 4, 4, 4]},
    "rotate_hands": {"card_type": "ACTION", "card_effect": "all players pass their hands to the next player in turn order", "effect_parameters": [0, 0, 0, 0, 4, 6, 0, 0, 0, 0, 0, 8, 6, 0, 0, 0]},
    "everybody_gets_1": {"card_type": "ACTION", "card_effect": "set your hand aside draw enough cards so each player including you gets one and deal them out", "effect_parameters": [1, 4, 0, 0, 5, 2, 0, 0, 0, 0, 0, 0, 4, 0, 1, 1]},

    # Free Action Rules
    "mystery_play": {"card_type": "RULE", "RulesOptions": {"free_action": True}, "card_effect": "once per turn you may take the top card of the draw pile and play it immediately as if you had drawn it", "effect_parameters": [1, 1, 1, 1, 7, 3, 3, 0, 0, 0, 0, 0, 5, 0, 1, 1]},
    "swap_plays_for_draws": {"card_type": "RULE", "RulesOptions": {"free_action": True}, "card_effect": "instead of playing your remaining cards this turn discard them and draw the same number ending your turn", "effect_parameters": [4, 4, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 4, 3, 0, 4]},
    "get_on_with_it": {"card_type": "RULE", "RulesOptions": {"free_action": True}, "card_effect": "discard your entire hand draw three cards and end your turn immediately", "effect_parameters": [3, 3, 0, 0, 6, 3, 0, 0, 0, 0, 0, 0, 4, 1, 0, 3]},
    "recycling": {"card_type": "RULE", "RulesOptions": {"free_action": True}, "card_effect": "once per turn you may discard one of your keepers and draw three cards", "effect_parameters": [3, 3, 0, 0, 3, 1, 1, 0, 4, 0, 0, 0, 5, 0, 2, 2]},
    "goal_mill": {"card_type": "RULE", "RulesOptions": {"free_action": True}, "card_effect": "once per turn you may discard any number of goal cards from your hand and draw the same number of replacements", "effect_parameters": [3, 3, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 5, 4, 0, 4]},

    "play_all": {"card_type": "RULE", "RulesOptions": {"play": -1}, "card_effect": "you must play every card in your hand each turn", "effect_parameters": [3, 3, 10, 10, 5, 7, 8, 0, 0, 0, 0, 0, 9, 0, 0, 0]},
    "play_all_but_1": {"card_type": "RULE", "RulesOptions": {"play": -1}, "card_effect": "you must play every card in your hand each turn except one which you keep", "effect_parameters": [3, 3, 9, 9, 5, 5, 7, 0, 0, 0, 0, 0, 8, 2, 0, 0]},
    "no_hand_bonus": {"card_type": "RULE", "RulesOptions": {}, "card_effect": "if you have no cards in your hand at the start of your turn draw three extra cards before your normal draw", "effect_parameters": [4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 4, 4]},
    "party_bonus": {"card_type": "RULE", "RulesOptions": {}, "card_effect": "if you have the party keeper in play you may draw one extra card and play one extra card each turn", "effect_parameters": [3, 3, 3, 3, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0, 3, 3]},
    "poor_bonus": {"card_type": "RULE", "RulesOptions": {}, "card_effect": "if you have fewer keepers than every other player you may draw one extra card and play one extra card each turn", "effect_parameters": [3, 3, 3, 3, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0, 3, 3]},
    "rich_bonus": {"card_type": "RULE", "RulesOptions": {}, "card_effect": "if you have more keepers than every other player you may draw one extra card and play one extra card each turn", "effect_parameters": [3, 3, 3, 3, 0, 0, 4, 0, 0, 0, 0, 0, 5, 0, 3, 3]},
    "double_agenda": {"card_type": "RULE", "RulesOptions": {}, "card_effect": "two goal cards may be in play at the same time and a player wins if they meet either goal", "effect_parameters": [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0]},
    "first_play_random": {"card_type": "RULE", "RulesOptions": {}, "card_effect": "the first card you play each turn must be chosen at random from your hand", "effect_parameters": [0, 0, 0, 0, 9, 6, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0]},
    "inflation": {"card_type": "RULE", "RulesOptions": {}, "card_effect": "any number printed on a card is increased by one so draw two becomes draw three play one becomes play two and so on", "effect_parameters": [3, 3, 3, 3, 0, 1, 4, 0, 0, 0, 0, 0, 7, 0, 3, 3]},

}