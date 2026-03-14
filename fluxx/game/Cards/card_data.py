CARD_DATA = {
    # Rules - Draw/Play/Limits
    "hand_limit_2": {
        "card_type": "RULE",
        "RulesOptions": {"hand_limit": 2}
    },
    "hand_limit_1": {
        "card_type": "RULE",
        "RulesOptions": {"hand_limit": 1}
    },
    "hand_limit_0": {
        "card_type": "RULE",
        "RulesOptions": {"hand_limit": 0}
    },
    "keeper_limit_4": {
        "card_type": "RULE",
        "RulesOptions": {"keeper_limit": 4}
    },
    "keeper_limit_3": {
        "card_type": "RULE",
        "RulesOptions": {"keeper_limit": 3}
    },
    "keeper_limit_2": {
        "card_type": "RULE",
        "RulesOptions": {"keeper_limit": 2}
    },
    "draw_5": {
        "card_type": "RULE",
        "RulesOptions": {"draw": 5}
    },
    "draw_4": {
        "card_type": "RULE",
        "RulesOptions": {"draw": 4}
    },
    "draw_2": {
        "card_type": "RULE",
        "RulesOptions": {"draw": 2}
    },
    "play_2": {
        "card_type": "RULE",
        "RulesOptions": {"play": 2}
    },
    "draw_3": {
        "card_type": "RULE",
        "RulesOptions": {"draw": 3}
    },
    "play_4": {
        "card_type": "RULE",
        "RulesOptions": {"play": 4}
    },
    "play_3": {
        "card_type": "RULE",
        "RulesOptions": {"play": 3}
    },

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
    "use_what_you_take": {"card_type": "ACTION"},
    "zap_a_card": {"card_type": "ACTION"},
    "trash_a_new_rule": {"card_type": "ACTION"},
    "trash_a_keeper": {"card_type": "ACTION"},
    "trade_hands": {"card_type": "ACTION"},
    "todays_special": {"card_type": "ACTION"},
    "draw_2_and_use_em": {"card_type": "ACTION"},
    "draw_3_play_2_of_them": {"card_type": "ACTION"},
    "steal_a_keeper": {"card_type": "ACTION"},
    "share_the_wealth": {"card_type": "ACTION"},
    "rules_reset": {"card_type": "ACTION"},
    "rock_paper_scissors_showdown": {"card_type": "ACTION"},
    "random_tax": {"card_type": "ACTION"},
    "no_limits": {"card_type": "ACTION"},
    "lets_simplify": {"card_type": "ACTION"},
    "lets_do_that_again": {"card_type": "ACTION"},
    "jackpot": {"card_type": "ACTION"},
    "exchange_keepers": {"card_type": "ACTION"},
    "empty_the_trash": {"card_type": "ACTION"},
    "discard_and_draw": {"card_type": "ACTION"},
    "take_another_turn": {"card_type": "ACTION"},
    "rotate_hands": {"card_type": "ACTION"},
    "everybody_gets_1": {"card_type": "ACTION"},

    # Free Action Rules
    "mystery_play": {
        "card_type": "RULE",
        "RulesOptions": {"free_action": True}
    },
    "swap_plays_for_draws": {
        "card_type": "RULE",
        "RulesOptions": {"free_action": True}
    },
    "get_on_with_it": {
        "card_type": "RULE",
        "RulesOptions": {"free_action": True}
    },
    "recycling": {
        "card_type": "RULE",
        "RulesOptions": {"free_action": True}
    },
    "goal_mill": {
        "card_type": "RULE",
        "RulesOptions": {"free_action": True}
    },
    "play_all": {
        "card_type": "RULE",
        "RulesOptions": {"play": -1}
    },
    "play_all_but_1": {
        "card_type": "RULE",
        "RulesOptions": {"play": -1}
    },
    "no_hand_bonus": {
        "card_type": "RULE",
        "RulesOptions": {}
    },
    "party_bonus": {
        "card_type": "RULE",
        "RulesOptions": {}
    },
    "poor_bonus": {
        "card_type": "RULE",
        "RulesOptions": {}
    },
    "rich_bonus": {
        "card_type": "RULE",
        "RulesOptions": {}
    },
    "double_agenda": {
        "card_type": "RULE",
        "RulesOptions": {}
    },
    "first_play_random": {
        "card_type": "RULE",
        "RulesOptions": {}
    },
    "inflation": {
        "card_type": "RULE",
        "RulesOptions": {}
    },
}