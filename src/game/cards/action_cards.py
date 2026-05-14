import math
from typing import TYPE_CHECKING

from random import Random

from src.agents.utils import agent_utils
# avoiding circular import
from src.game.FluxxEnums import CardType, GamePhase, GamePhaseType
from src.game.GameSchema import GameSchema
from src.game.game_messages import GameMessageType

if TYPE_CHECKING:
    pass


def activate_use_what_you_take(game_state: 'GameSchema', user_number: int, rng: Random):
    other_player_number = user_number ^ 1
    other_player = game_state.players[other_player_number]

    if len(other_player.hand) == 0:
        return

    card_to_take = rng.randint(0, len(other_player.hand) - 1)
    card_to_play = other_player.hand[card_to_take]
    del other_player.hand[card_to_take]
    game_state.game_message(f"<< (p{user_number}): ACTIVATING {card_to_play.name} >>", GameMessageType.SPECIAL_EFFECT)
    game_state.activate_card(user_number, card_to_play)


def activate_zap_a_card(game_state: 'GameSchema', user_number: int, rng: Random):
    if len(game_state.get_cards_in_play_by_name()) == 0:
        return
    game_state.stack.append(
        GamePhase(GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND, user_number, decisions_left=1)
    )


def activate_trash_a_new_rule(game_state: 'GameSchema', user_number: int, rng: Random):
    if len(game_state.rules) == 0:
        return
    game_state.stack.append(GamePhase(GamePhaseType.DISCARD_RULE_IN_PLAY, user_number, decisions_left=1))


def activate_trash_a_keeper(game_state: 'GameSchema', user_number: int, rng: Random):
    if len(game_state.get_all_keepers_by_name_flat()) == 0:
        game_state.game_message("No keepers to trash!", GameMessageType.NOTIFICATION)
        return
    game_state.stack.append(GamePhase(GamePhaseType.DISCARD_KEEPER_IN_PLAY, user_number, decisions_left=1))


def activate_trade_hands(game_state: 'GameSchema', user_number: int, rng: Random):
    user_player = game_state.players[user_number]

    temp = game_state.players[user_number ^ 1].hand
    game_state.players[user_number ^ 1].hand= user_player.hand
    user_player.hand = temp

    game_state.game_message(f"<< Player {user_number} traded hands with player {user_number ^ 1} >> ", GameMessageType.SPECIAL_EFFECT)


def activate_todays_special(game_state: 'GameSchema', user_number: int, rng: Random):
    latent_space = [game_state.get_card_from_draw_pile() for i in range(3+game_state.inflation())]
    latent_space = [l for l in latent_space if l is not None]
    if len(latent_space) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE,
        user_number,
        decisions_left=1+game_state.inflation(),
        latent_space=latent_space
    ))


def activate_draw_2_and_use_em(game_state: 'GameSchema', user_number: int, rng: Random):
    latent_space = [game_state.get_card_from_draw_pile() for i in range(2+game_state.inflation())]
    latent_space = [l for l in latent_space if l is not None]
    if len(latent_space) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE,
        user_number,
        decisions_left=2+game_state.inflation(),
        latent_space=latent_space
    ))


def activate_draw_3_play_2_of_them(game_state: 'GameSchema', user_number: int, rng: Random):
    latent_space = [game_state.get_card_from_draw_pile() for i in range(3 + game_state.inflation())]
    latent_space = [l for l in latent_space if l is not None]
    if len(latent_space) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE,
        user_number,
        decisions_left=2 + game_state.inflation(),
        latent_space=latent_space
    ))

def activate_steal_a_keeper(game_state: 'GameSchema', user_number: int, rng: Random):
    if len(game_state.players[user_number ^ 1].keepers) == 0:
        return
    game_state.stack.append(
        GamePhase(GamePhaseType.SELECT_KEEPER_TO_STEAL, user_number, decisions_left=1)
    )


def activate_share_the_wealth(game_state: 'GameSchema', user_number: int, rng: Random):
    all_keepers = [keeper for player in game_state.players for keeper in player.keepers]
    for player in game_state.players:
        player.keepers = []

    if len(all_keepers) == 0:
        return

    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT,
        user_number,
        decisions_left = int(math.ceil(len(all_keepers) / 2)),
        latent_space = all_keepers
    ))


def activate_rules_reset(game_state: 'GameSchema', user_number: int, rng: Random):
    for rule in game_state.rules:
        game_state.discard_pile.append(rule)

    game_state.rules = []


def activate_rock_paper_scissors_showdown(game_state: 'GameSchema', user_number: int, rng: Random):
    user_player = game_state.players[user_number]

    selected_player_number = user_number ^ 1
    selected_player = game_state.players[selected_player_number]

    coin = rng.randint(0, 1)
    if coin == 0:
        winner = user_player
        loser = selected_player
    else:
        winner = selected_player
        loser = user_player

    game_state.game_message(f"<< Player {winner.id} defeated {loser.id} in RPS! >>", GameMessageType.SPECIAL_EFFECT)

    winner.hand += loser.hand
    loser.hand = []


def activate_random_tax(game_state: 'GameSchema', user_number: int, rng: Random):
    user_player = game_state.players[user_number]

    for _ in range(game_state.inflation() + 1):
        for player_number, player in enumerate(game_state.players):
            if len(player.hand) == 0 or player_number == user_number:
                continue

            index_to_take = rng.randint(0, len(player.hand) - 1)
            card_to_take = player.hand[index_to_take]

            game_state.game_message(f"<< Player {player_number} took {card_to_take.name} from their hand >>", GameMessageType.SPECIAL_EFFECT)

            user_player.hand.append(card_to_take)
            del player.hand[index_to_take]


def activate_no_limits(game_state: 'GameSchema', user_number: int, rng: Random):
    new_rules = []
    for rule in game_state.rules:
        if rule.keeper_limit is None and rule.hand_limit is None:
            new_rules.append(rule)
        else:
            game_state.discard_pile.append(rule)

    game_state.rules = new_rules


def activate_lets_simplify(game_state: 'GameSchema', user_number: int, rng: Random):
    to_discard = (len(game_state.rules) + 1) // 2
    for i in range(to_discard):
        game_state.stack.append(GamePhase(GamePhaseType.DISCARD_RULE_IN_PLAY, user_number, decisions_left=to_discard - i))


def activate_lets_do_that_again(game_state: 'GameSchema', user_number: int, rng: Random):
    if len([card for card in game_state.discard_pile if card.card_type == CardType.ACTION or card.card_type == CardType.RULE]) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE,
        user_number,
        decisions_left=1
    ))


def activate_jackpot(game_state: 'GameSchema', user_number: int, rng: Random):
    user_player = game_state.players[user_number]

    for _ in range(3 + game_state.inflation()):
        game_state.draw(user_player)


def activate_exchange_keepers(game_state: 'GameSchema', user_number: int, rng: Random):
    if len(game_state.players[user_number].keepers) == 0 or len(game_state.players[user_number ^ 1].keepers) == 0:
        return
    game_state.stack.append(GamePhase(GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE, user_number, decisions_left=1))


def activate_empty_the_trash(game_state: 'GameSchema', user_number: int, rng: Random):
    game_state.draw_pile += game_state.discard_pile
    discard_pile_size = len(game_state.discard_pile)
    game_state.discard_pile = []
    rng.shuffle(game_state.draw_pile)
    game_state.game_message(f"<< Shuffled {discard_pile_size} cards from discard pile into deck! >>", GameMessageType.SPECIAL_EFFECT)


def activate_discard_and_draw(game_state: 'GameSchema', user_number: int, rng: Random):
    user_player = game_state.players[user_number]

    draw_amount = len(user_player.hand)

    game_state.discard_pile.extend(user_player.hand)
    user_player.hand = []

    for _ in range(draw_amount):
        game_state.draw(user_player)

def activate_everybody_gets_1(game_state: 'GameSchema', user_number: int, rng: Random):
    latent_space = [game_state.get_card_from_draw_pile() for i in range((1+game_state.inflation()) * len(game_state.players))]
    latent_space = [l for l in latent_space if l is not None]
    if len(latent_space) == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND,
        user_number,
        decisions_left=1+game_state.inflation(),
        latent_space=latent_space
    ))

def activate_take_another_turn(game_state: 'GameSchema', user_number: int, rng: Random):
    if game_state.extra_turns_taken < 2:
        game_state.extra_turns_taken += 1
        game_state.game_message("<< Extra turn! >>", GameMessageType.SPECIAL_EFFECT)
        game_state.extra_turn = True


def activate_rotate_hands(game_state: 'GameSchema', user_number: int, rng: Random):
    rotation_direction = 1
    n = len(game_state.players)

    current_index = 0
    temp = game_state.players[current_index].hand
    for _ in range(n):
        next_index = current_index + rotation_direction
        if next_index == -1:
            next_index = n - 1
        elif next_index == n:
            next_index = 0

        swap = game_state.players[next_index].hand

        game_state.players[next_index].hand = temp
        temp = swap

        current_index = (current_index + rotation_direction) % n
        if current_index == -1:
            current_index = n - 1
        elif current_index == n:
            current_index = 0

# ----------------
# Expanded set cards
# ----------------

def activate_pandoras_box(game_state: 'GameSchema', user_number: int, rng: Random):
    rules_played = 0
    seen_cards = set()
    while rules_played < 3:
        card = game_state.get_card_from_draw_pile()
        if card is None:
            game_state.game_message(
                "<< Pandora's Box: deck exhausted before 3 Rules played >>",
                GameMessageType.SPECIAL_EFFECT,
            )
            return
        if card.card_type == CardType.RULE:
            game_state.game_message(
                f"<< Pandora's Box revealed Rule: {card.name} >>",
                GameMessageType.SPECIAL_EFFECT,
            )
            game_state.stack.append(GamePhase(
                GamePhaseType.DEFERRED_PLAY_CARD,
                user_number,
                card=card
            ))
            rules_played += 1
        else:
            game_state.discard_pile.append(card)

        # If there are fewer than 3 rule cards in the draw pile, avoid entering an infinite loop
        if card in seen_cards:
            break
        seen_cards.add(card)


def activate_rewind(game_state: 'GameSchema', user_number: int, rng: Random):
    goals_in_discard = [c for c in game_state.discard_pile if c.card_type == CardType.GOAL]
    if len(goals_in_discard) == 0:
        return
    game_state.stack.append(
        GamePhase(GamePhaseType.PLAY_GOAL_FROM_DISCARD_PILE, user_number, decisions_left=1)
    )

def activate_robin_hood(game_state: 'GameSchema', user_number: int, rng: Random):
    user_player = game_state.players[user_number]
    opponent_number = user_number ^ 1
    opponent = game_state.players[opponent_number]

    user_count = len(user_player.keepers)
    opp_count = len(opponent.keepers)

    if user_count == 0 and opp_count == 0:
        return  # nothing to move

    if opp_count > user_count:
        game_state.stack.append(
            GamePhase(GamePhaseType.SELECT_KEEPER_TO_STEAL, user_number, decisions_left=1)
        )
        return

    game_state.stack.append(
        GamePhase(GamePhaseType.GIVE_KEEPER_TO_OPPONENT, user_number, decisions_left=1)
    )
    return


def activate_time_vortex(game_state: 'GameSchema', user_number: int, rng: Random):
    pool = []
    for p in game_state.players:
        pool.extend(p.hand)
        p.hand = []

    if len(pool) == 0:
        return

    rng.shuffle(pool)

    n = len(game_state.players)
    for i in range(len(pool)):
        recipient_number = (user_number + i) % n
        game_state.players[recipient_number].hand.append(pool[i])

    game_state.game_message(
        f"<< Time Vortex: redealt {len(pool)} cards across {n} players >>",
        GameMessageType.SPECIAL_EFFECT,
    )


def activate_gift_giveaway(game_state: 'GameSchema', user_number: int, rng: Random):
    if all(len(p.keepers) == 0 for p in game_state.players):
        return

    n = len(game_state.players)
    for offset in range(1, n + 1):
        giver_number = (user_number + offset) % n
        if len(game_state.players[giver_number].keepers) == 0:
            continue
        game_state.stack.append(GamePhase(
            GamePhaseType.GIVE_KEEPER_TO_OPPONENT,
            giver_number,
            decisions_left=1,
        ))


def activate_dig_up_the_past(game_state: 'GameSchema', user_number: int, rng: Random):
    keepers_in_discard = [c for c in game_state.discard_pile if c.card_type == CardType.KEEPER]
    if len(keepers_in_discard) == 0:
        return
    game_state.stack.append(
        GamePhase(GamePhaseType.SELECT_KEEPER_FROM_DISCARD_PILE, user_number, decisions_left=1)
    )


def activate_space_jackpot(game_state: 'GameSchema', user_number: int, rng: Random):
    user_player = game_state.players[user_number]
    for _ in range(5 + game_state.inflation()):
        game_state.draw(user_player)

    discards_required = min(2, len(user_player.hand))
    if discards_required == 0:
        return
    game_state.stack.append(GamePhase(
        GamePhaseType.DISCARD_CARD_FROM_HAND,
        user_number,
        decisions_left=discards_required,
    ))


def activate_supernova(game_state: 'GameSchema', user_number: int, rng: Random):
    spared = {"dog", "cat", "monkey"}

    for player in game_state.players:
        kept = []
        for keeper in player.keepers:
            if keeper.name in spared:
                kept.append(keeper)
            else:
                game_state.discard_pile.append(keeper)
        player.keepers = kept

    game_state.draw_pile += game_state.discard_pile
    discard_size = len(game_state.discard_pile)
    game_state.discard_pile = []
    rng.shuffle(game_state.draw_pile)

    game_state.game_message(
        f"<< Supernova: shuffled {discard_size} cards back into deck. "
        f"Spared: {sorted(spared) if spared else 'none'} >>",
        GameMessageType.SPECIAL_EFFECT,
    )

def activate_destroy_all_keepers(game_state: 'GameSchema', user_number: int, rng: Random):
    total = 0
    for player in game_state.players:
        for keeper in player.keepers:
            game_state.discard_pile.append(keeper)
            total += 1
        player.keepers = []
    if total > 0:
        game_state.game_message(
            f"<< Destroy All Keepers: discarded {total} Keepers >>",
            GameMessageType.SPECIAL_EFFECT,
        )


def activate_roll_for_it(game_state: 'GameSchema', user_number: int, rng: Random):
    roll = rng.randint(1, 6)
    game_state.game_message(
        f"<< Roll For It! rolled a {roll} >>", GameMessageType.SPECIAL_EFFECT
    )
    user_player = game_state.players[user_number]

    if roll == 1:
        latent_space = [game_state.get_card_from_draw_pile()]
        latent_space = [c for c in latent_space if c is not None]
        if len(latent_space) == 0:
            return
        game_state.stack.append(GamePhase(
            GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE,
            user_number,
            decisions_left=1,
            latent_space=latent_space,
        ))
    elif roll == 2:
        for _ in range(2):
            game_state.draw(user_player)
    elif roll == 3:
        for _ in range(3):
            game_state.draw(user_player)
    elif roll == 4:
        activate_trash_a_new_rule(game_state, user_number, rng)
    elif roll == 5:
        activate_steal_a_keeper(game_state, user_number, rng)
    elif roll == 6:
        activate_take_another_turn(game_state, user_number, rng)


def activate_close_enough(game_state: 'GameSchema', user_number: int, rng: Random):
    for current_goal in game_state.goals:
        required = set(getattr(current_goal, "required_keepers", []) or [])
        if len(required) == 0:
            return

        user_keeper_names = {k.name for k in game_state.players[user_number].keepers}
        if required & user_keeper_names:
            game_state.game_message(
                f"<< Close Enough! Player {user_number} wins! >>",
                GameMessageType.SPECIAL_EFFECT,
            )
            game_state.winner = user_number


def activate_psychic_paper(game_state: 'GameSchema', user_number: int, rng: Random):
    opponent_number = user_number ^ 1
    opponent = game_state.players[opponent_number]
    actions_in_hand = [c for c in opponent.hand if c.card_type == CardType.ACTION]
    if len(actions_in_hand) == 0:
        return
    # New phase: SELECT_ACTION_FROM_OPPONENT_HAND — resolver removes the
    # chosen Action from opponent.hand and routes it through activate_card
    # under user_number, so the user (the chooser) resolves its effect —
    # matching how use_what_you_take currently activates the stolen card.
    game_state.stack.append(GamePhase(
        GamePhaseType.SELECT_ACTION_FROM_OPPONENT_HAND,
        user_number,
        decisions_left=1,
        target_player_number=opponent_number,
    ))


def activate_rough_seas(game_state: 'GameSchema', user_number: int, rng: Random):
    for p_num, player in enumerate(game_state.players):
        for _ in range(max(0, len(player.hand) - 3)):
            game_state.stack.append(GamePhase(
                GamePhaseType.DISCARD_CARD_FROM_HAND, p_num, decisions_left=1,
            ))
        for _ in range(max(0, len(player.keepers) - 2)):
            # Reuse the existing DISCARD_KEEPER_IN_PLAY phase, but constrained
            # to the player's own keepers via phase metadata.
            game_state.stack.append(GamePhase(
                GamePhaseType.DISCARD_KEEPER, p_num,
                decisions_left=1,
            ))


def activate_brain_drain(game_state: 'GameSchema', user_number: int, rng: Random):
    for p_num, player in enumerate(game_state.players):
        if len(player.hand) == 0:
            continue
        game_state.stack.append(GamePhase(
            GamePhaseType.DISCARD_CARD_FROM_HAND, p_num, decisions_left=1,
        ))

def activate_oops(game_state: 'GameSchema', user_number: int, rng: Random):
    user_player = game_state.players[user_number]
    if len(user_player.hand) == 0:
        return
    discarded = len(user_player.hand)
    game_state.discard_pile.extend(user_player.hand)
    user_player.hand = []
    game_state.game_message(
        f"<< Oops! Player {user_number} discarded {discarded} cards >>",
        GameMessageType.SPECIAL_EFFECT,
    )

ACTION_FUNCTIONS = {
    "use_what_you_take": activate_use_what_you_take,
    "zap_a_card": activate_zap_a_card,
    "trash_a_new_rule": activate_trash_a_new_rule,
    "trash_a_keeper": activate_trash_a_keeper,
    "trade_hands": activate_trade_hands,
    "todays_special": activate_todays_special,
    "draw_2_and_use_em": activate_draw_2_and_use_em,
    "draw_3_play_2_of_them": activate_draw_3_play_2_of_them,
    "steal_a_keeper": activate_steal_a_keeper,
    "share_the_wealth": activate_share_the_wealth,
    "rules_reset": activate_rules_reset,
    "rock_paper_scissors_showdown": activate_rock_paper_scissors_showdown,
    "random_tax": activate_random_tax,
    "no_limits": activate_no_limits,
    "lets_simplify": activate_lets_simplify,
    "lets_do_that_again": activate_lets_do_that_again,
    "jackpot": activate_jackpot,
    "exchange_keepers": activate_exchange_keepers,
    "empty_the_trash": activate_empty_the_trash,
    "discard_and_draw": activate_discard_and_draw,
    "everybody_gets_1": activate_everybody_gets_1,
    "take_another_turn": activate_take_another_turn,
    "rotate_hands": activate_rotate_hands,
    "pandoras_box": activate_pandoras_box,
    "rewind": activate_rewind,
    "robin_hood": activate_robin_hood,
    "time_vortex": activate_time_vortex,
    "gift_giveaway": activate_gift_giveaway,
    "dig_up_the_past": activate_dig_up_the_past,
    "space_jackpot": activate_space_jackpot,
    "supernova": activate_supernova,
    "destroy_all_keepers": activate_destroy_all_keepers,
    "oops": activate_oops,
    "close_enough": activate_close_enough,
    "roll_for_it": activate_roll_for_it,
    "brain_drain": activate_brain_drain,
    "rough_seas": activate_rough_seas,
}


def activate_action(action_name: str, game_state: 'GameSchema', user_number: int, rng: Random):
    action_function = ACTION_FUNCTIONS.get(action_name)
    if action_function is None:
        raise Exception(f"Error: Action [[{action_name}]] not implemented")

    action_function(game_state, user_number, rng)