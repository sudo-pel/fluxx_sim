from fluxx.game.FluxxEnums import GameState


def index_of_card(list_of_cards, card_name):
    for i, card in enumerate(list_of_cards):
        if card.name == card_name:
            return i
    return -1

def print_game_state(state: GameState) -> None:
    num_players = len(state.hands)

    print("=" * 60)
    print("GAME STATE")
    print("=" * 60)

    # Rules
    print(f"\nRules: {', '.join(state.rules) if state.rules else '(none)'}")

    # Goals
    print(f"Goals: {', '.join(state.goals) if state.goals else '(none)'}")

    # Per-player info
    print(f"\nPlayers ({num_players}):")
    for i in range(num_players):
        hand = state.hands[i]
        keepers = state.keepers[i]
        extras = []
        if state.cards_played and i < len(state.cards_played):
            extras.append(f"played={state.cards_played[i]}")
        if state.plays_remaining and i < len(state.plays_remaining):
            extras.append(f"plays_left={state.plays_remaining[i]}")
        if state.cards_drawn and i < len(state.cards_drawn):
            extras.append(f"drawn={state.cards_drawn[i]}")
        extra_str = f"  ({', '.join(extras)})" if extras else ""
        print(f"  Player {i}{extra_str}:")
        print(f"    Hand:    [{', '.join(hand)}]")
        print(f"    Keepers: [{', '.join(keepers)}]")

    # Piles
    print(f"\nDraw pile ({len(state.draw_pile)}): [{', '.join(state.draw_pile)}]")
    print(f"Discard pile ({len(state.discard_pile)}): [{', '.join(state.discard_pile)}]")

    # Stack
    print(f"\nStack ({len(state.stack)}):")
    for j, phase in enumerate(state.stack):
        print(f"  [{j}] {phase}")

    # Optional fields
    if state.game_over is not None:
        print(f"\nGame over: {state.game_over}")
    if state.starting_player is not None:
        print(f"Starting player: {state.starting_player}")

    print("=" * 60)