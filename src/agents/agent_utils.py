from typing import Optional

from src.game.cards.card_data import CARD_DATA
from src.game.FluxxEnums import GameState, GameConfig, GamePhaseType, DecisionEncodingType, OnCompleteBehaviour
import numpy.typing as npt
import numpy as np

def populate_card_vector(card_list: list[str], to_populate: list[str]) -> npt.NDArray[np.int8]:
    vector = np.zeros(len(card_list), dtype=np.int8)

    card_to_index = {card:i for i, card in enumerate(card_list)}

    # TODO: vectorise this
    for card in to_populate:
        vector[card_to_index[card]] = 1
    return vector

def convert_decision_encoding(decision_encoding: list[DecisionEncodingType], decisions_left: int, counter: Optional[int] = 0, on_complete: Optional[OnCompleteBehaviour]=OnCompleteBehaviour.DRAW) -> npt.NDArray[np.int8]:
    decision_context_vector = np.zeros(19, dtype=np.int8)
    for d in decision_encoding:
        decision_context_vector[d.value] = 1
    decision_context_vector[16] = decisions_left

    if counter is None:
        decision_context_vector[17] = 0
    else:
        decision_context_vector[17] = counter

    if on_complete == OnCompleteBehaviour.DRAW:
        decision_context_vector[18] = 1

    return decision_context_vector

decision_context_vectors: dict[GamePhaseType, list[DecisionEncodingType]] = {
        GamePhaseType.DISCARD_CARD_FROM_HAND: [DecisionEncodingType.PLACE_DISCARD_PILE,
                                               DecisionEncodingType.REMAIN_PLAYER_HAND],
        GamePhaseType.PLAY_CARD_FOR_TURN: [DecisionEncodingType.PLAY, DecisionEncodingType.REMAIN_PLAYER_HAND],
        GamePhaseType.DISCARD_KEEPER: [DecisionEncodingType.PLACE_DISCARD_PILE,
                                       DecisionEncodingType.REMAIN_PLAYER_KEEPERS],
        GamePhaseType.DISCARD_RULE_IN_PLAY: [DecisionEncodingType.PLACE_DISCARD_PILE,
                                             DecisionEncodingType.REMAIN_IN_PLAY],
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE: [DecisionEncodingType.PLAY, DecisionEncodingType.PLACE_DISCARD_PILE],
        GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND: [DecisionEncodingType.PLACE_PLAYER_HAND,
                                                 DecisionEncodingType.REMAIN_IN_PLAY],
        GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND: [DecisionEncodingType.PLACE_PLAYER_HAND,
                                                                DecisionEncodingType.REMAIN_OPPONENT_HAND],
        GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE: [DecisionEncodingType.PLAY,
                                                              DecisionEncodingType.REMAIN_DISCARD_PILE],
        GamePhaseType.DISCARD_KEEPER_IN_PLAY: [DecisionEncodingType.PLACE_DISCARD_PILE,
                                               DecisionEncodingType.REMAIN_IN_PLAY],
        GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT: [DecisionEncodingType.PLAY,
                                                                             DecisionEncodingType.PLAY_FOR_OPPONENT],
        GamePhaseType.SELECT_KEEPER_TO_STEAL: [DecisionEncodingType.PLACE_PLAYER_KEEPERS,
                                               DecisionEncodingType.REMAIN_OPPONENT_KEEPERS],
        GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE: [DecisionEncodingType.PLACE_PLAYER_KEEPERS,
                                                            DecisionEncodingType.REMAIN_OPPONENT_KEEPERS],
        GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE: [DecisionEncodingType.PLACE_OPPONENT_KEEPERS,
                                                          DecisionEncodingType.REMAIN_PLAYER_KEEPERS],
        GamePhaseType.ACTIVATE_FREE_ACTION: [DecisionEncodingType.PLAY, DecisionEncodingType.REMAIN_IN_PLAY],
        GamePhaseType.DISCARD_OWN_KEEPER_IN_PLAY: [DecisionEncodingType.PLACE_DISCARD_PILE,
                                                   DecisionEncodingType.REMAIN_IN_PLAY],
        GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND: [DecisionEncodingType.PLACE_DISCARD_PILE,
                                                         DecisionEncodingType.REMAIN_PLAYER_HAND],
        GamePhaseType.DISCARD_GOAL_IN_PLAY: [DecisionEncodingType.PLACE_DISCARD_PILE,
                                             DecisionEncodingType.REMAIN_IN_PLAY],
        GamePhaseType.GAME_OVER: []
}

def observe_hot_encoded(agent, game_state: GameState, game_config: GameConfig):
    current_phase = game_state.stack[-1]

    # ----
    # OBSERVATION
    # ----

    decisions_left = current_phase.decisions_left

    decision_context_vector = convert_decision_encoding(decision_context_vectors[current_phase.type],
                                                        decisions_left, current_phase.counter,
                                                        current_phase.on_complete)

    # Get all keepers in play
    keepers_in_play = game_state.keepers
    keeper_vectors = [populate_card_vector(game_config.card_list, keeper_list) for keeper_list in keepers_in_play]
    agent_keeper_vector = keeper_vectors[agent.player_number]
    other_keeper_vectors = keeper_vectors[:agent.player_number] + keeper_vectors[
        agent.player_number + 1:]

    # Get cards in hand
    cards_in_hand = game_state.hands[agent.player_number]
    cards_in_hand_vector = populate_card_vector(game_config.card_list, cards_in_hand)

    # Get goals in play
    goals_in_play = game_state.goals
    goals_in_play_vector = populate_card_vector(game_config.card_list, goals_in_play)

    # Get rules in play
    rules_in_play = game_state.rules
    rules_in_play_vector = populate_card_vector(game_config.card_list, rules_in_play)

    # Get discard pile
    discard_pile = game_state.discard_pile
    discard_pile_vector = populate_card_vector(game_config.card_list, discard_pile)

    # Get draw pile size
    draw_pile_size = [len(game_state.draw_pile)]

    # Get opponent hand size
    # TODO: how to encode for variable opponent count? Is this worth doing?
    opponent_hand_sizes = []
    for i in range(game_config.player_count):
        if i != agent.player_number:
            opponent_hand_sizes.append(len(game_state.hands[i]))

    observation = np.concatenate(
        (decision_context_vector, cards_in_hand_vector, agent_keeper_vector, *other_keeper_vectors,
         goals_in_play_vector, rules_in_play_vector, discard_pile_vector, draw_pile_size, opponent_hand_sizes))

    """
    assert len(observation) == agent.observation_space["observation"].shape[0], \
        f"Observation size mismatch: built {len(observation)}, expected {agent.observation_space['observation'].shape[0]}"
    """

    # ----
    # ACTION MASK
    # ----

    action_mask = np.zeros(len(game_config.card_list), dtype=np.int8)
    card_to_index = {card: i for i, card in enumerate(game_config.card_list)}

    no_free_action_legal = False

    # TODO: Mask *in* legal plays (cards in hand, keepers owned) and then return the concatenation of all
    if current_phase.type == GamePhaseType.PLAY_CARD_FOR_TURN:
        action_mask = cards_in_hand_vector
    elif current_phase.type == GamePhaseType.DISCARD_CARD_FROM_HAND:
        action_mask = cards_in_hand_vector
    elif current_phase.type == GamePhaseType.DISCARD_KEEPER:
        action_mask = agent_keeper_vector
    elif current_phase.type == GamePhaseType.DISCARD_RULE_IN_PLAY:
        action_mask = rules_in_play_vector
    elif current_phase.type == GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE:
        action_mask = populate_card_vector(game_config.card_list, [card.name for card in current_phase.latent_space])
    elif current_phase.type == GamePhaseType.ADD_CARD_IN_PLAY_TO_HAND:
        action_mask = np.bitwise_or.reduce((*keeper_vectors, rules_in_play_vector, goals_in_play_vector))
    elif current_phase.type == GamePhaseType.SHARE_CARDS_FROM_LATENT_SPACE_INTO_HAND:
        action_mask = populate_card_vector(game_config.card_list, [card.name for card in current_phase.latent_space])
    elif current_phase.type == GamePhaseType.PLAY_ACTION_OR_RULE_FROM_DISCARD_PILE:
        action_mask = populate_card_vector(game_config.card_list,
            [card for card in game_state.discard_pile if
             CARD_DATA[card]["card_type"] == "RULE" or CARD_DATA[card]["card_type"] == "ACTION"]
        )
    elif current_phase.type == GamePhaseType.DISCARD_KEEPER_IN_PLAY:
        action_mask = np.bitwise_or.reduce(keeper_vectors)
    elif current_phase.type == GamePhaseType.PLAY_CARD_FROM_LATENT_SPACE_OTHERS_PLAY_FOR_OPPONENT:
        action_mask = populate_card_vector(game_config.card_list, [card.name for card in current_phase.latent_space])
    elif current_phase.type == GamePhaseType.SELECT_KEEPER_TO_STEAL:
        action_mask = np.bitwise_or.reduce(other_keeper_vectors)
    elif current_phase.type == GamePhaseType.SELECT_OPPONENT_KEEPER_FOR_EXCHANGE:
        action_mask = np.bitwise_or.reduce(other_keeper_vectors)
    elif current_phase.type == GamePhaseType.SELECT_PLAYER_KEEPER_FOR_EXCHANGE:
        action_mask = agent_keeper_vector
        action_mask[card_to_index[
            current_phase.labelled_card.name]] = 0  # mask out the keeper that was stolen from the opponent in this exchange
    elif current_phase.type == GamePhaseType.ACTIVATE_FREE_ACTION:
        action_mask = populate_card_vector(game_config.card_list,
            [free_action_name for free_action_name in game_state.available_free_actions])
        no_free_action_legal = True
    elif current_phase.type == GamePhaseType.DISCARD_OWN_KEEPER_IN_PLAY:
        action_mask = agent_keeper_vector
    elif current_phase.type == GamePhaseType.DISCARD_VARIABLE_CARDS_FROM_HAND:
        valid_cards_in_hand = []
        for card in game_state.hands[agent.player_number]:
            if CARD_DATA[card]["card_type"] in [t.name for t in current_phase.card_types]:
                valid_cards_in_hand.append(card)
        action_mask = populate_card_vector(game_config.card_list, valid_cards_in_hand)
        no_free_action_legal = True
    elif current_phase.type == GamePhaseType.DISCARD_GOAL_IN_PLAY:
        action_mask = goals_in_play_vector
    else:
        raise Exception(f"Invalid game phase type: {current_phase.type}")

    action_mask = np.append(action_mask, 0)
    if no_free_action_legal:
        action_mask[-1] = 1

    return {
        "observation": observation,
        "action_mask": action_mask
    }