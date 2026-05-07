from dataclasses import dataclass

import numpy as np


@dataclass
class GameLogConfig:
    log_name: str
    agent_names: list[str]

@dataclass
class LearningCheckpoint:
    global_timestep: int
    run_name: str
    model_checkpoints_taken: int

@dataclass
class BufferEntry:
    decision_context: np.ndarray
    hand_ids: np.ndarray            # CHANGED: was hand: list[str]
    discard_ids: np.ndarray         # CHANGED: was discard: list[str]
    own_keeper_ids: np.ndarray      # CHANGED: was own_keepers: list[str]
    opp_keeper_ids: np.ndarray      # CHANGED: was opp_keepers: list[str]
    goal_ids: np.ndarray            # CHANGED: was goals: list[str]
    rules_ids: np.ndarray           # CHANGED: was rules: list[str]
    draw_pile_size: int
    opponent_hand_size: int
    action_mask: np.ndarray
    hand_size: int
    discard_pile_size: int
    own_keepers_in_play_count: int
    opponent_keepers_in_play_count: int
    goals_in_play_count: int
    rules_in_play_count: int