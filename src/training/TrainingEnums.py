from dataclasses import dataclass

@dataclass
class GameLogConfig:
    log_name: str
    agent_names: list[str]

@dataclass
class LearningCheckpoint:
    global_timestep: int
    run_name: str
    model_checkpoints_taken: int
