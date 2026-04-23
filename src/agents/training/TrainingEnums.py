from dataclasses import dataclass

@dataclass
class LearningCheckpoint:
    global_timestep: int
    run_name: str
    model_checkpoints_taken: int
