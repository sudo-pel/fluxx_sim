"""
Results:
- run the training obviously
- create a log in experiments. Structure:

experiments/
└── (SCRIPT-NAME)_(RUN-NAME)_(TIMESTAMP)/
    ├── logs/
    │   └── (tensorboard logfiles)
    ├── models/
    │   └── (state_dicts of models generated during training)
    ├── final/
    │   └── (stat_dicts of final models)
    └── run_metadata.json

run_metadata.txt: contains all arguments, seed used (even if not specified), git commit hash
    Python version, PyTorch version, NumPy version, CUDA version, cuDNN version
"""

import argparse
import json
import torch
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from src.agents.utils.card_embeddings import load_embedding_table
from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists
from src.training.TrainingEnums import LearningCheckpoint
from src.training.dqn.dqn import DQN
from src.training.dqn.dqn_general import DQNGeneralized
from src.training.ppo.ppo import PPO
from src.training.ppo.ppo_general import PPOGeneralized
from src.training.ppo.ppo_general_with_reward_shaping import PPOGeneralizedRewardShaped


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a specified training script over a number of timesteps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "script",
        help="Training script to run",
    )
    parser.add_argument(
        "timesteps",
        type=int,
        help="Number of episode timesteps",
    )
    parser.add_argument(
        "-r", "--run-name",
        type=str,
        default=None,
        help="Run name (for log directory name)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        help="CUDA device to use: 'cuda:0', 'cuda:1', ... , 'cpu'"
    )
    parser.add_argument(
        "-cname", "--checkpoint-name",
        type=str,
        default=None,
        help="Run name of a checkpoint. Must also pass -ct and -cpt"
    )
    parser.add_argument(
        "-ct", "--checkpoint-timestep",
        type=int,
        default=None,
        help="Timestep of a checkpoint. Must also pass -cname and -cpt"
    )
    parser.add_argument(
        "-cpt", "--checkpoint-points-taken",
        type=int,
        default=None,
        help="Number of points taken from checkpointed run up to this point. Must also pass -cname and -ct"
    )
    return parser.parse_args()

def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD"], stderr=subprocess.DEVNULL
        ).decode()
        return {"commit": commit, "diff": diff, "dirty": bool(diff.strip())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "diff": None, "dirty": None}

def write_metadata(run_dir: Path, args: argparse.Namespace, master_seed: int) -> None:
    """
    Writes a metadata.json file to the directory run_dir
    """
    cuda_version = torch.version.cuda if torch.cuda.is_available() else None
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None

    metadata = {
        "arguments": vars(args),
        "seed": master_seed,
        "seed_was_specified": args.seed is not None,
        "timestamp": datetime.now().isoformat(),
        "versions": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "cuda": cuda_version,
            "cudnn": cudnn_version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "git": git_info(),
    }

    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    args = parse_args()

    # seed generation
    master_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(2), "big")
    master_ss = np.random.SeedSequence(master_seed)
    game_ss, env_ss, training_ss, torch_ss = master_ss.spawn(4)

    # set pytorch device
    device = torch.device(args.device) if args.device else (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # torch determinism (global)
    torch.manual_seed(torch_ss.generate_state(1)[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # create a training environment
    fluxx_game_2p = Game(2, card_lists.base_deck, disable_game_messages=True, seed=game_ss)
    env = FluxxEnv(fluxx_game_2p, 2, seed=env_ss)

    # create run name
    run_name = args.run_name + "_" if args.run_name is not None else ""
    run_name = f"{args.script}_{run_name}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # create experiment directory structure
    run_dir = Path("experiments") / run_name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "final").mkdir(parents=True, exist_ok=True)

    # write metadata before training kicks off
    write_metadata(run_dir, args, master_seed)

    # checkpointing
    if args.checkpoint_name is not None or args.checkpoint_timestep is not None or args.checkpoint_points_taken is not None:
        if args.checkpoint_name is None or args.checkpoint_timestep is None or args.checkpoint_points_taken is None:
            logging.error("Must pass all three of -cname, -ct and -cpt when using checkpointing")
            return 1
        checkpoint = LearningCheckpoint(
            args.checkpoint_timestep,
            args.checkpoint_name,
            args.checkpoint_points_taken,
        )
    else:
        checkpoint = None

    # create an instance of the correct training script
    if args.script == "dqn":
        training_script = DQN(env, ["player_0", "player_1"], run_name, seed=training_ss, device=device)
    elif args.script == "ppo":
        training_script = PPO(env, ["player_0", "player_1"], run_name, seed=training_ss, device=device, from_checkpoint=checkpoint)
    elif args.script == "ppo_general":
        load_embedding_table("base_deck")
        training_script = PPOGeneralized(env, ["player_0", "player_1"], run_name, seed=training_ss, device=device, from_checkpoint=checkpoint)
    elif args.script == "ppo_general_with_reward_shaping":
        load_embedding_table("base_deck")
        training_script = PPOGeneralizedRewardShaped(env, ["player_0", "player_1"], run_name, seed=training_ss, device=device, from_checkpoint=checkpoint)
    elif args.script == "dqn_general":
        load_embedding_table("base_deck")
        training_script = DQNGeneralized(env, ["player_0", "player_1"], run_name, seed=training_ss, device=device)
    else:
        logging.error("Unknown training script: {}".format(args.script))
        return 1

    # run training script
    training_script.learn(args.timesteps)

    return 0

if __name__ == "__main__":
    sys.exit(main())