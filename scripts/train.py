"""
Universal training script. Arguments:

- training script to run
- number of timesteps
- (optional) run name
- (optional) seed
- (optional) CUDA device

- TO ADD LATER (potentially)
- card list


Results:
- run the training obviously
- create a log in experiments. Structure:

experiments/
└── (SCRIPT-NAME)_(RUN-NAME)_(TIMESTAMP)/
    ├── logs/
    │   └── (tensorboard logfile)
    ├── models/
    │   └── (state_dicts of all models (including e.g optimizer) during training)
    ├── final/
    │   └── (stat_dicts of final models)
    └── run_metadata.json

run_metadata.txt: contains all arguments, seed used (even if not specified), git commit hash
    Python version, PyTorch version, NumPy version, CUDA version, cuDNN version

Resumability is limited: RNG states are not checkpointed. Could be extended to include this in the future
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists
from src.training.dqn.dqn import DQN
from src.training.ppo.ppo import PPO


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
        default="output.txt",
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
        help="CUDA device to use"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # seed generation
    master_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(2), "big")
    master_ss = np.random.SeedSequence(master_seed)
    game_ss, env_ss, training_ss, torch_ss = master_ss.spawn(4)

    # torch determinism (global)
    torch.manual_seed(torch_ss.generate_state(1)[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # create a training environment
    fluxx_game_2p = Game(2, card_lists.base_deck, disable_game_messages=True, seed=game_ss)
    env = FluxxEnv(fluxx_game_2p, 2, seed=env_ss)

    # create an instance of the correct training script
    training_script = None
    if args.script == "dqn":
        training_script = DQN(env, ["player_0", "player_1"], seed=training_ss)
    elif args.script == "ppo":
        training_script = PPO(env, ["player_0", "player_1"], seed=training_ss)
    else:
        logging.error("Unknown training script: {}".format(args.script))
        return 1

    # (optionally) set os.cuda device
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # run training script
    training_script.learn(args.timesteps)

    return 0

if __name__ == "__main__":
    sys.exit(main())