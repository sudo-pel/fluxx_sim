from pathlib import Path

import torch

from src.agents.PPOAgentGeneralized import PPOAgentGeneralized
from src.agents.RandomAgent import RandomAgent
from src.agents.card_embeddings import generate_embedding_table
from src.env.FluxxEnv import FluxxEnv
from src.game.cards import card_lists
from src.game.Game import Game
from src.game.game_states import puzzle_d

"""

A quickly modifiable script for running random tests.

"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main(one_turn_win_simple_fluxx=None):
    two_player_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True, force_game_state=puzzle_d)
    env = FluxxEnv(two_player_fluxx, 2, render_mode="human")

    generate_embedding_table(card_lists.base_deck)

    agents = {
        "player_0": PPOAgentGeneralized(env.game.game_config, 0),
        "player_1": RandomAgent(env.game.game_config, 1)
    }

    agents["player_0"].policy_network.load_state_dict(
        torch.load(f"{PROJECT_ROOT}/final_experiments/ppo_general_2026-05-02_09-11-14/final/final_model_50006336.pt"),
        strict=False)
    agents["player_0"].policy_network.eval()

    victories = {
        "player_0": 0,
        "player_1": 0
    }

    GAME_COUNT = 200

    for i in range(5):

        round_victories = {
            "player_0": 0,
            "player_1": 0,
            "draws": 0
        }

        for j in range(GAME_COUNT):
            env.reset()
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()

                if env.game.turn_count > 3:
                    #print(f"turn limit reached for game {i*2000+j}")
                    break

                if termination or truncation:
                    action = None
                else:
                    action, _, _ = agents[agent].act(observation)
                    action = env.decode_action(action)
                    #input("Press enter to continue...")

                #print(f"Agent {agent} took action {action}")
                env.step(action)
                #print("\n".join(printout_state(env.get_player_number(agent), env.game.get_game_state())))
                #print(env.game.stack)

            #print(env.game.winner)
            if env.game.winner is None:
                round_victories["draws"] += 1
            else:
                round_victories[f"player_{env.game.winner}"] += 1
            env.close()


        victories["player_0"] += round_victories["player_0"]
        victories["player_1"] += round_victories["player_1"]

        print(f"Round Victories: {round_victories}")
        print(f"Victories: {victories}")

if __name__ == '__main__':
    main()