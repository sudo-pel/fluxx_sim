import torch

from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.agents.RandomAgent import RandomAgent
from src.env.FluxxEnv import FluxxEnv
from src.game.cards import card_lists
from src.game.Game import Game
from src.game.game_states import two_player_p0_one_turn_win, two_player_p0_two_turn_win


def main(one_turn_win_simple_fluxx=None):
    two_player_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True)
    two_player_simple_fluxx = Game(2, card_lists.simple_fluxx_deck, disable_game_messages=True)
    one_turn_win_simple_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True, force_game_state=two_player_p0_one_turn_win)
    two_turn_win_simple_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True, force_game_state=two_player_p0_two_turn_win)

    action_testing = Game(2, card_lists.for_action_testing, disable_game_messages=True)


    env = FluxxEnv(two_player_fluxx, 2, render_mode="human")

    actor = PPOAgent(env.game.game_config, 1)  # same architecture
    actor.policy_network.load_state_dict(torch.load("../experiments/ppo_20-04-2026_10:40:57/models/final/actor.pt"))
    actor.policy_network.eval()

    agents = {
        "player_0": RandomAgent(env.game.game_config, 0),
        "player_1": actor
    }

    victories = {
        "player_0": 0,
        "player_1": 0
    }

    GAME_COUNT = 2000

    for i in range(5):

        round_victories = {
            "player_0": 0,
            "player_1": 0,
            "draws": 0
        }

        for i in range(GAME_COUNT):
            env.reset()

            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()

                if termination or truncation:
                    action = None
                else:
                    # this is where you would insert your policy
                    action, _, _ = agents[agent].act(observation)
                    action = env.decode_action(action)
                    #input("Press enter to continue...")

                #print(f"Agent {agent} took action {action}")
                env.step(action)
                #printout_state(env.get_player_number(agent), env.game.get_game_state())
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