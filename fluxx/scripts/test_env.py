from agents.RandomAgent import RandomAgent
from fluxx.env.FluxxEnv import FluxxEnv
from fluxx.game.Cards import card_lists
from fluxx.game.Game import Game

agents = {
    "player_0": RandomAgent(),
    "player_1": RandomAgent()
}

def main():
    two_player_simple_fluxx = Game(2, card_lists.simple_fluxx_deck)
    env = FluxxEnv(two_player_simple_fluxx, 2, render_mode="human")
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = agents[agent].act(observation)
            action = env.decode_action(action)
            print(f"Agent {agent} took action {action}")

        env.step(action)
    env.close()

if __name__ == '__main__':
    main()