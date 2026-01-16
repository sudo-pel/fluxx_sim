from fluxx.Agents.Agent import Agent

class PlayerControlledAgent(Agent):
    def play_free_action(self, game_state):
        pass

    def play_card(self, game_state):
        print("---GOALS IN PLAY---")
        print("\n".join([g.name for g in game_state.goals]))

        print("---RULES IN PLAY---")
        print("\n".join([r.name for r in game_state.rules]))

        print("---KEEPERS IN PLAY---")
        for i, player in enumerate(game_state.players):
            string = f"Player {player}"
            if i == game_state.player_turn:
                string += " (you)"
            print(string)
            print("\n".join([k.name for k in player.keepers]))

        return int(input("Please choose a card to play >> "))

    def discard_keeper(self, game_state):
        pass

    def discard_from_hand(self, game_state):
        pass