from Agents.Agent import Agent

class PlayerControlledAgent(Agent):
    def play_free_action(self, game_state):
        pass

    def play_card(self, game_state):
        self.printout_state(game_state)
        return int(input("Please choose a card to play >> "))

    def discard_keeper(self, game_state):
        self.printout_state(game_state)
        return int(input("Please choose a KEEPER to discard >>"))

    def discard_from_hand(self, game_state):
        self.printout_state(game_state)
        return int(input("Please choose a CARD to discard from HAND >>"))

    def printout_state(self, game_state):
        print("\n\n---GOALS IN PLAY---")
        # print("\n".join([g.name for g in game_state["goal"]]))
        string = "None"
        if game_state["goal"]:
            string = game_state["goal"].name
        print(string)

        print("---RULES IN PLAY---")
        print("\n".join([r.name for r in game_state["rules"]]))

        print("---KEEPERS IN PLAY---")
        for i, player in enumerate(game_state["players"]):
            string = f"-Player {i}"
            if i == self.player_number:
                string += " (you)-"
            else:
                string += "-"
            print(string)
            print("\n".join([k.name for k in player.keepers]))

        print("---CARDS IN HAND---")
        print("\n".join([c.name for c in game_state["players"][self.player_number].hand]))

