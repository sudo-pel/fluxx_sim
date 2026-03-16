### Environment-game interface
`Game` stores players as integers, `FluxxEnv` stores players as strings according to **PettingZoo's** conventions: `player_[PLAYER_NO]`

### Storage and passing of cards
Cards are passed around as `string`s. Game actions assumed validated inputs: validation is handled in `Game.step()`. When a function needs to access a card within a zone, it searches that zone for the name of the card (a lambda is needed here to compare the `name` field of the `Card` object against the `string` argument) and then operates upon that index accordingly.


> This is the most sensible way to store cards; agents receive state as a one-hot encoding and therefore do not know the order of cards in the hand. They will return a number from their actions, which can be decoded into a card name using the map stored inside the environment. This can then be converted into an index **whenever appropriate**. This is a slight inefficiency that may be worth fixing in the future.

### Storage and passing of players
Players are referred to by `player_number`. Functions get the `Player` object by indexing `Game.players`.

