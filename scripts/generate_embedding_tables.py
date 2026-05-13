import os

from src.agents.utils.card_embeddings import generate_embedding_table, get_embedding_table
from src.game.cards import card_lists
import numpy as np

os.makedirs("embedding_tables", exist_ok=True)

for card_list_name in ['base_deck', 'expanded_deck', 'simple_fluxx_deck']:
    card_list = getattr(card_lists, card_list_name)
    generate_embedding_table(card_list)
    np.save(f'embedding_tables/embedding_table_{card_list_name}.npy', get_embedding_table())
    print(f'Saved {card_list_name}')