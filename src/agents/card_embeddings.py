"""
Card embedding construction

Each card gets a 3860-dim embedding built from:
    type one-hot (4) -- KEEPER, GOAL, ACTION, RULE
    + cached name embedding (768)
    + goal-only: requisite/disallowed/optional keeper name pools (768 each)
    + rule/action-only: effect text embedding (768) + hand-defined axes (16)
    Total: 4 + 768 + 768 * 3 + 768 + 16 = 3860

Embeddings for all cards are precomputed once at startup (build_card_embedding_table) ...
... and looked up by card name during rollout / training.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from src.game.cards.card_data import CARD_DATA


model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', truncate_dim=64)
def model_encode(text: str) -> np.ndarray:
    return model.encode(f"clustering: {text}", convert_to_numpy=True)


CARD_TYPES = ["KEEPER", "GOAL", "ACTION", "RULE"]
CARD_TYPE_TO_IDX = {t: i for i, t in enumerate(CARD_TYPES)}

NAME_EMBED_DIM = 64
EFFECT_TEXT_DIM = 64
HAND_AXES_DIM = 16
EFFECT_BLOCK_DIM = EFFECT_TEXT_DIM + HAND_AXES_DIM

TYPE_DIM = len(CARD_TYPES)
GOAL_KEEPER_BLOCK_DIM = NAME_EMBED_DIM * 3  # requisite + disallowed + optional
CARD_EMBED_DIM = TYPE_DIM + NAME_EMBED_DIM + GOAL_KEEPER_BLOCK_DIM + EFFECT_BLOCK_DIM
assert CARD_EMBED_DIM == 340

def convert_card_name(card_name: str) -> str:
    return card_name.replace(" ", "_")

def build_card_embedding(
    card_name: str,
) -> np.ndarray:
    """
    Build the 3860-dim embedding for a single card. Slots that don't apply for the card's type are zeroed.
    """
    embedding = np.zeros(CARD_EMBED_DIM, dtype=np.float32)
    offset = 0

    # Type one-hot (4)
    card_type = CARD_DATA[card_name]["card_type"]
    embedding[offset + CARD_TYPE_TO_IDX[card_type]] = 1.0
    offset += TYPE_DIM

    # Name embedding
    embedding[offset : offset + NAME_EMBED_DIM] = model_encode(card_name)
    offset += NAME_EMBED_DIM

    # Keeper pools (goal only)
    if card_type == "GOAL":
        for keeper_field in ("required_keepers", "disallowed_keepers", "optional_keepers"):
            keeper_names = CARD_DATA.get(keeper_field, [])
            if keeper_names:
                pooled = np.sum(
                    [model_encode(convert_card_name(k)) for k in keeper_names], axis=0
                )
                embedding[offset : offset + NAME_EMBED_DIM] = pooled
            offset += NAME_EMBED_DIM
    else:
        offset += GOAL_KEEPER_BLOCK_DIM

    # Effect text + hand-defined axes (action/rules only)
    if card_type in ("ACTION", "RULE"):
        embedding[offset : offset + EFFECT_TEXT_DIM] = model_encode(CARD_DATA[card_name]["card_effect"])
        offset += EFFECT_TEXT_DIM
        embedding[offset : offset + HAND_AXES_DIM] = CARD_DATA[card_name]["effect_parameters"]
        offset += HAND_AXES_DIM
    else:
        offset += EFFECT_BLOCK_DIM

    assert offset == CARD_EMBED_DIM
    return embedding


def build_card_embedding_table(
    card_list: list[str],
) -> dict[str, np.ndarray]:
    return {
        name: build_card_embedding(
            name,
        )
        for name in card_list
    }

_EMBEDDING_TABLE: dict[str, np.ndarray] | None = None

def generate_embedding_table(card_list: list[str]) -> None:
    global _EMBEDDING_TABLE
    _EMBEDDING_TABLE = {card: build_card_embedding(card) for card in card_list}


def get_embedding_table() -> dict[str, np.ndarray]:
    if _EMBEDDING_TABLE is None:
        raise RuntimeError(
            "Embedding table not loaded. Call load_embedding_table(path) "
            "or set_embedding_table(table) before constructing agents."
        )
    return _EMBEDDING_TABLE