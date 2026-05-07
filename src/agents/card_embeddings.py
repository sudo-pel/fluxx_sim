"""
Card embedding construction

Each card gets a 340-dim embedding built from:
    type one-hot (4) -- KEEPER, GOAL, ACTION, RULE
    + cached name embedding (64)
    + goal-only: requisite/disallowed/optional keeper name pools (64 each)
    + rule/action-only: effect text embedding (64) + hand-defined axes (16)
    Total: 4 + 64 + 64 * 3 + 64 + 16 = 340

Embeddings for all cards are precomputed once at startup (generate_embedding_table) ...
... and looked up by card name during rollout / training.
"""

from __future__ import annotations
import numpy as np
import torch  # CHANGED: added for tensor support
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

PAD_ID = 0


def convert_card_name(card_name: str) -> str:
    return card_name.replace(" ", "_")

def build_card_embedding(
    card_name: str,
) -> np.ndarray:
    """
    Build the 340-dim embedding for a single card. Slots that don't apply for the card's type are zeroed.
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
_CARD_NAME_TO_ID: dict[str, int] | None = None
_EMBEDDING_TENSOR_CPU: torch.Tensor | None = None
_EMBEDDING_TENSOR_BY_DEVICE: dict[torch.device, torch.Tensor] = {}


def generate_embedding_table(card_list: list[str]) -> None:
    """
    Build the embedding dict, the name->id map, and the stacked tensor in one pass.
    Index 0 is reserved as a zero-vector pad row; real cards have ids 1..N.
    """
    global _EMBEDDING_TABLE, _CARD_NAME_TO_ID, _EMBEDDING_TENSOR_CPU, _EMBEDDING_TENSOR_BY_DEVICE

    _EMBEDDING_TABLE = {card: build_card_embedding(card) for card in card_list}

    # Real cards get ids 1..N. Index 0 is the pad row
    _CARD_NAME_TO_ID = {card: i + 1 for i, card in enumerate(card_list)}

    # Stack into a single contiguous (N+1, CARD_EMBED_DIM) tensor
    # Row 0 is the zero pad embedding; rows 1..N are real card embeddings in card_list order
    pad_row = np.zeros((1, CARD_EMBED_DIM), dtype=np.float32)
    real_rows = np.stack([_EMBEDDING_TABLE[card] for card in card_list], axis=0)
    stacked = np.concatenate([pad_row, real_rows], axis=0)
    _EMBEDDING_TENSOR_CPU = torch.from_numpy(stacked)

    # Reset GPU cache so a stale device tensor isn't reused after a rebuild
    _EMBEDDING_TENSOR_BY_DEVICE = {}


def get_embedding_table() -> dict[str, np.ndarray]:
    if _EMBEDDING_TABLE is None:
        raise RuntimeError(
            "Embedding table not loaded. Call generate_embedding_table(card_list) "
            "before constructing agents."
        )
    return _EMBEDDING_TABLE

def get_card_id_map() -> dict[str, int]:
    """Returns the {card_name: id} map. Ids are 1-indexed; 0 is reserved for padding."""
    if _CARD_NAME_TO_ID is None:
        raise RuntimeError(
            "Embedding table not loaded. Call generate_embedding_table(card_list) "
            "before constructing agents."
        )
    return _CARD_NAME_TO_ID


def get_embedding_tensor(device: torch.device) -> torch.Tensor:
    """
    Returns the stacked embedding tensor on the requested device.
    Cached per-device so the CPU->GPU transfer happens only once per device.
    Shape: (N_cards + 1, CARD_EMBED_DIM). Row 0 is the pad embedding.
    """
    if _EMBEDDING_TENSOR_CPU is None:
        raise RuntimeError(
            "Embedding table not loaded. Call generate_embedding_table(card_list) "
            "before constructing agents."
        )
    if device not in _EMBEDDING_TENSOR_BY_DEVICE:
        _EMBEDDING_TENSOR_BY_DEVICE[device] = _EMBEDDING_TENSOR_CPU.to(device)
    return _EMBEDDING_TENSOR_BY_DEVICE[device]