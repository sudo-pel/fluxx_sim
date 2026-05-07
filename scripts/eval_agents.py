from pathlib import Path

import torch

from src.agents.DQNAgent import DQNAgent
from src.agents.HeuristicAgentMKI import HeuristicAgentMKI
from src.agents.HeuristicAgentMKII import HeuristicAgentMKII
from src.agents.PPOAgent import PPOAgent
from src.agents.PPOAgentGeneralized import PPOAgentGeneralized
from src.agents.RandomAgent import RandomAgent
from src.agents.card_embeddings import generate_embedding_table
from src.training.TrainingEnums import GameLogConfig
from src.env.AgentBattler import AgentBattler
from src.env.FluxxEnv import FluxxEnv
from src.game.Game import Game
from src.game.cards import card_lists

def format_matchup_grid(
    results: dict[tuple[str, str], dict[str, float]],
) -> str:
    """
    Format pairwise agent matchup results as a readable grid.

    Each cell shows the row agent's win rate against the column agent,
    plus the average game length in parentheses. Diagonal cells (self-play)
    are filled with "—".

    The function looks up each cell's stats by trying both directions:
        - results[(row, col)] with row as player_0 -> use player_wins["player_0"]
        - results[(col, row)] with row as player_1 -> use player_wins["player_1"]
    Whichever direction is present in `results` is used. If both are present,
    the (row, col) direction takes precedence.

    Args:
        results: Mapping from (agent_a, agent_b) tuples to a dict of stats.
            Each stats dict must contain:
                - "player_wins": dict with keys "player_0" and "player_1".
                  agent_a is treated as player_0 and agent_b as player_1.
                - "total_games": int, total games played in this matchup.
                - "average_game_length": float, mean game length.
            Draws (if any) are inferred as total_games - sum of wins.

    Returns:
        A multi-line string containing the formatted grid.

    Example:
        Matchup win rates (row vs column) — avg game length in parens
        ════════════════════════════════════════════════════════════════
                       │ DQN          │ Heuristic    │ Random       │
        ───────────────┼──────────────┼──────────────┼──────────────┤
        DQN            │ —            │ 64.0% (24.7) │ 94.0% (18.4) │
        Heuristic      │ 36.0% (24.7) │ —            │ 82.0% (21.2) │
        Random         │  6.0% (18.4) │ 18.0% (21.2) │ —            │
        ───────────────┴──────────────┴──────────────┴──────────────┘
    """
    if not results:
        return "No matchup data."

    # Collect all agent names from both sides of every pairing.
    agents = sorted({name for pair in results for name in pair})

    name_col_width = max(len("Agent"), max(len(a) for a in agents))
    cell_width = max(12, max(len(a) for a in agents))

    def lookup_cell(row: str, col: str) -> tuple[int, int, float] | None:
        """Return (row_wins, total_games, avg_game_length) for row vs col, or None."""
        # Try the (row, col) direction first: row is player_0
        if (row, col) in results:
            stats = results[(row, col)]
            return (
                stats["player_wins"]["player_0"],
                stats["total_games"],
                stats["average_game_length"],
            )
        # Fall back to the reverse: row is player_1 in results[(col, row)]
        if (col, row) in results:
            stats = results[(col, row)]
            return (
                stats["player_wins"]["player_1"],
                stats["total_games"],
                stats["average_game_length"],
            )
        return None

    def fmt_cell(row: str, col: str) -> str:
        if row == col:
            return "—".ljust(cell_width)
        looked_up = lookup_cell(row, col)
        if looked_up is None:
            return "—".center(cell_width)
        wins, total, avg_len = looked_up
        winrate = (wins / total) * 100 if total else 0.0
        text = f"{winrate:5.1f}% ({avg_len:.1f})"
        return text.ljust(cell_width)

    lines: list[str] = []

    # Title
    title = "Matchup win rates (row vs column) — avg game length in parens"
    lines.append(title)
    lines.append("═" * len(title))

    # Column header
    header = (
        " " * name_col_width
        + " │ " + " │ ".join(a.ljust(cell_width) for a in agents)
        + " │"
    )
    lines.append(header)

    # Separator
    sep = (
        "─" * name_col_width
        + "─┼─" + "─┼─".join("─" * cell_width for _ in agents)
        + "─┤"
    )
    lines.append(sep)

    # Body rows
    for row_agent in agents:
        cells = [fmt_cell(row_agent, col_agent) for col_agent in agents]
        line = (
            row_agent.ljust(name_col_width)
            + " │ " + " │ ".join(cells)
            + " │"
        )
        lines.append(line)

    # Bottom border
    bottom = (
        "─" * name_col_width
        + "─┴─" + "─┴─".join("─" * cell_width for _ in agents)
        + "─┘"
    )
    lines.append(bottom)

    return "\n".join(lines)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

generate_embedding_table(card_lists.base_deck)

two_player_fluxx = Game(2, card_lists.base_deck, disable_game_messages=True, logger=None)
env = FluxxEnv(two_player_fluxx, 2, render_mode="human")
agent_battler = AgentBattler(env)

# Prepare the agents.
agents = {
    "ppo": PPOAgent(env.game.game_config, 0),
    "ppo_general": PPOAgentGeneralized(env.game.game_config, 0),
    "dqn": DQNAgent(env.game.game_config, 0),
    "dqn_general": None,
    "ppo_general_with_reward_shaping": None,
    "random": RandomAgent(env.game.game_config, 0),
    "heuristic_agent_mki": HeuristicAgentMKI(env.game.game_config, 0),
    "heuristic_agent_mkii": HeuristicAgentMKII(env.game.game_config, 0),
}

# Load the trained state dicts.
agents["ppo"].policy_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/ppo_2026-04-28_17-11-02/final/final_model_50004751.pt"))
agents["ppo"].policy_network.eval()

# strict=False because card embeds was a part of state_dict when this code was run
agents["ppo_general"].policy_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/ppo_general_2026-05-02_09-11-14/models/model_42008245.pt"), strict=False)
agents["ppo_general"].policy_network.eval()

agents["dqn"].q_network.load_state_dict(torch.load(f"{PROJECT_ROOT}/experiments/dqn_2026-04-29_08-06-30/final/final_model_50000050.pt"))
agents["dqn"].q_network.eval()

# TODO: dqn_general and ppo_general_with_reward_shaping

# (Temporary) filter out uninitialized agents
agent_names = [a for a in agents.keys()]
for agent_name in agent_names:
    if agents[agent_name] is None:
        del agents[agent_name]

results: dict[tuple[str, str], dict[str, float]] = {}

for agent_name, agent in agents.items():
    for other_agent_name, other_agent in agents.items():
        if agent_name == other_agent_name: continue
        print(f"RUNNING {agent_name} vs {other_agent_name}")
        agent.player_number = 0
        other_agent.player_number = 1
        results[(agent_name, other_agent_name)] = agent_battler.run_games([agent, other_agent], 10000, 50000, log_games=False)
        print(f"RESULTS: {results[(agent_name, other_agent_name)]}")

print(format_matchup_grid(results))